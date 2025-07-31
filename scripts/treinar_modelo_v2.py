import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yfinance as yf
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def apagar_arquivos_antigos():
    caminhos = [
        "../data/processed/top_features.txt",
        "../data/processed/dados_para_backtesting.csv",
        "../data/processed/falsospositivos.csv",
        "../data/processed/verdadeirospostivos.csv",
        "../data/processed/cross_validation_results.csv",
        "../data/image/shap_importance.png",
        "../data/image/backtest_plot.png",
        "../data/image/confusionmatix.png",
        "../models/modelo_xgb.pkl",
        "../data/raw/treino.csv",
        "../data/raw/validacao.csv",
        "../data/raw/teste.csv",
    ]
    for caminho in caminhos:
        if os.path.exists(caminho):
            os.remove(caminho)
            print(f"Arquivo removido: {caminho}")


def baixar_dados(ticker, anos=20):
    fim = pd.Timestamp.today()
    inicio = fim - pd.DateOffset(years=anos)
    df = yf.download(ticker, start=inicio, end=fim)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def preparar_dados(df):
    df = df.copy()
    df["future_return"] = df["Close"].pct_change(10).shift(-10)
    df["target"] = (df["future_return"] > 0).astype(int)
    df["return"] = df["Close"].pct_change(5)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()
    df["close_vs_high"] = df["Close"] / df["High"].rolling(20).max()
    df["close_vs_low"] = df["Close"] / df["Low"].rolling(20).min()

    for col in ["return", "volatility", "close_vs_high", "close_vs_low"]:
        for i in range(1, 21):
            df[f"{col}_lag{i}"] = df[col].shift(i)

    return df.dropna()


def pegar_colunas(df):
    colunas_excluir = [
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        "future_return",
        "target",
        "return",
        "volatility",
        "close_vs_high",
        "close_vs_low",
    ]
    return [col for col in df.columns if col not in colunas_excluir]


def separar_dados_temporais(df):
    df.index = pd.to_datetime(df.index)
    return (
        df[(df.index >= "2005-01-01") & (df.index < "2019-01-01")],
        df[(df.index >= "2019-01-01") & (df.index < "2022-01-01")],
        df[df.index >= "2022-01-01"],
    )


def carregar_ou_baixar_dados_divididos(ticker, anos=20):
    caminhos = {
        "brutos": "../data/raw/dados_brutos.csv",
        "treino": "../data/raw/treino.csv",
        "teste": "../data/raw/teste.csv",
        "valid": "../data/raw/validacao.csv",
    }
    os.makedirs("../data/raw", exist_ok=True)
    if all(os.path.exists(p) for p in caminhos.values() if "brutos" not in p):
        print("✅ Arquivos existentes encontrados. Carregando dados salvos...")
        return tuple(
            pd.read_csv(p, index_col=0, parse_dates=True)
            for k, p in caminhos.items()
            if k != "brutos"
        )

    df_raw = (
        pd.read_csv(caminhos["brutos"], index_col=0, parse_dates=True)
        if os.path.exists(caminhos["brutos"])
        else baixar_dados(ticker, anos)
    )
    if not os.path.exists(caminhos["brutos"]):
        df_raw.to_csv(caminhos["brutos"])

    df_prep = preparar_dados(df_raw)
    df_treino, df_teste, df_valid = separar_dados_temporais(df_prep)
    df_treino.to_csv(caminhos["treino"])
    df_teste.to_csv(caminhos["teste"])
    df_valid.to_csv(caminhos["valid"])

    print("\n✅ Treino, validação e teste salvos.")
    return df_treino, df_teste, df_valid


def tunar_modelo(df_treino):
    X = df_treino[pegar_colunas(df_treino)]
    y = df_treino["target"]

    base_model = XGBClassifier(
        eval_metric=["logloss", "aucpr"],
        objective="binary:logistic",
        tree_method="hist",
        use_label_encoder=False,
        n_jobs=1,
        random_state=42,
    )
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.5],
        "min_child_weight": [1, 3],
        "reg_alpha": [1, 1.5],
        "reg_lambda": [1, 1.5],
    }
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=50,
        cv=10,
        scoring="accuracy",
        random_state=42,
        n_jobs=1,
    )
    search.fit(X, y)

    stack = StackingClassifier(
        estimators=[("mod", DecisionTreeClassifier(random_state=42))],
        final_estimator=search.best_estimator_,
        cv=10,
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=True,
    )
    stack.fit(X, y)
    return stack


def selecionar_melhores_features(
    modelo,
    X,
    top_n=50,
    salvar_em="../data/processed/top_features.txt",
    salvar_grafico="../data/image/shap_importance.png",
):

    print("\n🔍 Calculando valores SHAP...")
    shap_values = shap.TreeExplainer(modelo).shap_values(X)
    importancias = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    top = importancias.head(top_n)

    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
    top.to_csv(salvar_em, index=True, header=False)

    plt.figure(figsize=(10, 8))
    top[::-1].plot(kind="barh")
    plt.title(f"Top {top_n} Features - SHAP Importance")
    plt.tight_layout()
    plt.savefig(salvar_grafico)
    plt.close()

    return top.index.tolist()


def treinar_e_salvar_modelo(df_treino, df_teste):
    modelo = tunar_modelo(df_treino)
    X_train = df_treino[pegar_colunas(df_treino)]
    y_train = df_treino["target"]
    top_features = selecionar_melhores_features(modelo, X_train)

    X_train = df_treino[top_features]

    modelo_final = XGBClassifier(**modelo.get_params())
    modelo_final.fit(X_train, y_train)

    X_test = df_teste[top_features]
    y_test = df_teste["target"]
    y_pred = modelo_final.predict(X_test)

    print("\n📊 Métricas de Teste:")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precisão: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred):.2%}")
    print(f"F1-score: {f1_score(y_test, y_pred):.2%}")

    print("\n🧾 Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📄 Relatório:")
    print(classification_report(y_test, y_pred))

    joblib.dump(modelo_final, "../models/modelo_xgb.pkl")

    X_test[(y_test == 0) & (y_pred == 1)].to_csv(
        "../data/processed/falsospositivos.csv"
    )
    X_test[(y_test == 1) & (y_pred == 1)].to_csv(
        "../data/processed/verdadeirospostivos.csv"
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Negativo", "Positivo"], cmap="Blues"
    ).figure_.savefig("../data/image/confusionmatix.png")


def main():
    apagar_arquivos_antigos()
    ticker = "PETR4.SA"
    treino, teste, validacao = carregar_ou_baixar_dados_divididos(ticker)
    treinar_e_salvar_modelo(treino, teste)
    validacao.to_csv("../data/processed/dados_para_backtesting.csv")


if __name__ == "__main__":
    main()
