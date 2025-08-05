import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import yfinance as yf
import ta
from scipy.stats import randint, uniform
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

"""
XGB 2025 - v1.0

Data da modificação:

Modificações:
    - v1.0 -> Definição target

"""


def apagar_arquivos_antigos():
    """
    Função que elimina os arquivos antigos.
    """

    print("#######################################\n")
    arquivos = [
        "../data/processed/top_features.txt",
        "../data/processed/dados_para_backtesting.csv",
        "../data/processed/falsospositivos.csv",
        "../data/processed/verdadeirospostivos.csv",
        "../data/processed/cross_validation_results.csv",
        "../data/image/shap_importance.png",
        "../data/image/backtest_plot.png",
        "../data/image/confusionmatix.png",
        "../data/image/Beeswarm_plot_shap_importance.png",
        "../data/image/Global_bar_plot_shap_importance.png",
        "../data/image/Local_bar_plot_shap_importance.png",
        "../models/modelo_xgb.pkl",
        "../data/raw/treino.csv",
        "../data/raw/validacao.csv",
        "../data/raw/teste.csv",
    ]
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            os.remove(arquivo)
            print(f"Arquivo removido: {arquivo}")


def baixar_dados(ticker, anos=20):
    """
    Função que baixa os dados do Yahoo Finance.
    """

    fim = pd.Timestamp.today()
    inicio = fim - pd.DateOffset(years=anos)
    df = yf.download(ticker, start=inicio, end=fim)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def preparar_dados(df):
    """
    Função para criar as variáveis dependentes e a variável independente(Target)
    """

    df = df.copy()

    df["future_return"] = (
        df["Close"].pct_change(10).shift(-10)
    )  # Target: retorno futuro em 10 dias
    df["target"] = (df["future_return"] > 0).astype(int)  # Target: Binario

    # Variáveis baseados no preço
    df["volatility"] = df["Close"].rolling(20).std()

    for col in ["volatility"]:
        for i in range(1, 21):
            df[f"{col}_lag{i}"] = df[col].shift(i)
    """
    # Variáveis de Indicadores Técnicos
    df["RSI14"] = ta.momentum.rsi(df["Close"], window=14)

    indicator_atr14 = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14,
    )

    indicator_atr50 = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=50,
    )

    df["ATR14"] = indicator_atr14.average_true_range()
    df["ATR50"] = indicator_atr50.average_true_range()
    df['REASON_ATR'] = df['ATR14'] / df['ATR50']
    df['Z-Score_ATR14'] = (df['ATR14'] - df['ATR14'].mean()) / df['ATR14'].std()
    df['Z-Score_ATR50'] = (df['ATR50'] - df['ATR50'].mean()) / df['ATR50'].std()
    """

    columns = df.columns.tolist()
    df.dropna(subset=columns, inplace=True)  #
    df = df.replace([np.inf, -np.inf], np.nan).dropna()  #
    return df


def pegar_colunas(df):
    """
    Função que elimina as colunas que não serão utilizadas no modelo.
    """

    return df.columns.drop(
        [
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
            "future_return",
            "target",
            "volatility",
        ]
    ).tolist()


def separar_dados_temporais(df):
    """
    Função que separa os dados temporais em treino, teste e validação.
    """

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    TREINO = df[(df.index >= "2005-01-01") & (df.index < "2019-01-01")]
    TESTE = df[(df.index >= "2019-01-01") & (df.index < "2022-01-01")]
    VALIDACAO = df[df.index >= "2022-01-01"]
    return TREINO, TESTE, VALIDACAO


def carregar_ou_baixar_dados_divididos(ticker, anos=20):
    """
    Função que carrega os dados brutos e os separa em treino, teste e validação.
    """

    caminho_dados_brutos = "../data/raw/dados_brutos.csv"
    caminho_treino = "../data/raw/treino.csv"
    caminho_teste = "../data/raw/teste.csv"
    caminho_valid = "../data/raw/validacao.csv"

    os.makedirs("../data/raw", exist_ok=True)

    if all(os.path.exists(p) for p in [caminho_treino, caminho_valid, caminho_teste]):
        print("✅ Arquivos existentes encontrados. Carregando dados salvos...")
        df_treino = pd.read_csv(caminho_treino, index_col=0, parse_dates=True)
        df_teste = pd.read_csv(caminho_teste, index_col=0, parse_dates=True)
        df_valid = pd.read_csv(caminho_valid, index_col=0, parse_dates=True)
        return df_treino, df_teste, df_valid

    if os.path.exists(caminho_dados_brutos):
        print("📄 Carregando dados brutos existentes...")
        df_raw = pd.read_csv(caminho_dados_brutos, index_col=0, parse_dates=True)
    else:
        print("⬇️ Baixando dados brutos do Yahoo Finance...")
        df_raw = baixar_dados(ticker, anos)
        df_raw.to_csv(caminho_dados_brutos)
        print("✅ Dados brutos salvos em:", caminho_dados_brutos)

    print("\n⚙️ Preparando e separando dados...")
    df_preparado = preparar_dados(df_raw)
    df_treino, df_teste, df_valid = separar_dados_temporais(df_preparado)
    df_treino.to_csv(caminho_treino)
    df_teste.to_csv(caminho_teste)
    df_valid.to_csv(caminho_valid)

    print("\n✅ Dados de Treino, Teste e Validação salvos.")
    print("\n#######################################")
    return df_treino, df_teste, df_valid


def modelo_top_features(df_treino):
    """
    Função com modelo usado para para selecionar as Top features.
    Utiliza RandomizedSearchCV para encontrar os melhores parâmetros do modelo XGBClassifier.

    df_treino: Dataframe de treino
    """

    print("\n🔍 Selecionando as Top Features...")
    features = pegar_colunas(df_treino)

    SEED = 42
    np.random.seed(SEED)

    X = df_treino[features]
    y = df_treino["target"]

    model = XGBClassifier(
        eval_metric=["logloss", "aucpr"],
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=1,
        random_state=SEED,
    )

    param_grid = {
        "n_estimators": randint(100, 500),
        # "max_depth": randint(3, 5),
        # "learning_rate": uniform(0.01, 0.05),
        # "subsample": uniform(0.8, 0.2),
        # "colsample_bytree": uniform(0.8, 0.2),
        # "gamma": uniform(0, 0.5),
        # "min_child_weight": randint(1, 3),
        # "reg_alpha": uniform(1, 1.5),
        # "reg_lambda": uniform(1, 1.5),
    }

    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=1,
        random_state=SEED,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_


def selecionar_melhores_features(
    modelo,
    X,
    top_n=int,
    salvar_em="../data/processed/top_features.txt",
    salvar_grafico="../data/image/",
):
    """
    Função que seleciona as top features mais importantes.
    Utiliza SHAP para calcular a importância das features.
    """

    print("\n🔍 Calculando valores SHAP...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X)

    # Calcula importância média absoluta
    if len(shap_values.shape) == 3:  # Para multi-classes
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 1))
    else:  # Para binário/regressão
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

    shap_importance = pd.Series(mean_abs_shap, index=X.columns)
    shap_importance = shap_importance.sort_values(ascending=False)
    top_features = shap_importance.head(top_n)

    # Salva top features
    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
    with open(salvar_em, "w", encoding="utf-8") as f:
        f.write("\n".join(top_features.index))
    print(f"\n✅ Top {top_n} features SHAP salvas em: {salvar_em}")

    # Cria e salva o gráfico
    os.makedirs(os.path.dirname(salvar_grafico), exist_ok=True)
    plt.figure()

    # Global Bar plot
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    plt.title(f"Top {top_n} Features - SHAP Importance (Global)", pad=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(salvar_grafico, "Global_bar_plot_shap_importance.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Configurações iniciais
    top_n = top_n
    plt.figure(figsize=(10, 6))  # Tamanho da figura

    # Gerar o gráfico beeswarm
    shap.plots.beeswarm(
        shap_values,
        max_display=top_n,
        show=False,
        color=plt.get_cmap("coolwarm"),  # Gradiente azul (baixo) -> vermelho (alto)
        alpha=0.7,  # Transparência para reduzir sobreposição de pontos
    )

    # Personalizar títulos e eixos
    plt.title(f"Top {top_n} Features - SHAP Importance", fontsize=14, pad=20)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    # Ajustar rótulos das features (remover caracteres inválidos)
    ax = plt.gca()
    labels = [
        label.get_text()
        .replace("_", " ")
        .replace("@", "")
        .replace("lag", "lag ")
        .strip()
        for label in ax.get_yticklabels()
    ]
    ax.set_yticklabels(labels, fontsize=12)

    # Adicionar legenda de cores manualmente (se necessário)
    import matplotlib.patches as mpatches

    low_patch = mpatches.Patch(color="blue", label="Low Value")
    high_patch = mpatches.Patch(color="red", label="High Value")
    plt.legend(handles=[low_patch, high_patch], loc="upper right", fontsize=8)

    # Ajustar layout e salvar
    plt.tight_layout()
    plt.savefig(
        os.path.join(salvar_grafico, "Beeswarm_plot_shap_importance.png"),
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )
    plt.close()

    return top_features.index.tolist()


def tunar_modelo(X_train, y_train):
    """
    Função que encontra os melhores parâmetros para o modelo.
    Utiliza Optuna para otimização de hiperparâmetros.
    A função treina um modelo XGBClassifier com os melhores parâmetros encontrados.
    Retorna o modelo treinado.

    X_train: DataFrame com as features de treino.
    y_train: Series com os rótulos de treino.
    """

    print("\n🔍 Tunando modelo...")
    X = X_train
    y = y_train

    def objective(trial):

        print("\n🔍 Tunando modelo...")
        xgb_params = {
            "eta": trial.suggest_float("xgb_eta", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 5),
            "learning_rate": trial.suggest_float(
                "xgb_learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("xgb_gamma", 0, 1.0),
            "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 3),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0, 2.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0, 2.0),
        }

        model = XGBClassifier(
            **xgb_params,
            eval_metric=["logloss", "aucpr"],
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )

        score = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=1,
        ).mean()

        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n🔍 Melhores hiperparâmetros encontrados:")
    print(study.best_params)

    best_params = study.best_params

    final_model = XGBClassifier(
        eta=best_params["xgb_eta"],
        n_estimators=best_params["xgb_n_estimators"],
        max_depth=best_params["xgb_max_depth"],
        learning_rate=best_params["xgb_learning_rate"],
        subsample=best_params["xgb_subsample"],
        colsample_bytree=best_params["xgb_colsample_bytree"],
        gamma=best_params["xgb_gamma"],
        min_child_weight=best_params["xgb_min_child_weight"],
        reg_alpha=best_params["xgb_reg_alpha"],
        reg_lambda=best_params["xgb_reg_lambda"],
        eval_metric=["logloss", "aucpr"],
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )

    final_model.fit(X, y)

    print("\n✅ Modelo final treinado com sucesso!")

    return final_model


def treinar_e_salvar_modelo(
    df_treino,
    df_teste,
    modelo_path="../models/modelo_xgb.pkl",
    top_features_path="../data/processed/top_features.txt",
):
    """
    Função que treina o modelo e salva o modelo e as features selecionadas.
    Recebe os DataFrames de treino e teste.
    Salva o modelo treinado e as top features em arquivos especificados.
    """

    # Pega todas as features originais
    features = pegar_colunas(df_treino)

    # Treina o modelo com todas as features
    modelo_features = modelo_top_features(df_treino)

    # Dados de validação para calcular SHAP
    X_train = df_treino[features]
    y_train = df_treino["target"]

    # Gera SHAP e seleciona top features
    top_features = selecionar_melhores_features(
        modelo_features, X_train, top_n=50, salvar_em=top_features_path
    )

    # Exclui algumas features baseado nos falsos positivos e verdadeiros positivos
    ######################## NEW ########################
    features_to_remove = []
    top_features = [feat for feat in top_features if feat not in features_to_remove]

    os.makedirs(os.path.dirname("../data/processed/top_features.txt"), exist_ok=True)

    with open("../data/processed/top_features.txt", "w") as f:
        for feat in top_features:
            f.write(feat + "\n")
    ######################## NEW ########################

    # Treinar um novo modelo só com as top features
    X_train = df_treino[top_features]
    y_train = df_treino["target"]

    modelo = tunar_modelo(X_train, y_train)

    print("\n✅ Modelo treinado com as melhores features.")

    X_test = df_teste[top_features]
    y_test = df_teste["target"]
    y_test_pred = modelo.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    print(f"\n📊 Acurácia teste final (modelo top features): {acc_test:.2%}")
    pre_test = precision_score(y_test, y_test_pred)
    print(f"📊 Precisão teste final (modelo top features): {pre_test:.2%}")
    rec_test = recall_score(y_test, y_test_pred)
    print(f"📊 Recall teste final (modelo top features): {rec_test:.2%}")
    fi_test = f1_score(y_test, y_test_pred)
    print(f"📊 F1-Score teste final (modelo top features): {fi_test:.2%}")

    print("\n🧾 Matriz de Confusão (teste final):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\n📄 Relatório de Classificação (teste final):")
    print(classification_report(y_test, y_test_pred))

    # Salva modelo
    os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
    joblib.dump(modelo, modelo_path)
    print(f"\n✅ Modelo final salvo em: {modelo_path}")

    # Falsos positivos
    fp_mask = (y_test == 0) & (y_test_pred == 1)
    fp_data = X_test[fp_mask].copy()
    fp_data.to_csv("../data/processed/falsospositivos.csv")

    # Verdadeiros positivos
    vp_mask = (y_test == 1) & (y_test_pred == 1)
    vp_data = X_test[vp_mask].copy()
    vp_data.to_csv("../data/processed/verdadeirospostivos.csv")

    # Matriz de confusão normalizada por predição
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_test_pred,
        display_labels=["Negativo", "Positivo"],
        cmap="Blues",
    )
    disp.ax_.set_title("Matriz de Confusão Normalizada por Predição")
    plt.savefig("../data/image/confusionmatix.png")


def main():
    """
    Função principal que executa o fluxo de trabalho do script.
    """

    apagar_arquivos_antigos()
    print("🗑️ Arquivos antigos removidos.\n")
    ticker = "PETR4.SA"
    (
        df_treino,
        df_teste,
        df_validacao,
    ) = carregar_ou_baixar_dados_divididos(ticker)
    treinar_e_salvar_modelo(df_treino, df_teste)
    df_validacao.to_csv("../data/processed/dados_para_backtesting.csv")
    print("\n📁 Dados para backtesting em: data/processed/dados_para_backtesting.csv")


if __name__ == "__main__":
    main()
