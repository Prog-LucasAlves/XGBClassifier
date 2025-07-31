import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore")


def baixar_dados(ticker, anos=8):
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

    df["return"] = df["Close"].shift(-7) / df["Close"] - 1
    df["target"] = (df["return"] > 0.015).astype(int)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()

    for i in range(1, 5 + 1):
        df[f"return_lag{i}"] = df["Close"].pct_change().shift(i)
        df[f"volatility_lag{i}"] = df["volatility"].shift(i)
        df[f"momentum_lag{i}"] = np.log(df["Close"] / df["Close"].shift(i))

    df.interpolate(inplace=True)
    df.dropna(inplace=True)

    return df


def pegar_colunas(df):
    return df.columns.drop(
        ["Close", "High", "Low", "Open", "Volume", "return", "volatility"]
    ).tolist()


def validar_dados_atuais(df):
    hoje = pd.Timestamp.today().normalize()
    ultimo_dado = df.index[-1].normalize()

    if ultimo_dado < hoje:
        print(f"⚠️ Dados mais recentes disponíveis: {ultimo_dado.date()}")
        print(
            f"📅 Hoje é {hoje.date(1)}. O mercado ainda não fechou ou os dados ainda não foram atualizados."
        )
        print(
            "⏳ Aguardando atualização do Yahoo Finance. Tente rodar novamente mais tarde."
        )
        sys.exit()
    else:
        print(f"✅ Dados atualizados! Último pregão: {ultimo_dado.date()}")


def prever_proximo_movimento(df, modelo_path="models/production/modelo_xgb.pkl"):
    features = pegar_colunas(df)

    modelo = joblib.load(modelo_path)

    ultima_linha = df[features].iloc[-1:]
    previsao = modelo.predict(ultima_linha)[0]
    proba = modelo.predict_proba(ultima_linha)[0][1]

    data_previsao = df.index[-1]
    data_entrada = (data_previsao + BDay(1)).date()
    data_saida = (data_previsao + BDay(7)).date()

    print("\n🧠 Previsão do modelo:")
    if previsao == 1:
        print(
            f"📈 Sinal de COMPRA gerado para {data_entrada} com saída prevista em {data_saida}"
        )
        print(f"🔢 Probabilidade de alta: {proba:.2%}")
    else:
        print(
            f"📉 Sinal de NÃO COMPRA para {data_entrada} (modelo prevê queda/lateralização)"
        )
        print(f"🔢 Probabilidade de alta: {proba:.2%}")

    return {
        "ticker": "PETR4.SA",
        "data_previsão": str(data_previsao),
        "data_entrada": str(data_entrada),
        "data_saida": str(data_saida),
        "sinal": "COMPRA" if previsao == 1 else "NÃO COMPRAR",
        "probabilidade_alta": round(proba * 100, 2),
    }


def main():
    ticker = "PETR4.SA"
    df = baixar_dados(ticker)
    df = preparar_dados(df)
    validar_dados_atuais(df)

    resultado = prever_proximo_movimento(df)

    os.makedirs("data/production", exist_ok=True)
    previsao_atual_path = "data/production/previsao_atual.csv"
    pd.DataFrame([resultado]).to_csv(previsao_atual_path, index=False)
    print(f"\n✅ Previsão salva em: {previsao_atual_path}")

    # Adiciona ao histórico
    historico_path = "data/production/previsoes_historicas.csv"
    resultado["realizado"] = None  # <- campo a ser preenchido futuramente

    if os.path.exists(historico_path):
        historico = pd.read_csv(historico_path)
        historico = pd.concat([historico, pd.DataFrame([resultado])], ignore_index=True)
    else:
        historico = pd.DataFrame([resultado])

    historico.to_csv(historico_path, index=False)
    print(f"📈 Histórico atualizado em: {historico_path}")


if __name__ == "__main__":
    main()
