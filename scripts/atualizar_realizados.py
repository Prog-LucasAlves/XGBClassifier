import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CAMINHO_CSV = "data/production/previsoes_historicas.csv"
TICKER = "PETR4.SA"
THRESHOLD = 0.015  # 1.5% de retorno


def obter_fechamento(ticker, data):
    df = yf.download(
        ticker, start=data, end=pd.to_datetime(data) + pd.Timedelta(days=3)
    )
    if df.empty:
        return None
    fechamento = df.loc[df.index.normalize() == pd.to_datetime(data), "Close"]
    return fechamento.iloc[0] if not fechamento.empty else None


def atualizar_realizados():
    df = pd.read_csv(CAMINHO_CSV)
    atualizou = False

    for i, row in df[df["realizado"].isna()].iterrows():
        data_entrada = row["data_entrada"]
        data_saida = row["data_saida"]

        preco_entrada = obter_fechamento(TICKER, data_entrada)
        preco_saida = obter_fechamento(TICKER, data_saida)

        if preco_entrada and preco_saida:
            retorno = (preco_saida - preco_entrada) / preco_entrada
            df.at[i, "realizado"] = 1 if retorno > THRESHOLD else 0
            atualizou = True
            print(
                f"🔁 {data_entrada} → {data_saida} | Retorno: {retorno:.2%} | realizado = {df.at[i, 'realizado']}"
            )
        else:
            print(f"⚠️ Dados ausentes para {data_entrada} ou {data_saida} — pulando...")

    if atualizou:
        df.to_csv(CAMINHO_CSV, index=False)
        print(f"\n✅ CSV atualizado: {CAMINHO_CSV}")
    else:
        print("\nℹ️ Nenhuma atualização necessária.")


if __name__ == "__main__":
    atualizar_realizados()
