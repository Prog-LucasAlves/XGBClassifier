import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def carregar_dados_teste(caminho="../data/processed/dados_para_backtesting.csv"):
    df = pd.read_csv(caminho, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df


def carregar_top_features(caminho="../data/processed/top_features.txt"):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de features não encontrado em: {caminho}")
    with open(caminho, "r") as f:
        return [linha.strip() for linha in f.readlines()]


def aplicar_modelo(df, modelo_path="../models/modelo_xgb.pkl"):
    modelo = joblib.load(modelo_path)
    features = carregar_top_features()
    df = df.copy()
    df["sinal"] = modelo.predict(df[features])
    return df


def calcular_max_drawdown(series):
    acumulado = series
    pico = acumulado.cummax()
    drawdown = (acumulado - pico) / pico
    return drawdown.min()


def simular_estrategia(df, holding_period=10):
    df = df.copy()
    df["retorno_acumulado"] = (1 + df["Close"].pct_change()).cumprod()

    sinais = df[df["sinal"] == 1].index
    capital = 1.0
    historico = []
    retornos_trades = []

    for i in range(len(sinais)):
        data_entrada = sinais[i]
        if i > 0 and sinais[i] < data_saida:
            continue  # Evita sobreposição de trades

        idx = df.index.get_loc(data_entrada)
        if idx + holding_period >= len(df):
            break  # Sem dados suficientes

        data_saida = df.index[idx + holding_period]
        preco_entrada = df.loc[data_entrada, "Close"]
        preco_saida = df.loc[data_saida, "Close"]
        retorno = preco_saida / preco_entrada
        retornos_trades.append(retorno - 1)

        capital *= retorno
        historico.append((data_saida, capital))

    df["estrategia_acumulada"] = 1.0
    for data, valor in historico:
        df.loc[data:, "estrategia_acumulada"] = valor

    return df, retornos_trades


def calcular_metricas(df, retornos_trades):
    retorno_final_modelo = df["estrategia_acumulada"].iloc[-1] - 1
    retorno_final_hold = df["retorno_acumulado"].iloc[-1] - 1

    drawdown_modelo = calcular_max_drawdown(df["estrategia_acumulada"])
    std = np.std(retornos_trades)
    sharpe_modelo = (np.mean(retornos_trades) / std) * np.sqrt(252) if std != 0 else 0

    acertos = sum(r > 0 for r in retornos_trades)
    total_trades = len(retornos_trades)
    taxa_acerto = acertos / total_trades if total_trades > 0 else 0

    print(f"\n📊 Retorno final modelo: {retorno_final_modelo * 100:+.2f}%")
    print(f"📈 Retorno Buy & Hold: {retorno_final_hold * 100:+.2f}%")
    print(f"📉 Máximo Drawdown modelo: {drawdown_modelo * 100:.2f}%")
    print(f"⚖️ Sharpe Ratio modelo: {sharpe_modelo:.2f}")
    print(f"✅ Trades: {total_trades}, Taxa de acerto: {taxa_acerto * 100:.2f}%")

    return {
        "retorno_final_modelo": retorno_final_modelo,
        "retorno_final_hold": retorno_final_hold,
        "drawdown_modelo": drawdown_modelo,
        "sharpe_modelo": sharpe_modelo,
        "taxa_acerto": taxa_acerto,
        "total_trades": total_trades,
    }


def plotar(df):
    df[["retorno_acumulado", "estrategia_acumulada"]].plot(figsize=(12, 6))
    plt.title("Backtest Modelo vs Buy & Hold")
    plt.ylabel("Crescimento do Capital")
    plt.grid(True)
    plt.savefig("../data/image/backtest_plot.png", dpi=300)
    plt.show()


def main():
    df = carregar_dados_teste()
    df = aplicar_modelo(df)
    df_bt, retornos_trades = simular_estrategia(df)
    calcular_metricas(df_bt, retornos_trades)
    plotar(df_bt)


if __name__ == "__main__":
    main()
