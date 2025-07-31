import os

import joblib
import mlflow
import pandas as pd
from predict_producao import main as treinar_modelo
from sklearn.metrics import accuracy_score, f1_score

CAMINHO_HISTORICO = "data/production/previsoes_historicas.csv"
CAMINHO_MODELO = "models/production/modelo_xgb.pkl"
F1_MINIMO = 0.60  # Limiar para retreinar


def avaliar_desempenho(df):
    df_validas = df.dropna(subset=["realizado"])
    if df_validas.empty:
        print("⚠️ Nenhuma previsão com valor realizado disponível.")
        return None
    y_true = df_validas["realizado"]
    y_pred = (df_validas["sinal"] == "COMPRA").astype(int)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"📊 Avaliação: F1 = {f1:.2f} | Acurácia = {acc:.2f}")
    return {"f1": f1, "accuracy": acc}


def retreinar_e_salvar():
    print("🔁 Iniciando retreinamento...")
    modelo, metricas = treinar_modelo()

    os.makedirs("models/production", exist_ok=True)
    joblib.dump(modelo, CAMINHO_MODELO)
    print(f"✅ Novo modelo salvo em: {CAMINHO_MODELO}")

    # Registro no MLflow
    mlflow.set_experiment("modelo_acao_petr4")
    with mlflow.start_run(run_name="Retreinamento automático"):
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(CAMINHO_MODELO)
        mlflow.log_param("tipo_retreino", "automático")


def main():
    if not os.path.exists(CAMINHO_HISTORICO):
        print("❌ Arquivo de histórico não encontrado.")
        return

    df = pd.read_csv(CAMINHO_HISTORICO)
    resultado = avaliar_desempenho(df)

    if resultado and resultado["f1"] < F1_MINIMO:
        print("⚠️ Desempenho abaixo do esperado. Retreinando modelo...")
        retreinar_e_salvar()
    else:
        print("✅ Desempenho satisfatório. Retreinamento não necessário.")


if __name__ == "__main__":
    main()
