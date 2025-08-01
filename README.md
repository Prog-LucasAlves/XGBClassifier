# Modelo de Previsão de Ações com XGBoost e Análise SHAP

## 📌 Visão Geral

Este projeto implementa um pipeline de machine learning para previsão do mercado de ações utilizando **XGBoost** e SHAP (SHapley Additive exPlanations) para análise de importância de features. O modelo prevê se o preço de uma ação vai subir (1) ou cair (0) num horizonte de 10 dias com base em indicadores técnicos e padrões de preço.

## ✨ Funcionalidades

- Preparação dos Dados Automatizada
    1. Download automático de dados históricos do Yahoo Finace.
    2. Cálculo(Criação) de variáveis.
    3. Separção temporal dos dados(Treino, Teste, Validação).
- Separação de **Features**:
    1. Método **SHAP** para identificar as variáveis mais importantes.
    2. Geração automática de ranking de features.
    3. Visualização gráfica das features mais relevantes.
- Modelagem Preditiva:
    1. **Stacking Classifier** combinando XGBoost e Decision Trees.
    2. Otimização de hiperparâmetros com Optuna.


## 📦 Dependências

| Ferramenta/Biblioteca | Versão | Descrição | Link |
| --------------------- | ------ | --------- | ---- |
| uv | ![version](https://img.shields.io/badge/0.1.0-blue) | Gerenciador de ambientes virtuais ultra-rápido | [GitHub](https://github.com/astral-sh/uv) |
| Python | ![version](https://img.shields.io/badge/3.12.4-red) | Linguagem de programação principal | [python.org](https://www.python.org/) |
| yfinance | ![version](https://img.shields.io/badge/0.2.64-green) | API para dados do Yahoo Finance | [GitHub](https://github.com/ranaroussi/yfinance) |
| xgboost | ![version](https://img.shields.io/badge/3.0.2-yellon) | Framework de machine learning | [xgboost](https://xgboost.readthedocs.io/en/stable/#) |
| shap | ![version](https://img.shields.io/badge/0.48.0-orange) | Análise de importância de features | [GitHub](https://github.com/shap/shap) |
| optuna | |[version](ttps://img.shields.io/badge/Optuna-3.4.0-blueviolet) | Otimização de hiperparâmetros | [GitHub](https://github.com/optuna/optuna) |
