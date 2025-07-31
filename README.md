# Modelo de Previsão de Ações com XGBoost e Análise SHAP

## 📌 Visão Geral

Este projeto implementa um pipeline de machine learning para previsão do mercado de ações utilizando **XGBoost** e SHAP (SHapley Additive exPlanations) para análise de importância de features. O modelo prevê se o preço de uma ação vai subir (1) ou cair (0) num horizonte de 10 dias com base em indicadores técnicos e padrões de preço.

## ✨ Funcionalidades

- Preparação dos Dados Automatizada
    1. Download automático de dados históricos do Yahoo Finace.
    2. Cálculo(Criação) de variáveis.
    3. Separção temporal dos dados(Treino, Teste, Validação).
- Separação de **Features**:
    1. A

## 📦 Dependências

| Ferramenta/Biblioteca | Versão | Descrição | Link |
| --------------------- | ------ | --------- | ---- |
| uv | ![version](https://img.shields.io/badge/0.1.0-blue) | Gerenciador de ambientes virtuais ultra-rápido | [GitHub](https://github.com/astral-sh/uv) |
| Python | ![version](https://img.shields.io/badge/0.1.0-blue) | Linguagem de programação principal | [python.org](...) |
