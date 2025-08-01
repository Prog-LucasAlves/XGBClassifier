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
- Avaliação e Análise:
    1. Métricas (acurácia, precisão).
    2. Matriz de confusão.
    3. Identificação de falsos/verdadeiros positivos.

## 📦 Dependências

| Biblioteca Versão | Descrição | Link |
| --------------------- | ------ | --------- | ---- |
| ![version](https://img.shields.io/badge/uv-0.1.0-blueviolet) | Gerenciador de ambientes virtuais ultra-rápido | [GitHub](https://github.com/astral-sh/uv) |
| ![version](https://img.shields.io/badge/Python-3.12.4-blueviolet) | Linguagem de programação principal | [python.org](https://www.python.org/) |
| ![version](https://img.shields.io/badge/yfinance-0.2.64-blueviolet) | API para dados do Yahoo Finance | [GitHub](https://github.com/ranaroussi/yfinance) |
| ![version](https://img.shields.io/badge/xgboost-3.0.2-blueviolet) | Framework de machine learning | [xgboost](https://xgboost.readthedocs.io/en/stable/#) |
| ![version](https://img.shields.io/badge/shap-0.48.0-blueviolet) | Análise de importância de features | [GitHub](https://github.com/shap/shap) |
| ![version](https://img.shields.io/badge/Optuna-3.4.0-blueviolet) | Otimização de hiperparâmetros | [GitHub](https://github.com/optuna/optuna) |

## 🚀 Como Usar

1. Clone o repositório:

``` bash
git clone https://github.com/Prog-LucasAlves/XGBClassifier.git
cd XGBClassifier
```

2. Crie e ative o ambiente virtual:

``` bash
uv venv .venv
source .venv/bin/activate # Linux/macOs
.venv\Scripts/activate # Windows
```
> [!NOTE]
> Nesse passo e necessario ter instalado o ***uv***.

3. Instale as dependências:

``` bash
uv pip install pyproject.toml
```
> [!NOTE]
> Nesse passo e necessario ter instalado o ***uv***.

## 🏗️ Estrutura do Projeto

``` text
📦 XGBClassifier
├── data/
|    ├── image/                    # **Imagens e gráficos gerados**
|    |   └── shap_importance.png   # Gráfico de importância de features
|    |
|    ├── processed/                # **Dados processados**
|    |   └── top_features.txt      # Lista das melhores features
|    |
|    └── raw/                      # **Dados**
|        ├── dados_brutos.csv      # Dados baixados do Yahoo Finance
|        ├── treino.csv            # Conjunto de treino
|        ├── teste.csv             # Conjunto de teste
|        └── validacao.csv         # Conjunto de validação
|
├── models/                        # **Modelo treinado**
|
├── scripts/                       # **Scripts do projeto**
|   ├── baktesting.py              # Script para backtesting
|   ├──
|   ├──
|   └──
|
├── .gitignore                     # Arquivos ignorados pelo git
├── .python-version                # Versão do python no projeto
├── pyproject.toml                 # Dependências do projeto (uv)
├── README.md                      # Documentação
```

## ⚙️ Configuração

- O script está configurado para analisar ações da PETR4.SA (Petrobras), mas pode ser facilmente adaptado:

``` python
# No arquivo treinar_modelo_v1, linha 480 altere:
ticker = "PETR4.SA"  # Para outro ticker do Yahoo Finance
```

## 📊 Métricas de Performance

## 🤝 Contribuição

- Contribuições são bem-vindas! Siga os passos:

1. Faça um fork do projeto
2. Crie sua branch (git checkout -b feature/nova-feature)
3. Commit suas mudanças (git commit -m 'Adiciona nova feature')
4. Push para a branch (git push origin feature/nova-feature)
5. Abra um Pull Request
