# 🚀 Azure ML - Credit Risk Prediction | Bootcamp Microsoft Data Scientist

**Projeto completo de MLOps com Azure Machine Learning para o Bootcamp Microsoft Data Scientist Azure**

[![Azure ML](https://img.shields.io/badge/Azure-ML-blue)](https://azure.microsoft.com/services/machine-learning/)
[![Python](https://img.shields.io/badge/Python-3.9-green)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org)
[![CI/CD](https://img.shields.io/badge/GitHub-Actions-yellow)](.github/workflows)

## 🎯 Objetivo

Implementação end-to-end de um sistema de **predição de risco de crédito** utilizando Azure Machine Learning, demonstrando todas as competências necessárias para o Bootcamp Microsoft Data Scientist Azure:

- ✅ **Design de soluções ML** com Azure ML Studio
- ✅ **Experimentos e tracking** com MLflow 
- ✅ **Deploy de modelos** com endpoints online/batch
- ✅ **MLOps e CI/CD** com GitHub Actions
- ✅ **Monitoramento** de drift e performance

## 🏗️ Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Azure ML       │───▶│  Model Registry │
│   (CSV/Blob)    │    │  Compute        │    │  + Endpoints    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Monitoring &   │◀───│  MLflow         │───▶│  CI/CD Pipeline │
│  Alerting       │    │  Tracking       │    │  (GitHub)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1️⃣ Setup Local
```bash
# Clone e setup
git clone https://github.com/gregorizeidler/bootcamp-microsoft-azure-ml
cd bootcamp-microsoft-azure-ml

# Ambiente conda
conda env create -f environment/conda.yml
conda activate bootcamp-azure-env

# Testes
pytest tests/ -v
```

### 2️⃣ Treino Local
```bash
# EDA e baseline
jupyter notebook notebooks/01-eda-baseline.ipynb

# Treinamento local
python src/train.py --data data/credit_risk.csv --model xgboost
```

### 3️⃣ Azure ML Setup
```bash
# Login e configuração
az login
az ml workspace create -n bootcamp-azure-workspace -g rg-ml --location eastus

# Submit job
az ml job create --file azure-ml/jobs/train-job.yml
```

### 4️⃣ Deploy Modelo
```bash
# Criar endpoint
az ml online-endpoint create --file azure-ml/endpoints/credit-endpoint.yml

# Deploy modelo
az ml online-deployment create --file azure-ml/deployments/xgboost-deployment.yml
```

## 📊 Resultados

| Modelo | Accuracy | AUC | F1-Score | Training Time |
|--------|----------|-----|----------|---------------|
| Logistic Regression | 0.78 | 0.82 | 0.75 | 2.3s |
| Random Forest | 0.85 | 0.91 | 0.84 | 12.4s |
| **XGBoost** ⭐ | **0.89** | **0.94** | **0.88** | **8.7s** |

## 📁 Estrutura do Projeto

```
📦 bootcamp-microsoft-azure-ml/
├── 📂 src/                     # Código fonte principal
│   ├── train.py               # Script de treinamento
│   ├── predict.py             # Script de inferência  
│   ├── utils/                 # Utilitários
│   └── models/                # Definições de modelos
├── 📂 notebooks/              # Jupyter notebooks
│   ├── 01-eda-baseline.ipynb # Análise exploratória
│   └── 02-model-comparison.ipynb # Comparação de modelos
├── 📂 azure-ml/              # Configurações Azure ML
│   ├── jobs/                 # Job definitions
│   ├── endpoints/            # Endpoint configs
│   ├── environments/         # Custom environments
│   └── pipelines/           # ML pipelines
├── 📂 tests/                # Testes automatizados
├── 📂 data/                 # Dados (sample)
├── 📂 .github/workflows/    # CI/CD GitHub Actions
├── requirements.txt         # Dependências
├── environment/conda.yml    # Ambiente conda
└── README.md               # Este arquivo
```

## 🧪 Testes e Qualidade

- **Unit Tests**: Validação de funções core
- **Integration Tests**: Pipeline end-to-end  
- **Model Tests**: Validação de outputs do modelo
- **Data Tests**: Schema e qualidade dos dados

```bash
# Executar todos os testes
make test

# Coverage report
make coverage
```

## 🔄 CI/CD Pipeline

O pipeline automatiza:
1. **Code Quality**: Linting (flake8, black) + testes
2. **Model Training**: Treino automático no Azure ML
3. **Model Validation**: Testes de performance vs baseline
4. **Deployment**: Deploy automático se aprovado
5. **Monitoring**: Setup de alerts de drift

## 📈 Monitoramento

- **Data Drift**: Detecção automática via Azure ML
- **Model Performance**: Tracking contínuo de métricas
- **Alerts**: Notificações Slack/Teams para anomalias
- **A/B Testing**: Comparação entre versões do modelo

## 💡 Decisões Técnicas

### Por que XGBoost?
- **Performance**: Melhor AUC (0.94) vs outros modelos
- **Interpretabilidade**: Feature importance nativa
- **Robustez**: Lida bem com missing values
- **Speed**: Tempo de treino aceitável (8.7s)

### Por que Azure ML?
- **Escalabilidade**: Compute elástico para grandes datasets
- **Tracking**: MLflow integrado para experiment management  
- **Deploy**: Endpoints managed com auto-scaling
- **Governance**: Model registry com versionamento

## 🎯 Próximos Passos

- [ ] **Real-time Inference**: Stream processing com Event Hubs
- [ ] **Advanced Monitoring**: Custom metrics + Prometheus
- [ ] **Multi-model Endpoints**: A/B testing automático
- [ ] **AutoML Integration**: Busca automatizada de hiperparâmetros
- [ ] **Edge Deployment**: ONNX + Azure IoT Edge

## 🏆 Competências do Bootcamp Microsoft Data Scientist Demonstradas

| Competência | Status | Evidência |
|-------------|---------|-----------|
| **Azure ML Fundamentals** | ✅ | Workspace setup + compute configuration |
| **Data Exploration & Analysis** | ✅ | EDA notebook com insights detalhados |
| **Feature Engineering** | ✅ | 20+ features criadas com domain knowledge |
| **Model Training & Tuning** | ✅ | Multiple algorithms + hyperparameter optimization |
| **Model Evaluation** | ✅ | Comprehensive metrics + cross-validation |
| **Model Deployment** | ✅ | Real-time endpoints + batch scoring |
| **MLOps & Automation** | ✅ | CI/CD pipelines + monitoring |
| **Azure Integration** | ✅ | MLflow tracking + Azure ML pipelines |


---
*Desenvolvido para o Bootcamp Microsoft Data Scientist Azure - DIO*
