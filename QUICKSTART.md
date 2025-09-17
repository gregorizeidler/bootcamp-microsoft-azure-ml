# ⚡ Quick Start - Bootcamp Microsoft Data Scientist Azure

Guia rápido para começar a usar o projeto em **5 minutos**!

## 🚀 Setup Rápido

### 1️⃣ Clone e Setup
```bash
git clone https://github.com/gregorizeidler/bootcamp-microsoft-azure-ml
cd bootcamp-microsoft-azure-ml

# Opção A: pip
pip install -r requirements.txt

# Opção B: conda (recomendado)
make setup-conda
conda activate bootcamp-azure-env
```

### 2️⃣ Verificar Instalação
```bash
make verify-install
```

## 🏋️ Treinar Primeiro Modelo (Local)

```bash
# Treinar XGBoost (melhor modelo)
make train-xgboost

# Ou treinar todos os modelos
make train
```

**Resultado esperado:**
```
🏆 MELHOR MODELO: XGBoost
🎯 AUC Score: 0.94X
💾 Modelo salvo: outputs/xgboost/xgboost_model.pkl
```

## 🔮 Fazer Predições

```bash
# Fazer predições com o melhor modelo
make predict
```

**Arquivo gerado:** `outputs/predictions.csv` com colunas:
- `prediction`: 0 (baixo risco) ou 1 (alto risco)  
- `default_probability`: probabilidade de default
- `risk_category`: "Low Risk" ou "High Risk"

## 📊 Explorar Dados

```bash
# Abrir notebook de análise exploratória
make notebook
```

Abrir: `notebooks/01-eda-baseline.ipynb`

## 🧪 Executar Testes

```bash
# Testes rápidos
make test-fast

# Todos os testes (incluindo integração)
make test
```

## ☁️ Azure ML (Opcional)

### Setup
```bash
# 1. Login no Azure
az login

# 2. Criar workspace
make azure-setup

# 3. Submeter job de treinamento
make azure-train
```

### Monitorar
```bash
# Ver jobs
az ml job list

# MLflow UI local
make mlflow-ui
# Abrir: http://localhost:5000
```

## 🐳 Docker

```bash
# Build imagem
make docker-build

# Executar container
make docker-run
```

## 📈 Comandos Úteis

```bash
# Ver todas as opções
make help

# Limpar arquivos temporários
make clean

# Verificar qualidade do código
make lint

# Formatar código
make format

# Benchmark de performance  
make benchmark

# Estatísticas do projeto
make stats
```

## 🎯 Casos de Uso Rápidos

### Para Estudantes do Bootcamp Microsoft
```bash
# 1. Executar análise completa
make notebook  # EDA + baseline models

# 2. Testar pipeline de CI/CD  
make ci

# 3. Deploy no Azure ML
make azure-setup azure-train
```

### Para Desenvolvedores
```bash
# 1. Setup ambiente de dev
make setup
make format lint test

# 2. Adicionar nova feature
# Editar src/
make test-fast

# 3. Commit com confiança
git commit -m "feat: nova feature"
```

### Para MLOps Engineers
```bash
# 1. Pipeline completo
make train test

# 2. Deploy e monitoramento  
make azure-train azure-endpoint
make benchmark

# 3. CI/CD via GitHub Actions (automático)
```

## 📊 Estrutura de Arquivos

```
📦 Projeto
├── 📂 src/                 # Código principal
│   ├── train.py           # Treinamento
│   ├── predict.py         # Inferência
│   └── utils/             # Utilitários
├── 📂 notebooks/          # Jupyter notebooks  
├── 📂 tests/              # Testes automatizados
├── 📂 azure-ml/           # Configs Azure ML
├── 📂 .github/workflows/  # CI/CD
└── 📋 Makefile            # Comandos úteis
```

## 🎓 Para o Bootcamp Microsoft Data Scientist Azure

### Competências Demonstradas
- ✅ **Design de soluções ML** - Architecture + decisions  
- ✅ **Exploração de dados** - EDA notebook com análise profunda
- ✅ **Feature engineering** - 20+ features criadas com domain knowledge
- ✅ **Treinamento de modelos** - Multiple algorithms + hyperparameter tuning
- ✅ **Avaliação** - Comprehensive metrics + cross-validation  
- ✅ **Deploy** - Real-time endpoints + batch scoring
- ✅ **MLOps** - CI/CD + monitoring + Azure integration

### Módulos do Bootcamp Cobertos
- [x] Azure ML Fundamentals
- [x] Data Exploration & Preparation  
- [x] Feature Engineering Avançado
- [x] Model Training & Optimization
- [x] Model Evaluation & Selection
- [x] Model Deployment (Real-time/Batch)
- [x] MLOps & Azure Integration
- [x] Project Portfolio Development

## 🚨 Troubleshooting

### Erro: "Module not found"
```bash
# Adicionar src/ ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Erro: "Azure CLI not found"  
```bash
# Instalar Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az extension add -n ml
```

### Erro: "Model performance baixa"
```bash
# Verificar dados
python -c "import pandas as pd; df=pd.read_csv('data/credit_risk.csv'); print(df.head())"

# Re-treinar com mais dados
python src/train.py --data data/credit_risk.csv --model xgboost --n-estimators 200
```

## 🎯 Next Steps

Após este quickstart:

1. 📚 **Estude o código** - `src/train.py` e `src/utils/`
2. 🧪 **Modifique parâmetros** - Teste diferentes algoritmos 
3. 📊 **Analise resultados** - Compare modelos no notebook
4. ☁️ **Deploy no Azure** - Use os YAMLs em `azure-ml/`
5. 🚀 **Customize** - Adicione suas próprias features

## 💡 Dicas de Performance

- Use `make test-fast` durante desenvolvimento
- Configure Azure ML compute corretamente
- Monitore custos no Azure Portal
- Use MLflow para comparar experimentos
- Mantenha dados balanceados

---

**🏆 Meta: Completar com sucesso o Bootcamp Microsoft Data Scientist Azure!**

📞 Dúvidas? Veja `README.md` para documentação completa.
