# Bootcamp Microsoft Data Scientist Azure - Credit Risk Prediction
# Makefile para facilitar comandos comuns

.PHONY: help setup test train predict clean lint format docker azure-setup

# Variáveis
PYTHON = python
SRC_DIR = src
TEST_DIR = tests
DATA_FILE = data/credit_risk.csv
MODEL_DIR = outputs

# Help
help:
	@echo "🚀 Bootcamp Microsoft Data Scientist Azure - Credit Risk Prediction"
	@echo "============================================"
	@echo ""
	@echo "Available commands:"
	@echo "  setup          📦 Setup development environment"
	@echo "  test           🧪 Run all tests"
	@echo "  test-fast      ⚡ Run fast tests only"
	@echo "  test-integration 🔗 Run integration tests"
	@echo "  lint           🔍 Run linting (flake8)"
	@echo "  format         ✨ Format code (black + isort)"
	@echo "  train          🏋️  Train all models"
	@echo "  train-xgboost  🤖 Train XGBoost model"
	@echo "  train-rf       🌲 Train Random Forest model"
	@echo "  predict        🔮 Make predictions"
	@echo "  notebook       📓 Start Jupyter notebook"
	@echo "  clean          🧹 Clean generated files"
	@echo "  docker         🐳 Build Docker image"
	@echo "  azure-setup    ☁️  Setup Azure ML workspace"
	@echo "  azure-train    🚀 Submit training job to Azure ML"
	@echo "  security       🔒 Run security checks"
	@echo ""

# Setup
setup:
	@echo "📦 Setting up development environment..."
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Setup completed!"

setup-conda:
	@echo "📦 Setting up conda environment..."
	conda env create -f environment/conda.yml
	@echo "✅ Conda environment 'bootcamp-azure-env' created!"
	@echo "💡 Activate with: conda activate bootcamp-azure-env"

# Testing
test:
	@echo "🧪 Running all tests..."
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html
	@echo "✅ All tests completed!"

test-fast:
	@echo "⚡ Running fast tests..."
	pytest $(TEST_DIR) -v -m "not slow" -x
	@echo "✅ Fast tests completed!"

test-integration:
	@echo "🔗 Running integration tests..."
	pytest $(TEST_DIR)/test_integration.py -v
	@echo "✅ Integration tests completed!"

test-unit:
	@echo "🧪 Running unit tests..."
	pytest $(TEST_DIR)/test_data_preprocessing.py $(TEST_DIR)/test_model_training.py -v
	@echo "✅ Unit tests completed!"

# Code Quality
lint:
	@echo "🔍 Running linting..."
	flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting completed!"

format:
	@echo "✨ Formatting code..."
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "✅ Code formatted!"

format-check:
	@echo "🔍 Checking code format..."
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)
	@echo "✅ Code format check completed!"

# Security
security:
	@echo "🔒 Running security checks..."
	safety check
	bandit -r $(SRC_DIR) -f json || true
	@echo "✅ Security checks completed!"

# Training
train: train-logistic train-rf train-xgboost
	@echo "🏋️  All models trained!"

train-logistic:
	@echo "🤖 Training Logistic Regression..."
	mkdir -p $(MODEL_DIR)/logistic
	$(PYTHON) $(SRC_DIR)/train.py --model logistic --output-dir $(MODEL_DIR)/logistic

train-rf:
	@echo "🌲 Training Random Forest..."
	mkdir -p $(MODEL_DIR)/rf
	$(PYTHON) $(SRC_DIR)/train.py --model random_forest --output-dir $(MODEL_DIR)/rf

train-xgboost:
	@echo "🚀 Training XGBoost..."
	mkdir -p $(MODEL_DIR)/xgboost
	$(PYTHON) $(SRC_DIR)/train.py --model xgboost --output-dir $(MODEL_DIR)/xgboost

# Prediction
predict:
	@echo "🔮 Making predictions..."
	@if [ ! -f $(MODEL_DIR)/xgboost/xgboost_model.pkl ]; then \
		echo "❌ Model not found. Run 'make train-xgboost' first."; \
		exit 1; \
	fi
	$(PYTHON) $(SRC_DIR)/predict.py \
		--model-path $(MODEL_DIR)/xgboost/xgboost_model.pkl \
		--input-data $(DATA_FILE) \
		--output-path $(MODEL_DIR)/predictions.csv
	@echo "✅ Predictions saved to $(MODEL_DIR)/predictions.csv"

# Development
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook notebooks/

create-data:
	@echo "📊 Creating synthetic dataset..."
	mkdir -p data
	$(PYTHON) -c "from src.train import create_synthetic_credit_data; import pandas as pd; df = create_synthetic_credit_data(2000); df.to_csv('$(DATA_FILE)', index=False); print(f'✅ Dataset created: {len(df)} samples')"

# Docker
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t bootcamp-azure-credit-risk:latest .
	@echo "✅ Docker image built!"

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --rm -v $(PWD):/app bootcamp-azure-credit-risk:latest bash

# Azure ML
azure-setup:
	@echo "☁️  Setting up Azure ML workspace..."
	@echo "Please ensure you have Azure CLI installed and logged in:"
	@echo "  az login"
	@echo "  az account set --subscription YOUR_SUBSCRIPTION_ID"
	@echo ""
	@echo "Creating resource group and workspace..."
	az group create --name rg-bootcamp-azure-demo --location eastus || true
	az ml workspace create --name bootcamp-azure-workspace --resource-group rg-bootcamp-azure-demo --location eastus || true
	@echo "✅ Azure ML workspace setup completed!"

azure-train:
	@echo "🚀 Submitting training job to Azure ML..."
	az ml job create --file azure-ml/jobs/train-job.yml
	@echo "✅ Training job submitted to Azure ML!"

azure-pipeline:
	@echo "🔄 Submitting pipeline to Azure ML..."
	az ml job create --file azure-ml/pipelines/training-pipeline.yml
	@echo "✅ Pipeline submitted to Azure ML!"

azure-endpoint:
	@echo "🌐 Creating Azure ML endpoint..."
	az ml online-endpoint create --file azure-ml/endpoints/credit-endpoint.yml
	az ml online-deployment create --file azure-ml/deployments/xgboost-deployment.yml
	@echo "✅ Azure ML endpoint created!"

# MLOps
mlflow-ui:
	@echo "📊 Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

benchmark:
	@echo "⚡ Running performance benchmark..."
	$(PYTHON) -m pytest tests/test_integration.py::TestPerformanceIntegration::test_training_speed -v
	$(PYTHON) -m pytest tests/test_integration.py::TestPerformanceIntegration::test_prediction_throughput -v
	@echo "✅ Benchmark completed!"

# Cleanup
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf outputs/*
	rm -rf mlruns
	rm -rf .mypy_cache
	@echo "✅ Cleanup completed!"

clean-data:
	@echo "🗑️  Cleaning data files..."
	rm -rf data/*.csv
	@echo "✅ Data files cleaned!"

# CI/CD simulation
ci: format-check lint test-fast
	@echo "🎯 CI pipeline simulation completed!"

# Installation verification
verify-install:
	@echo "✅ Verifying installation..."
	$(PYTHON) -c "import pandas, numpy, sklearn, xgboost, mlflow; print('✅ All core packages imported successfully')"
	$(PYTHON) -c "from src.utils import data_preprocessing; print('✅ Custom modules can be imported')"
	@echo "✅ Installation verified!"

# Help for specific Azure commands
azure-help:
	@echo "☁️  Azure ML Commands Help"
	@echo "=========================="
	@echo ""
	@echo "Prerequisites:"
	@echo "  1. Install Azure CLI: https://docs.microsoft.com/cli/azure/"
	@echo "  2. Install ML extension: az extension add -n ml"
	@echo "  3. Login: az login"
	@echo "  4. Set subscription: az account set --subscription YOUR_SUBSCRIPTION_ID"
	@echo ""
	@echo "Common workflows:"
	@echo "  make azure-setup     - Create workspace and compute"
	@echo "  make azure-train     - Submit training job"
	@echo "  make azure-pipeline  - Run full ML pipeline"
	@echo "  make azure-endpoint  - Deploy model as endpoint"
	@echo ""

# Project stats
stats:
	@echo "📈 Project Statistics"
	@echo "===================="
	@echo "Python files:"
	@find $(SRC_DIR) -name "*.py" | wc -l
	@echo "Test files:"
	@find $(TEST_DIR) -name "*.py" | wc -l
	@echo "Lines of code (src):"
	@find $(SRC_DIR) -name "*.py" -exec cat {} \; | wc -l
	@echo "Lines of tests:"
	@find $(TEST_DIR) -name "*.py" -exec cat {} \; | wc -l
	@echo ""
