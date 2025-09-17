"""
Configurações globais para testes pytest
Bootcamp Microsoft Data Scientist Azure - Credit Risk Prediction Project
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Cria diretório temporário para dados de teste"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_credit_dataset():
    """Dataset sintético de crédito para testes"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'annual_income': np.random.lognormal(10, 0.8, n_samples),
        'loan_amount': np.random.lognormal(9, 0.7, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_length': np.random.randint(0, 25, n_samples),
        'debt_to_income': np.random.uniform(0, 1, n_samples),
        'num_credit_lines': np.random.randint(1, 20, n_samples),
        'education': np.random.choice(['High School', 'College', 'Graduate'], n_samples),
        'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Target com lógica realista
    risk_score = (
        (df['credit_score'] < 600).astype(int) * 0.4 +
        (df['debt_to_income'] > 0.5).astype(int) * 0.3 +
        (df['annual_income'] < 30000).astype(int) * 0.2 +
        np.random.random(n_samples) * 0.1
    )
    
    df['default'] = (risk_score > 0.5).astype(int)
    
    return df


@pytest.fixture
def sample_csv_file(test_data_dir, sample_credit_dataset):
    """Salva dataset em arquivo CSV temporário"""
    csv_path = os.path.join(test_data_dir, "test_credit_data.csv")
    sample_credit_dataset.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def model_output_dir(test_data_dir):
    """Diretório para salvar outputs de modelos nos testes"""
    output_dir = os.path.join(test_data_dir, "model_outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# Configurações globais do pytest
def pytest_configure(config):
    """Configurações globais do pytest"""
    config.addinivalue_line(
        "markers", 
        "slow: marca testes que demoram para executar"
    )
    config.addinivalue_line(
        "markers", 
        "integration: marca testes de integração"
    )
    config.addinivalue_line(
        "markers", 
        "model: marca testes relacionados a modelos ML"
    )


# Hooks do pytest
def pytest_collection_modifyitems(config, items):
    """Modifica comportamento da coleta de testes"""
    for item in items:
        # Adicionar marca 'slow' para testes que treinam modelos
        if "train" in item.name or "xgboost" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Adicionar marca 'model' para testes de ML
        if any(keyword in item.name for keyword in ["model", "predict", "evaluate"]):
            item.add_marker(pytest.mark.model)


@pytest.fixture
def suppress_warnings():
    """Suprime warnings específicos durante testes"""
    import warnings
    
    # Suprimir warnings comuns em ML
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*The default value of numeric_only.*")
    
    yield
    
    # Restaurar warnings
    warnings.resetwarnings()


class TestConfig:
    """Configurações para testes"""
    
    # Parâmetros para testes
    MIN_ACCURACY = 0.6  # Accuracy mínima esperada
    MIN_AUC = 0.6       # AUC mínimo esperado
    MAX_TRAINING_TIME = 30  # Tempo máximo de treinamento (segundos)
    
    # Tolerâncias numéricas
    FLOAT_TOLERANCE = 1e-6
    
    # Tamanhos de dataset para testes
    SMALL_DATASET_SIZE = 100
    MEDIUM_DATASET_SIZE = 500
    LARGE_DATASET_SIZE = 1000
    
    @classmethod
    def get_test_models(cls):
        """Retorna lista de modelos para testes"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        return {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42)
        }


@pytest.fixture
def test_config():
    """Fixture para configurações de teste"""
    return TestConfig()


# Utilitários para testes
def assert_model_trained(model):
    """Verifica se um modelo foi treinado corretamente"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    
    if isinstance(model, LogisticRegression):
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
    elif isinstance(model, RandomForestClassifier):
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) > 0
    elif isinstance(model, xgb.XGBClassifier):
        assert hasattr(model, '_Booster')
    else:
        # Para outros modelos, verificar se tem método predict
        assert hasattr(model, 'predict')


def assert_valid_predictions(predictions, expected_length):
    """Verifica se predições são válidas"""
    assert len(predictions) == expected_length
    assert all(pred in [0, 1] for pred in predictions)


def assert_valid_probabilities(probabilities):
    """Verifica se probabilidades são válidas"""
    assert all(0 <= prob <= 1 for prob in probabilities)
    # Para classificação binária
    if probabilities.ndim == 2:
        assert np.allclose(probabilities.sum(axis=1), 1.0)


# Fixtures para mocking (se necessário)
@pytest.fixture
def mock_azure_ml_context(monkeypatch):
    """Mock para contexto do Azure ML"""
    # Se estivéssemos testando integração com Azure ML
    # poderíamos mockar as chamadas aqui
    pass
