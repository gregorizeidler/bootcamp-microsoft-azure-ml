"""
Testes para módulos de treinamento de modelos
Demonstra testes de ML para Bootcamp Microsoft Data Scientist Azure
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.model_evaluation import evaluate_model, compare_models


class TestModelTraining:
    """Testes para treinamento de modelos"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Fixture com dados sintéticos para treinamento"""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randint(0, 5, n_samples),
            'feature4': np.random.uniform(0, 1, n_samples)
        })
        
        # Target baseado em lógica simples
        y = ((X['feature1'] > 0) & (X['feature2'] > 0)).astype(int)
        
        # Adicionar ruído
        noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y.iloc[noise_idx] = 1 - y.iloc[noise_idx]
        
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_logistic_regression_training(self, sample_training_data):
        """Testa treinamento de Logistic Regression"""
        X_train, X_test, y_train, y_test = sample_training_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Verificar se modelo foi treinado
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Verificar predições
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})
        
        # Verificar probabilidades
        y_proba = model.predict_proba(X_test)
        assert y_proba.shape == (len(y_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
    
    def test_random_forest_training(self, sample_training_data):
        """Testa treinamento de Random Forest"""
        X_train, X_test, y_train, y_test = sample_training_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Verificar se modelo foi treinado
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) == 10
        
        # Verificar feature importance
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]
        assert np.allclose(model.feature_importances_.sum(), 1.0)
    
    def test_xgboost_training(self, sample_training_data):
        """Testa treinamento de XGBoost"""
        X_train, X_test, y_train, y_test = sample_training_data
        
        model = xgb.XGBClassifier(n_estimators=10, random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        # Verificar se modelo foi treinado
        assert hasattr(model, 'feature_importances_')
        
        # Verificar predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        assert len(y_pred) == len(y_test)
        assert y_proba.shape == (len(y_test), 2)
    
    def test_model_serialization(self, sample_training_data):
        """Testa serialização e desserialização de modelos"""
        X_train, X_test, y_train, y_test = sample_training_data
        
        # Treinar modelo
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Predições originais
        original_pred = model.predict(X_test)
        
        # Salvar e carregar modelo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            joblib.dump(model, tmp.name)
            loaded_model = joblib.load(tmp.name)
            os.unlink(tmp.name)
        
        # Verificar se predições são iguais
        loaded_pred = loaded_model.predict(X_test)
        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestModelEvaluation:
    """Testes para avaliação de modelos"""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Fixture com modelo treinado e dados de teste"""
        np.random.seed(42)
        n_samples = 100
        
        X_test = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        
        y_test = ((X_test['feature1'] > 0) & (X_test['feature2'] > 0)).astype(int)
        
        # Treinar modelo simples
        X_train = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        })
        y_train = ((X_train['feature1'] > 0) & (X_train['feature2'] > 0)).astype(int)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def test_evaluate_model_metrics(self, trained_model_and_data):
        """Testa se evaluate_model retorna métricas corretas"""
        model, X_test, y_test = trained_model_and_data
        
        metrics = evaluate_model(model, X_test, y_test, "Test Model")
        
        # Verificar se todas as métricas estão presentes
        expected_metrics = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Verificar se métricas estão nos limites esperados
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['auc'] <= 1
        assert metrics['support'] == len(y_test)
    
    def test_compare_models(self, sample_training_data):
        """Testa comparação entre múltiplos modelos"""
        X_train, X_test, y_train, y_test = sample_training_data
        
        # Treinar múltiplos modelos
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=5, random_state=42)
        }
        
        for model in models.values():
            model.fit(X_train, y_train)
        
        # Comparar modelos
        comparison = compare_models(models, X_test, y_test)
        
        # Verificações
        assert len(comparison) == len(models)
        assert all(model_name in comparison['model_name'].values for model_name in models.keys())
        
        # Verificar se está ordenado por AUC
        auc_values = comparison['auc'].values
        assert all(auc_values[i] >= auc_values[i+1] for i in range(len(auc_values)-1))


class TestModelPerformance:
    """Testes de performance e limites mínimos"""
    
    def test_model_minimum_performance(self):
        """Testa se modelos atingem performance mínima"""
        # Criar dados com padrão claro
        np.random.seed(42)
        n_samples = 500
        
        X = pd.DataFrame({
            'important_feature': np.random.randn(n_samples),
            'noise_feature': np.random.randn(n_samples) * 0.1
        })
        
        # Target com padrão claro
        y = (X['important_feature'] > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Treinar modelo
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar
        metrics = evaluate_model(model, X_test, y_test)
        
        # Performance mínima esperada (com dados sintéticos deve ser alta)
        assert metrics['accuracy'] > 0.7, f"Accuracy muito baixa: {metrics['accuracy']}"
        assert metrics['auc'] > 0.7, f"AUC muito baixo: {metrics['auc']}"
    
    def test_overfitting_detection(self):
        """Testa detecção de overfitting"""
        # Criar dataset pequeno para facilitar overfitting
        np.random.seed(42)
        n_samples = 50
        
        X = pd.DataFrame(np.random.randn(n_samples, 10))
        y = np.random.choice([0, 1], n_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Modelo propenso a overfitting
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        model.fit(X_train, y_train)
        
        # Métricas
        train_accuracy = model.score(X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)
        test_accuracy = test_metrics['accuracy']
        
        # Gap muito grande pode indicar overfitting
        accuracy_gap = train_accuracy - test_accuracy
        
        # Com dados aleatórios, não devemos ter performance muito alta
        assert test_accuracy < 0.8, "Performance suspeita em dados aleatórios"
    
    def test_prediction_consistency(self):
        """Testa consistência de predições"""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame(np.random.randn(n_samples, 4))
        y = (X.iloc[:, 0] > 0).astype(int)
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Fazer predições múltiplas vezes
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        pred3 = model.predict(X)
        
        # Predições devem ser consistentes
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)


class TestModelInputValidation:
    """Testes de validação de entrada"""
    
    def test_empty_input_handling(self):
        """Testa tratamento de entradas vazias"""
        model = LogisticRegression()
        
        with pytest.raises(ValueError):
            model.fit(pd.DataFrame(), pd.Series([]))
    
    def test_mismatched_dimensions(self):
        """Testa tratamento de dimensões incompatíveis"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], 90))  # Tamanho diferente
        
        model = LogisticRegression()
        
        with pytest.raises(ValueError):
            model.fit(X, y)
    
    def test_invalid_target_values(self):
        """Testa tratamento de valores de target inválidos"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(['invalid'] * 100)  # Strings em vez de 0/1
        
        model = LogisticRegression()
        
        with pytest.raises((ValueError, TypeError)):
            model.fit(X, y)
