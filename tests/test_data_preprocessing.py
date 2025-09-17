"""
Testes para o módulo de preprocessamento de dados
Demonstra boas práticas de testes para Bootcamp Microsoft Data Scientist Azure
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_preprocessing import (
    prepare_features,
    handle_missing_values,
    encode_categorical_features,
    create_engineered_features
)


class TestDataPreprocessing:
    """Classe de testes para preprocessamento de dados"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture com dados de exemplo para testes"""
        return pd.DataFrame({
            'age': [25, 35, 45, np.nan],
            'annual_income': [50000, 75000, 100000, 60000],
            'credit_score': [650, 720, 800, 580],
            'education': ['College', 'Graduate', 'High School', 'College'],
            'default': [0, 0, 1, 1]
        })
    
    def test_handle_missing_values_numeric(self, sample_data):
        """Testa tratamento de valores missing em colunas numéricas"""
        df_clean = handle_missing_values(sample_data)
        
        # Não deve ter valores missing
        assert df_clean.isnull().sum().sum() == 0
        
        # A idade missing deve ser preenchida com a mediana
        expected_age = sample_data['age'].median()
        assert df_clean.loc[3, 'age'] == expected_age
    
    def test_encode_categorical_features(self, sample_data):
        """Testa encoding de variáveis categóricas"""
        feature_cols = ['age', 'annual_income', 'credit_score', 'education']
        df_encoded = encode_categorical_features(sample_data, feature_cols)
        
        # Education deve ser codificada (3 categorias -> one-hot)
        expected_cols = ['education_Graduate', 'education_High School']
        for col in expected_cols:
            assert col in df_encoded.columns
        
        # Coluna original education deve ter sido removida
        assert 'education' not in df_encoded.columns
    
    def test_create_engineered_features(self, sample_data):
        """Testa criação de features engineered"""
        # Adicionar colunas necessárias para feature engineering
        sample_data['loan_amount'] = [30000, 50000, 80000, 25000]
        sample_data['employment_length'] = [2, 5, 10, 1]
        sample_data['debt_to_income'] = [0.3, 0.4, 0.6, 0.7]
        
        df_eng = create_engineered_features(sample_data)
        
        # Verificar se novas features foram criadas
        assert 'loan_to_income_ratio' in df_eng.columns
        assert 'credit_score_normalized' in df_eng.columns
        assert 'high_risk_combo' in df_eng.columns
        
        # Verificar cálculo correto de loan_to_income_ratio
        expected_ratio = sample_data['loan_amount'] / (sample_data['annual_income'] + 1)
        pd.testing.assert_series_equal(
            df_eng['loan_to_income_ratio'], 
            expected_ratio, 
            check_names=False
        )
    
    def test_prepare_features_complete_pipeline(self, sample_data):
        """Testa pipeline completo de preparação de features"""
        # Adicionar todas as colunas necessárias
        sample_data['loan_amount'] = [30000, 50000, 80000, 25000]
        sample_data['employment_length'] = [2, 5, 10, 1]
        sample_data['debt_to_income'] = [0.3, 0.4, 0.6, 0.7]
        sample_data['num_credit_lines'] = [3, 5, 8, 2]
        sample_data['home_ownership'] = ['Rent', 'Own', 'Mortgage', 'Rent']
        
        X, y = prepare_features(sample_data)
        
        # Verificações básicas
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == len(sample_data)
        assert 'default' not in X.columns
        
        # Não deve ter valores missing
        assert X.isnull().sum().sum() == 0
        
        # Features engineered devem estar presentes
        engineered_features = ['loan_to_income_ratio', 'credit_score_normalized']
        for feature in engineered_features:
            if feature in X.columns:  # Algumas podem não estar se faltar dados
                assert not X[feature].isnull().any()
    
    def test_data_types(self, sample_data):
        """Testa se os tipos de dados estão corretos após preprocessamento"""
        sample_data['loan_amount'] = [30000, 50000, 80000, 25000]
        sample_data['employment_length'] = [2, 5, 10, 1]
        sample_data['debt_to_income'] = [0.3, 0.4, 0.6, 0.7]
        sample_data['num_credit_lines'] = [3, 5, 8, 2]
        sample_data['home_ownership'] = ['Rent', 'Own', 'Mortgage', 'Rent']
        
        X, y = prepare_features(sample_data)
        
        # Target deve ser int ou float
        assert y.dtype in [int, float, 'int64', 'float64']
        
        # Features numéricas devem ser numéricas
        for col in X.select_dtypes(include=[np.number]).columns:
            assert pd.api.types.is_numeric_dtype(X[col])
    
    def test_empty_dataframe(self):
        """Testa comportamento com DataFrame vazio"""
        empty_df = pd.DataFrame()
        
        with pytest.raises((KeyError, ValueError)):
            prepare_features(empty_df)
    
    def test_single_row(self):
        """Testa preprocessamento com uma única linha"""
        single_row = pd.DataFrame({
            'age': [30],
            'annual_income': [50000],
            'credit_score': [650],
            'loan_amount': [30000],
            'employment_length': [5],
            'debt_to_income': [0.3],
            'num_credit_lines': [3],
            'education': ['College'],
            'home_ownership': ['Rent'],
            'default': [0]
        })
        
        X, y = prepare_features(single_row)
        
        assert len(X) == 1
        assert len(y) == 1
        assert not X.isnull().any().any()


class TestDataQuality:
    """Testes de qualidade de dados"""
    
    def test_no_data_leakage(self):
        """Verifica se não há vazamento de dados"""
        # Criar dados com vazamento intencional
        data_with_leakage = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40],
            'target_related': [0, 0, 1, 1],  # Feature que "vaza" o target
            'default': [0, 0, 1, 1]
        })
        
        X, y = prepare_features(data_with_leakage)
        
        # Verificar se colunas suspeitas foram removidas/transformadas
        suspicious_cols = [col for col in X.columns if 'target' in col.lower()]
        
        # Se houver colunas suspeitas, deve haver tratamento específico
        if suspicious_cols:
            # Implementar lógica específica para detectar vazamento
            pass
    
    def test_feature_scaling_bounds(self):
        """Testa se features normalizadas estão nos limites esperados"""
        data = pd.DataFrame({
            'credit_score': [300, 850, 600],
            'age': [18, 80, 35],
            'default': [1, 0, 0]
        })
        
        X, y = prepare_features(data)
        
        # Credit score normalizado deve estar entre 0 e 1
        if 'credit_score_normalized' in X.columns:
            assert X['credit_score_normalized'].min() >= 0
            assert X['credit_score_normalized'].max() <= 1
