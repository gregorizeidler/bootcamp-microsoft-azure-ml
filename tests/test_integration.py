"""
Testes de integração para pipeline completo
Demonstra testes end-to-end para Bootcamp Microsoft Data Scientist Azure
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from conftest import TestConfig, assert_model_trained, assert_valid_predictions


@pytest.mark.integration
class TestEndToEndPipeline:
    """Testes de integração do pipeline completo"""
    
    def test_complete_training_pipeline(self, sample_csv_file, model_output_dir):
        """Testa pipeline completo de treinamento"""
        
        # Executar script de treinamento
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
            '--data', sample_csv_file,
            '--model', 'random_forest',
            '--output-dir', model_output_dir,
            '--n-estimators', '10'  # Poucos estimators para teste rápido
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Verificar se executou sem erros
        assert result.returncode == 0, f"Script falhou: {result.stderr}"
        
        # Verificar se arquivos foram criados
        expected_files = ['random_forest_model.pkl', 'metrics.json']
        for file in expected_files:
            assert os.path.exists(os.path.join(model_output_dir, file))
        
        # Verificar métricas
        with open(os.path.join(model_output_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert metrics['accuracy'] > 0.5  # Performance mínima
        assert metrics['auc'] > 0.5
    
    def test_complete_prediction_pipeline(self, sample_csv_file, model_output_dir):
        """Testa pipeline completo de predição"""
        
        # Primeiro, treinar um modelo
        train_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
            '--data', sample_csv_file,
            '--model', 'logistic',
            '--output-dir', model_output_dir
        ]
        
        train_result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=30)
        assert train_result.returncode == 0
        
        # Agora fazer predições
        model_path = os.path.join(model_output_dir, 'logistic_model.pkl')
        predictions_path = os.path.join(model_output_dir, 'predictions.csv')
        
        predict_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'predict.py'),
            '--model-path', model_path,
            '--input-data', sample_csv_file,
            '--output-path', predictions_path
        ]
        
        predict_result = subprocess.run(predict_cmd, capture_output=True, text=True, timeout=30)
        
        # Verificar execução
        assert predict_result.returncode == 0, f"Predição falhou: {predict_result.stderr}"
        
        # Verificar se arquivos foram criados
        assert os.path.exists(predictions_path)
        
        # Verificar conteúdo das predições
        predictions_df = pd.read_csv(predictions_path)
        assert 'prediction' in predictions_df.columns
        assert 'risk_category' in predictions_df.columns
        assert len(predictions_df) > 0
        
        # Verificar valores válidos
        assert set(predictions_df['prediction'].unique()).issubset({0, 1})
        assert set(predictions_df['risk_category'].unique()).issubset({'High Risk', 'Low Risk'})
    
    @pytest.mark.slow
    def test_model_comparison_pipeline(self, sample_csv_file, model_output_dir):
        """Testa comparação de múltiplos modelos"""
        
        models_to_test = ['logistic', 'random_forest']
        results = {}
        
        for model_type in models_to_test:
            output_subdir = os.path.join(model_output_dir, model_type)
            os.makedirs(output_subdir, exist_ok=True)
            
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
                '--data', sample_csv_file,
                '--model', model_type,
                '--output-dir', output_subdir,
                '--n-estimators', '5'  # Rápido para teste
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            assert result.returncode == 0, f"Falhou para {model_type}: {result.stderr}"
            
            # Carregar métricas
            with open(os.path.join(output_subdir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
            
            results[model_type] = metrics
        
        # Comparar resultados
        assert len(results) == len(models_to_test)
        
        for model_type, metrics in results.items():
            assert metrics['accuracy'] > 0.4  # Performance mínima
            assert metrics['auc'] > 0.4
            assert 'training_time_seconds' in metrics
    
    def test_data_validation_pipeline(self, test_data_dir):
        """Testa pipeline com dados inválidos"""
        
        # Criar arquivo com dados inválidos
        invalid_data = pd.DataFrame({
            'invalid_col1': ['a', 'b', 'c'],
            'invalid_col2': [None, None, None]
        })
        
        invalid_file = os.path.join(test_data_dir, 'invalid_data.csv')
        invalid_data.to_csv(invalid_file, index=False)
        
        # Tentar treinar com dados inválidos
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
            '--data', invalid_file,
            '--model', 'logistic',
            '--output-dir', test_data_dir
        ]
        
        # Deve falhar ou usar dados sintéticos
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # O script deve lidar com dados inválidos criando dados sintéticos
        # ou deve falhar graciosamente
        if result.returncode == 0:
            # Se passou, deve ter criado dados sintéticos
            assert "Criando dados sintéticos" in result.stdout
        else:
            # Se falhou, deve ter uma mensagem de erro apropriada
            assert len(result.stderr) > 0


@pytest.mark.integration  
class TestDataPipeline:
    """Testes de integração do pipeline de dados"""
    
    def test_data_preprocessing_consistency(self, sample_credit_dataset):
        """Testa consistência do preprocessamento"""
        
        from utils.data_preprocessing import prepare_features
        
        # Processar dados múltiplas vezes
        X1, y1 = prepare_features(sample_credit_dataset.copy())
        X2, y2 = prepare_features(sample_credit_dataset.copy())
        
        # Resultados devem ser idênticos
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)
    
    def test_feature_engineering_impact(self, sample_credit_dataset):
        """Testa impacto da feature engineering na performance"""
        
        from utils.data_preprocessing import prepare_features
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        
        # Preparar dados com feature engineering
        X_eng, y = prepare_features(sample_credit_dataset)
        
        # Preparar dados sem feature engineering (só colunas numéricas originais)
        numeric_cols = sample_credit_dataset.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'default']
        X_basic = sample_credit_dataset[numeric_cols]
        
        # Splits idênticos
        X_eng_train, X_eng_test, y_train, y_test = train_test_split(
            X_eng, y, test_size=0.3, random_state=42
        )
        X_basic_train, X_basic_test, _, _ = train_test_split(
            X_basic, y, test_size=0.3, random_state=42
        )
        
        # Treinar modelos
        model_eng = LogisticRegression(random_state=42, max_iter=1000)
        model_basic = LogisticRegression(random_state=42, max_iter=1000)
        
        model_eng.fit(X_eng_train, y_train)
        model_basic.fit(X_basic_train, y_train)
        
        # Avaliar
        auc_eng = roc_auc_score(y_test, model_eng.predict_proba(X_eng_test)[:, 1])
        auc_basic = roc_auc_score(y_test, model_basic.predict_proba(X_basic_test)[:, 1])
        
        # Feature engineering deve melhorar ou pelo menos não prejudicar
        assert auc_eng >= auc_basic - 0.05  # Tolerância pequena


@pytest.mark.integration
class TestMLOpsWorkflow:
    """Testes de workflow MLOps"""
    
    def test_model_versioning(self, sample_csv_file, model_output_dir):
        """Testa versionamento de modelos"""
        
        # Treinar primeira versão
        v1_dir = os.path.join(model_output_dir, 'v1')
        os.makedirs(v1_dir, exist_ok=True)
        
        cmd_v1 = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
            '--data', sample_csv_file,
            '--model', 'logistic',
            '--output-dir', v1_dir,
            '--random-state', '42'
        ]
        
        result_v1 = subprocess.run(cmd_v1, capture_output=True, text=True, timeout=30)
        assert result_v1.returncode == 0
        
        # Treinar segunda versão (com parâmetros diferentes)
        v2_dir = os.path.join(model_output_dir, 'v2')
        os.makedirs(v2_dir, exist_ok=True)
        
        cmd_v2 = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
            '--data', sample_csv_file,
            '--model', 'logistic', 
            '--output-dir', v2_dir,
            '--random-state', '123'  # Seed diferente
        ]
        
        result_v2 = subprocess.run(cmd_v2, capture_output=True, text=True, timeout=30)
        assert result_v2.returncode == 0
        
        # Verificar se versões são diferentes
        with open(os.path.join(v1_dir, 'metrics.json'), 'r') as f:
            metrics_v1 = json.load(f)
        
        with open(os.path.join(v2_dir, 'metrics.json'), 'r') as f:
            metrics_v2 = json.load(f)
        
        # Devem ter métricas (podem ser iguais ou diferentes)
        assert 'accuracy' in metrics_v1
        assert 'accuracy' in metrics_v2
        
        # Ambas versões devem ter performance razoável
        assert metrics_v1['accuracy'] > 0.4
        assert metrics_v2['accuracy'] > 0.4
    
    def test_model_reproducibility(self, sample_csv_file, model_output_dir):
        """Testa reprodutibilidade de modelos"""
        
        # Treinar mesmo modelo duas vezes com mesmo seed
        dirs = [os.path.join(model_output_dir, f'run_{i}') for i in range(2)]
        
        for i, run_dir in enumerate(dirs):
            os.makedirs(run_dir, exist_ok=True)
            
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
                '--data', sample_csv_file,
                '--model', 'random_forest',
                '--output-dir', run_dir,
                '--random-state', '42',
                '--n-estimators', '10'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, f"Run {i} falhou: {result.stderr}"
        
        # Comparar métricas
        metrics = []
        for run_dir in dirs:
            with open(os.path.join(run_dir, 'metrics.json'), 'r') as f:
                metrics.append(json.load(f))
        
        # Métricas devem ser idênticas (ou muito próximas devido ao random_state)
        for key in ['accuracy', 'auc', 'f1_score']:
            if key in metrics[0] and key in metrics[1]:
                diff = abs(metrics[0][key] - metrics[1][key])
                assert diff < 0.01, f"Diferença muito grande em {key}: {diff}"


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Testes de performance do sistema completo"""
    
    def test_training_speed(self, sample_csv_file, model_output_dir):
        """Testa se treinamento é executado em tempo razoável"""
        
        import time
        
        start_time = time.time()
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
            '--data', sample_csv_file,
            '--model', 'logistic',
            '--output-dir', model_output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        assert result.returncode == 0
        assert training_time < TestConfig.MAX_TRAINING_TIME, f"Treinamento demorou {training_time:.1f}s"
    
    def test_prediction_throughput(self, sample_csv_file, model_output_dir):
        """Testa throughput de predições"""
        
        # Primeiro treinar modelo
        train_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'),
            '--data', sample_csv_file,
            '--model', 'logistic',
            '--output-dir', model_output_dir
        ]
        
        train_result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=30)
        assert train_result.returncode == 0
        
        # Fazer predições
        model_path = os.path.join(model_output_dir, 'logistic_model.pkl')
        predictions_path = os.path.join(model_output_dir, 'predictions.csv')
        
        import time
        start_time = time.time()
        
        predict_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), '..', 'src', 'predict.py'),
            '--model-path', model_path,
            '--input-data', sample_csv_file,
            '--output-path', predictions_path
        ]
        
        predict_result = subprocess.run(predict_cmd, capture_output=True, text=True, timeout=30)
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        assert predict_result.returncode == 0
        assert prediction_time < 10, f"Predições demoraram {prediction_time:.1f}s"
