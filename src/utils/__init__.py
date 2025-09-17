"""
Utilitários para o projeto Bootcamp Microsoft Data Scientist Azure
"""

from .data_preprocessing import prepare_features, handle_missing_values, encode_categorical_features
from .model_evaluation import evaluate_model, compare_models, generate_model_report

__all__ = [
    'prepare_features',
    'handle_missing_values', 
    'encode_categorical_features',
    'evaluate_model',
    'compare_models',
    'generate_model_report'
]
