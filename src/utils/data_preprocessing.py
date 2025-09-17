"""
Utilitários para preprocessamento de dados
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def prepare_features(df):
    """
    Prepara features para treinamento do modelo
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        tuple: (X, y) - features e target
    """
    df_processed = df.copy()
    
    # Separar features e target
    target_col = 'default'
    feature_cols = [col for col in df_processed.columns if col != target_col]
    
    # Tratar valores missing
    df_processed = handle_missing_values(df_processed)
    
    # Encoding de variáveis categóricas
    df_processed = encode_categorical_features(df_processed, feature_cols)
    
    # Feature engineering
    df_processed = create_engineered_features(df_processed)
    
    # Atualizar lista de features após feature engineering
    feature_cols = [col for col in df_processed.columns if col != target_col]
    
    # Normalização (opcional - alguns modelos não precisam)
    # df_processed[feature_cols] = normalize_features(df_processed[feature_cols])
    
    X = df_processed[feature_cols]
    y = df_processed[target_col]
    
    return X, y


def handle_missing_values(df):
    """Trata valores missing no dataset"""
    df_clean = df.copy()
    
    # Para variáveis numéricas: mediana
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
    
    # Para variáveis categóricas: moda
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_value, inplace=True)
    
    return df_clean


def encode_categorical_features(df, feature_cols):
    """Codifica variáveis categóricas"""
    df_encoded = df.copy()
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    categorical_features = [col for col in categorical_cols if col in feature_cols]
    
    for col in categorical_features:
        # Para variáveis com poucas categorias: one-hot encoding
        unique_values = df_encoded[col].nunique()
        
        if unique_values <= 5:  # One-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
        
        else:  # Label encoding para muitas categorias
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded


def create_engineered_features(df):
    """Cria features engineered baseadas no domínio"""
    df_eng = df.copy()
    
    # Feature 1: Loan to Income Ratio
    if 'loan_amount' in df_eng.columns and 'annual_income' in df_eng.columns:
        df_eng['loan_to_income_ratio'] = df_eng['loan_amount'] / (df_eng['annual_income'] + 1)
    
    # Feature 2: Age groups
    if 'age' in df_eng.columns:
        df_eng['age_group'] = pd.cut(
            df_eng['age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=['young', 'young_adult', 'middle_aged', 'senior', 'elderly']
        ).astype(str)
    
    # Feature 3: Credit utilization (se disponível)
    if 'credit_score' in df_eng.columns:
        df_eng['credit_score_normalized'] = (df_eng['credit_score'] - 300) / (850 - 300)
        df_eng['credit_tier'] = pd.cut(
            df_eng['credit_score'],
            bins=[0, 580, 670, 740, 850],
            labels=['poor', 'fair', 'good', 'excellent']
        ).astype(str)
    
    # Feature 4: Income per year of employment
    if 'annual_income' in df_eng.columns and 'employment_length' in df_eng.columns:
        df_eng['income_per_employment_year'] = df_eng['annual_income'] / (df_eng['employment_length'] + 1)
    
    # Feature 5: High risk combination
    if all(col in df_eng.columns for col in ['debt_to_income', 'credit_score']):
        df_eng['high_risk_combo'] = (
            (df_eng['debt_to_income'] > 0.5) & 
            (df_eng['credit_score'] < 600)
        ).astype(int)
    
    return df_eng


def normalize_features(X):
    """Normaliza features numéricas"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def get_feature_importance_summary(model, feature_names, top_n=10):
    """
    Retorna um resumo das features mais importantes
    
    Args:
        model: Modelo treinado
        feature_names: Lista com nomes das features
        top_n: Número de top features para retornar
        
    Returns:
        pd.DataFrame: DataFrame com feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()  # Modelo não suporta feature importance
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    return feature_importance_df
