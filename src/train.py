"""
Script de treinamento para modelo de predição de risco de crédito
Compatível com Azure ML e execução local
"""

import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import xgboost as xgb
import joblib
import json
from datetime import datetime

# Importa utilitários locais
from utils.data_preprocessing import prepare_features
from utils.model_evaluation import evaluate_model


def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/credit_risk.csv",
                       help="Path para o dataset")
    parser.add_argument("--model", type=str, default="xgboost", 
                       choices=["logistic", "random_forest", "xgboost"],
                       help="Tipo de modelo a treinar")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proporção do dataset para teste")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Seed para reprodutibilidade")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Diretório para salvar outputs")
    
    # Hiperparâmetros específicos
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    
    return parser.parse_args()


def load_and_prepare_data(data_path, test_size=0.2, random_state=42):
    """Carrega e prepara os dados para treinamento"""
    print(f"📥 Carregando dados de: {data_path}")
    
    # Se não existir, criar dados sintéticos para demonstração
    if not os.path.exists(data_path):
        print("⚠️  Dataset não encontrado. Criando dados sintéticos...")
        df = create_synthetic_credit_data(1000, random_state)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    print(f"📊 Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Preprocessamento
    X, y = prepare_features(df)
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"🔄 Split realizado - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def create_synthetic_credit_data(n_samples=1000, random_state=42):
    """Cria dataset sintético para demonstração"""
    np.random.seed(random_state)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'annual_income': np.random.lognormal(10, 1, n_samples),
        'loan_amount': np.random.lognormal(9, 1, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_length': np.random.randint(0, 25, n_samples),
        'debt_to_income': np.random.uniform(0, 1, n_samples),
        'num_credit_lines': np.random.randint(1, 20, n_samples),
        'education': np.random.choice(['High School', 'College', 'Graduate'], n_samples),
        'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Criar variável target baseada em lógica de negócio
    risk_score = (
        (df['credit_score'] < 600).astype(int) * 0.4 +
        (df['debt_to_income'] > 0.5).astype(int) * 0.3 +
        (df['annual_income'] < 30000).astype(int) * 0.2 +
        np.random.random(n_samples) * 0.1
    )
    
    df['default'] = (risk_score > 0.5).astype(int)
    
    return df


def get_model(model_type, **params):
    """Retorna modelo baseado no tipo especificado"""
    if model_type == "logistic":
        return LogisticRegression(random_state=42, max_iter=1000)
    
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            random_state=42
        )
    
    elif model_type == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42,
            eval_metric='logloss'
        )
    
    else:
        raise ValueError(f"Modelo não suportado: {model_type}")


def main():
    """Função principal de treinamento"""
    args = parse_args()
    
    # Configurar MLflow
    mlflow.set_experiment("Credit_Risk_Prediction")
    
    with mlflow.start_run():
        # Log parâmetros
        mlflow.log_params({
            "model_type": args.model,
            "test_size": args.test_size,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate
        })
        
        print(f"🚀 Iniciando treinamento - Modelo: {args.model}")
        start_time = datetime.now()
        
        # Carregar dados
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            args.data, args.test_size, args.random_state
        )
        
        # Criar modelo
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate
        }
        model = get_model(args.model, **model_params)
        
        # Treinamento
        print("🏋️  Treinando modelo...")
        model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  Tempo de treinamento: {training_time:.1f}s")
        
        # Avaliação
        print("📊 Avaliando modelo...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred),
            "training_time_seconds": training_time
        }
        
        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())
        
        # Salvar modelo
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"{args.model}_model.pkl")
        joblib.dump(model, model_path)
        
        # Log modelo no MLflow
        if args.model == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Salvar métricas
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Report final
        print("\n" + "="*50)
        print("📈 RESULTADOS FINAIS")
        print("="*50)
        print(f"Modelo: {args.model}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"AUC: {metrics['auc']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"Tempo de treinamento: {training_time:.1f}s")
        print(f"CV AUC (média ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"\nModelo salvo em: {model_path}")
        print(f"Métricas salvas em: {metrics_path}")
        print("="*50)
        
        # Classification report detalhado
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
