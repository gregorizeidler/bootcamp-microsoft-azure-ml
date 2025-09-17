"""
Script de inferência para o modelo de predição de risco de crédito
Compatível com Azure ML endpoints e execução local
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from utils.data_preprocessing import prepare_features


def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs/xgboost_model.pkl",
                       help="Path para o modelo treinado")
    parser.add_argument("--input-data", type=str, required=True,
                       help="Path para os dados de entrada (CSV)")
    parser.add_argument("--output-path", type=str, default="outputs/predictions.csv",
                       help="Path para salvar as predições")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Tamanho do batch para processamento")
    
    return parser.parse_args()


def load_model(model_path):
    """Carrega modelo treinado"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print(f"📥 Carregando modelo de: {model_path}")
    model = joblib.load(model_path)
    print(f"✅ Modelo carregado: {type(model).__name__}")
    
    return model


def preprocess_input_data(df):
    """
    Preprocessa dados de entrada para inferência
    Aplica o mesmo preprocessamento usado no treinamento
    """
    print("🔄 Preprocessando dados de entrada...")
    
    # Se temos a coluna target, removemos para inferência
    if 'default' in df.columns:
        print("⚠️  Removendo coluna target dos dados de entrada")
        df = df.drop('default', axis=1)
    
    # Aplicar mesmo preprocessamento do treinamento
    # Note: Em produção, você salvaria o preprocessor junto com o modelo
    df_processed = df.copy()
    
    # Simular preprocessamento (em produção seria mais robusto)
    # Encoding categóricas
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].dtype == 'object':
            # Para simplificar, usar label encoding
            unique_vals = df_processed[col].unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            df_processed[col] = df_processed[col].map(mapping)
    
    # Tratar missing values
    df_processed = df_processed.fillna(df_processed.median())
    
    print(f"✅ Preprocessamento concluído. Shape: {df_processed.shape}")
    return df_processed


def make_predictions(model, X, batch_size=1000):
    """
    Faz predições em batches
    
    Args:
        model: Modelo treinado
        X: Features preprocessadas
        batch_size: Tamanho do batch
        
    Returns:
        tuple: (predictions, probabilities)
    """
    n_samples = len(X)
    predictions = []
    probabilities = []
    
    print(f"🔮 Fazendo predições para {n_samples} amostras...")
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X.iloc[i:end_idx] if hasattr(X, 'iloc') else X[i:end_idx]
        
        # Predições
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)
        
        # Probabilidades (se disponível)
        if hasattr(model, 'predict_proba'):
            batch_proba = model.predict_proba(batch)[:, 1]  # Probabilidade da classe positiva
            probabilities.extend(batch_proba)
        
        if i % (batch_size * 10) == 0:
            print(f"   Processado: {min(end_idx, n_samples)}/{n_samples} amostras")
    
    return np.array(predictions), np.array(probabilities) if probabilities else None


def save_predictions(predictions, probabilities, output_path, original_data=None):
    """Salva predições em CSV"""
    results_df = pd.DataFrame()
    
    # Adicionar dados originais se fornecidos
    if original_data is not None:
        results_df = original_data.copy()
    
    # Adicionar predições
    results_df['prediction'] = predictions
    results_df['risk_category'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
    
    # Adicionar probabilidades se disponíveis
    if probabilities is not None:
        results_df['default_probability'] = probabilities
        results_df['confidence'] = np.where(
            predictions == 1, 
            probabilities, 
            1 - probabilities
        )
        
        # Categorias de risco baseadas em probabilidade
        results_df['risk_tier'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar
    results_df.to_csv(output_path, index=False)
    print(f"💾 Predições salvas em: {output_path}")
    
    return results_df


def generate_prediction_summary(predictions, probabilities=None):
    """Gera resumo das predições"""
    total_samples = len(predictions)
    high_risk_count = np.sum(predictions == 1)
    low_risk_count = np.sum(predictions == 0)
    
    summary = {
        'total_samples': int(total_samples),
        'high_risk_count': int(high_risk_count),
        'low_risk_count': int(low_risk_count),
        'high_risk_percentage': float(high_risk_count / total_samples * 100),
        'low_risk_percentage': float(low_risk_count / total_samples * 100)
    }
    
    if probabilities is not None:
        summary.update({
            'mean_probability': float(np.mean(probabilities)),
            'std_probability': float(np.std(probabilities)),
            'min_probability': float(np.min(probabilities)),
            'max_probability': float(np.max(probabilities))
        })
    
    return summary


def main():
    """Função principal de inferência"""
    args = parse_args()
    
    print("🚀 Iniciando processo de inferência...")
    start_time = datetime.now()
    
    # Carregar modelo
    model = load_model(args.model_path)
    
    # Carregar dados
    print(f"📥 Carregando dados de entrada: {args.input_data}")
    try:
        input_df = pd.read_csv(args.input_data)
        print(f"📊 Dados carregados: {input_df.shape[0]} linhas, {input_df.shape[1]} colunas")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return
    
    # Preprocessar dados
    try:
        X_processed = preprocess_input_data(input_df)
    except Exception as e:
        print(f"❌ Erro no preprocessamento: {e}")
        return
    
    # Fazer predições
    try:
        predictions, probabilities = make_predictions(model, X_processed, args.batch_size)
    except Exception as e:
        print(f"❌ Erro nas predições: {e}")
        return
    
    # Salvar resultados
    try:
        results_df = save_predictions(predictions, probabilities, args.output_path, input_df)
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
        return
    
    # Gerar resumo
    summary = generate_prediction_summary(predictions, probabilities)
    
    # Salvar resumo
    summary_path = args.output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Tempo total
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Report final
    print("\n" + "="*60)
    print("📊 RESUMO DAS PREDIÇÕES")
    print("="*60)
    print(f"Total de amostras: {summary['total_samples']:,}")
    print(f"Alto risco: {summary['high_risk_count']:,} ({summary['high_risk_percentage']:.1f}%)")
    print(f"Baixo risco: {summary['low_risk_count']:,} ({summary['low_risk_percentage']:.1f}%)")
    
    if probabilities is not None:
        print(f"\nProbabilidades:")
        print(f"   Média: {summary['mean_probability']:.3f}")
        print(f"   Desvio padrão: {summary['std_probability']:.3f}")
        print(f"   Min: {summary['min_probability']:.3f}")
        print(f"   Max: {summary['max_probability']:.3f}")
    
    print(f"\nTempo total: {total_time:.1f}s")
    print(f"Throughput: {len(predictions)/total_time:.1f} predições/segundo")
    print(f"\nResultados salvos em: {args.output_path}")
    print(f"Resumo salvo em: {summary_path}")
    print("="*60)


# Função para compatibilidade com Azure ML
def init():
    """Inicialização para Azure ML endpoint"""
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR', '.'), 'model.pkl')
    model = load_model(model_path)


def run(raw_data):
    """Função de inferência para Azure ML endpoint"""
    try:
        # Parse input data
        data = json.loads(raw_data)['data']
        df = pd.DataFrame(data)
        
        # Preprocess
        X_processed = preprocess_input_data(df)
        
        # Predict
        predictions, probabilities = make_predictions(model, X_processed)
        
        # Format response
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': int(pred),
                'risk_category': 'High Risk' if pred == 1 else 'Low Risk'
            }
            
            if probabilities is not None:
                result['probability'] = float(probabilities[i])
                result['confidence'] = float(probabilities[i] if pred == 1 else 1 - probabilities[i])
            
            results.append(result)
        
        return results
        
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    main()
