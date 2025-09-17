"""
Utilitários para avaliação de modelos de machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Avaliação completa do modelo
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_name: Nome do modelo para logs
        
    Returns:
        dict: Dicionário com todas as métricas
    """
    # Predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métricas básicas
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'support': len(y_test)
    }
    
    # Métricas que precisam de probabilidades
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Average Precision Score
        precision_scores, recall_scores, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['avg_precision'] = np.trapz(precision_scores, recall_scores)
    
    return metrics


def plot_model_evaluation(model, X_test, y_test, model_name="Model", save_path=None):
    """
    Cria visualizações completas da avaliação do modelo
    
    Args:
        model: Modelo treinado
        X_test: Features de teste  
        y_test: Target de teste
        model_name: Nome do modelo
        save_path: Path para salvar o plot (opcional)
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} - Avaliação Completa', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Matriz de Confusão')
    axes[0, 0].set_xlabel('Predito')
    axes[0, 0].set_ylabel('Real')
    
    # 2. ROC Curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('Taxa de Falsos Positivos')
        axes[0, 1].set_ylabel('Taxa de Verdadeiros Positivos')
        axes[0, 1].set_title('Curva ROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 3. Precision-Recall Curve
    if y_pred_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[0, 2].plot(recall, precision, label=f'PR Curve')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Curva Precision-Recall')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # 4. Distribuição de Probabilidades
    if y_pred_proba is not None:
        axes[1, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Não Default', color='blue')
        axes[1, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Default', color='red')
        axes[1, 0].set_xlabel('Probabilidade Predita')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição das Probabilidades')
        axes[1, 0].legend()
    
    # 5. Calibration Plot
    if y_pred_proba is not None:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
        axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfeitamente calibrado")
        axes[1, 1].set_xlabel('Probabilidade Média Predita')
        axes[1, 1].set_ylabel('Fração de Positivos')
        axes[1, 1].set_title('Calibração do Modelo')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # 6. Feature Importance (se disponível)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
        if hasattr(X_test, 'columns'):
            feature_names = X_test.columns.tolist()
        
        # Top 10 features mais importantes
        top_indices = np.argsort(importances)[-10:]
        top_importances = importances[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        axes[1, 2].barh(range(len(top_importances)), top_importances)
        axes[1, 2].set_yticks(range(len(top_importances)))
        axes[1, 2].set_yticklabels(top_names)
        axes[1, 2].set_xlabel('Importância')
        axes[1, 2].set_title('Top 10 Features Mais Importantes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_models(models_dict, X_test, y_test):
    """
    Compara múltiplos modelos lado a lado
    
    Args:
        models_dict: Dicionário {'nome_modelo': modelo_treinado}
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        pd.DataFrame: DataFrame com comparação das métricas
    """
    results = []
    
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    
    # Ordenar por AUC (se disponível) ou F1-Score
    sort_column = 'auc' if 'auc' in comparison_df.columns else 'f1_score'
    comparison_df = comparison_df.sort_values(sort_column, ascending=False)
    
    return comparison_df


def plot_model_comparison(comparison_df, save_path=None):
    """
    Visualiza comparação entre modelos
    
    Args:
        comparison_df: DataFrame retornado por compare_models()
        save_path: Path para salvar o plot (opcional)
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    if 'auc' in comparison_df.columns:
        metrics_to_plot.append('auc')
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4*len(metrics_to_plot), 6))
    
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics_to_plot):
        axes[i].bar(comparison_df['model_name'], comparison_df[metric])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Adicionar valores no topo das barras
        for j, v in enumerate(comparison_df[metric]):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Comparação de Modelos', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_model_report(model, X_test, y_test, model_name="Model"):
    """
    Gera relatório textual completo do modelo
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_name: Nome do modelo
        
    Returns:
        str: Relatório formatado
    """
    y_pred = model.predict(X_test)
    
    report = f"\n{'='*60}\n"
    report += f"RELATÓRIO DE AVALIAÇÃO - {model_name}\n"
    report += f"{'='*60}\n\n"
    
    # Métricas principais
    metrics = evaluate_model(model, X_test, y_test, model_name)
    
    report += "📊 MÉTRICAS PRINCIPAIS:\n"
    report += f"   • Accuracy:  {metrics['accuracy']:.4f}\n"
    report += f"   • Precision: {metrics['precision']:.4f}\n"
    report += f"   • Recall:    {metrics['recall']:.4f}\n"
    report += f"   • F1-Score:  {metrics['f1_score']:.4f}\n"
    
    if 'auc' in metrics:
        report += f"   • AUC:       {metrics['auc']:.4f}\n"
    
    report += f"   • Amostras:  {metrics['support']}\n\n"
    
    # Classification Report
    report += "📋 CLASSIFICATION REPORT:\n"
    report += classification_report(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    report += f"\n🎯 MATRIZ DE CONFUSÃO:\n"
    report += f"   Verdadeiros Negativos:  {cm[0, 0]}\n"
    report += f"   Falsos Positivos:       {cm[0, 1]}\n"
    report += f"   Falsos Negativos:       {cm[1, 0]}\n"
    report += f"   Verdadeiros Positivos:  {cm[1, 1]}\n"
    
    return report
