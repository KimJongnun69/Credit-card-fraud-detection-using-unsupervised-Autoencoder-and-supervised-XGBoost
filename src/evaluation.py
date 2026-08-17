import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_anomaly_scores(y_true, anomaly_scores, plot=False, title='Precision-Recall Curve'):
    """
    Evaluates anomaly scores using PR-AUC and finds the best threshold maximizing F1-score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = auc(recalls, precisions)
    
    # Calculate F1 scores for each threshold to find the best one
    # f1 = 2 * (p * r) / (p + r). Handle divide by zero.
    f1_scores = np.divide(2 * precisions[:-1] * recalls[:-1], (precisions[:-1] + recalls[:-1]), out=np.zeros_like(precisions[:-1]), where=(precisions[:-1] + recalls[:-1]) != 0)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label=f'PR-AUC = {pr_auc:.4f}')
        plt.scatter([best_recall], [best_precision], color='red', label=f'Best F1 Threshold ({best_threshold:.4f})')
        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return {
        'pr_auc': float(pr_auc),
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'best_precision': float(best_precision),
        'best_recall': float(best_recall)
    }

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'], 
                yticklabels=['Legitimate', 'Fraud'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
