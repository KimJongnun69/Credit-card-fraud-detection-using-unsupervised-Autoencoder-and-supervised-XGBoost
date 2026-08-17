import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_xgboost(X_train, y_train, scale_pos_weight=None, params=None):
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
    
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
        
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_xgboost(model, X_val, y_val, plot=False):
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Threshold at 0.5 for simple predictions (although could be tuned)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate PR curve and AUC
    precisions, recalls, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Calculate precision, recall, f1 for threshold 0.5
    # Handle zero division
    tp = np.sum((y_pred == 1) & (y_val == 1))
    fp = np.sum((y_pred == 1) & (y_val == 0))
    fn = np.sum((y_pred == 0) & (y_val == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PR Curve
        axes[0].plot(recalls, precisions, label=f'PR-AUC = {pr_auc:.4f}')
        axes[0].set_title('Precision-Recall Curve (XGBoost)')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend()
        axes[0].grid(True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                    xticklabels=['Legitimate', 'Fraud'], 
                    yticklabels=['Legitimate', 'Fraud'])
        axes[1].set_title('Confusion Matrix (XGBoost)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
        
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
