import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import split_data_anomaly_detection, preprocess_features

def test_split_data():
    # Create dummy data
    np.random.seed(42)
    n_legit = 1000
    n_fraud = 50
    
    df_legit = pd.DataFrame(np.random.randn(n_legit, 30), columns=[f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'])
    df_legit['Class'] = 0
    
    df_fraud = pd.DataFrame(np.random.randn(n_fraud, 30), columns=[f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'])
    df_fraud['Class'] = 1
    
    df = pd.concat([df_legit, df_fraud]).sample(frac=1).reset_index(drop=True)
    
    train_df, val_df, test_df = split_data_anomaly_detection(df, val_size=0.2, test_size=0.2)
    
    # 1. Train set must have ONLY legitimate transactions
    assert train_df['Class'].sum() == 0
    
    # 2. Fraud must be split between val and test
    assert val_df['Class'].sum() > 0
    assert test_df['Class'].sum() > 0
    
    # 3. Total fraud should match
    assert val_df['Class'].sum() + test_df['Class'].sum() == n_fraud

def test_preprocess_features():
    # Dummy data
    np.random.seed(42)
    n = 100
    train_df = pd.DataFrame(np.random.randn(n, 30), columns=[f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'])
    train_df['Class'] = 0
    
    val_df = pd.DataFrame(np.random.randn(20, 30), columns=[f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'])
    val_df['Class'] = np.random.randint(0, 2, 20)
    
    test_df = pd.DataFrame(np.random.randn(20, 30), columns=[f'V{i}' for i in range(1, 29)] + ['Amount', 'Time'])
    test_df['Class'] = np.random.randint(0, 2, 20)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = preprocess_features(train_df, val_df, test_df)
    
    # Output arrays must have 30 columns (Time_scaled, V1-V28, Amount_scaled)
    assert X_train.shape[1] == 30
    assert X_val.shape[1] == 30
    assert X_test.shape[1] == 30
    
    # y must be correct
    assert (y_train == train_df['Class'].values).all()
