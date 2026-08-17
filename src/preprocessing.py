import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import yaml

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_path):
    return pd.read_csv(data_path)

def split_data_anomaly_detection(df, val_size=0.2, test_size=0.2, random_state=42):
    """
    Splits the dataset such that the training set contains ONLY legitimate transactions.
    Fraud transactions are split evenly between validation and test sets.
    """
    # Separate classes
    df_legit = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]
    
    # Split legitimate data
    # Calculate proportions
    train_size = 1.0 - val_size - test_size
    # First split into train and temp (val+test)
    X_train_legit, X_temp_legit = train_test_split(
        df_legit, test_size=(val_size + test_size), random_state=random_state
    )
    
    # Split temp into val and test
    relative_test_size = test_size / (val_size + test_size)
    X_val_legit, X_test_legit = train_test_split(
        X_temp_legit, test_size=relative_test_size, random_state=random_state
    )
    
    # Split fraud data evenly into val and test (50/50)
    X_val_fraud, X_test_fraud = train_test_split(
        df_fraud, test_size=0.5, random_state=random_state
    )
    
    # Combine back to form full val and test sets
    train_df = X_train_legit.copy()
    val_df = pd.concat([X_val_legit, X_val_fraud]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([X_test_legit, X_test_fraud]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_df, val_df, test_df

def preprocess_features(train_df, val_df, test_df):
    """
    Scales the Amount and Time features.
    V1-V28 are already PCA transformed and generally scaled, but we'll leave them as is.
    We use RobustScaler for Amount to handle outliers.
    """
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()
    
    # Fit only on training data (which is entirely legitimate)
    train_df['Amount_scaled'] = scaler_amount.fit_transform(train_df[['Amount']])
    train_df['Time_scaled'] = scaler_time.fit_transform(train_df[['Time']])
    
    val_df['Amount_scaled'] = scaler_amount.transform(val_df[['Amount']])
    val_df['Time_scaled'] = scaler_time.transform(val_df[['Time']])
    
    test_df['Amount_scaled'] = scaler_amount.transform(test_df[['Amount']])
    test_df['Time_scaled'] = scaler_time.transform(test_df[['Time']])
    
    # Drop original columns and reorder
    cols_to_drop = ['Amount', 'Time']
    
    # Get all features except Class and the ones to drop
    feature_cols = [c for c in train_df.columns if c not in cols_to_drop + ['Class']]
    
    # Make sure we reorder to have Time_scaled, V1..V28, Amount_scaled
    final_features = ['Time_scaled'] + [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']
    
    X_train = train_df[final_features].values
    y_train = train_df['Class'].values
    
    X_val = val_df[final_features].values
    y_val = val_df['Class'].values
    
    X_test = test_df[final_features].values
    y_test = test_df['Class'].values
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (scaler_time, scaler_amount)
