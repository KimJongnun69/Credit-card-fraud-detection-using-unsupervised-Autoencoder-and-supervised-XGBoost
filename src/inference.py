import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from .preprocessing import load_config
from .autoencoder import Autoencoder, get_device
import os

class FraudInferencePipeline:
    def __init__(self, config_path='configs/config.yaml'):
        # Determine the project root assuming inference.py is in src/
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config = load_config(os.path.join(self.base_dir, config_path))
        
        self.device = get_device()
        self.input_dim = 30 # Time_scaled + V1..V28 + Amount_scaled
        
        self.load_models()
        
    def load_models(self):
        # Load Autoencoder
        self.ae_model = Autoencoder(
            input_dim=self.input_dim, 
            latent_dim=self.config['model']['latent_dim'], 
            dropout=self.config['model']['dropout']
        )
        ae_path = os.path.join(self.base_dir, 'models/autoencoder_baseline.pth')
        self.ae_model.load_state_dict(torch.load(ae_path, map_location=self.device, weights_only=True))
        self.ae_model.to(self.device)
        self.ae_model.eval()
        
        # Load Threshold
        # Default to a safe high threshold if not yet set
        self.threshold = self.config.get('threshold', {}).get('value', 20.0) 
        
        # Load XGBoost
        xgb_path = os.path.join(self.base_dir, 'models/xgboost_baseline.pkl')
        self.xgb_model = joblib.load(xgb_path)
        
    def predict(self, feature_array):
        """
        Predicts fraud for a single transaction or batch of transactions.
        feature_array should be preprocessed (Time_scaled, V1-V28, Amount_scaled).
        """
        # Convert to numpy if needed
        if isinstance(feature_array, pd.DataFrame):
            x_np = feature_array.values
        else:
            x_np = np.array(feature_array)
            
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            
        # 1. Autoencoder Prediction
        x_tensor = torch.FloatTensor(x_np).to(self.device)
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    reconstructed = self.ae_model(x_tensor)
                    ae_scores = torch.mean((x_tensor - reconstructed)**2, dim=1).cpu().numpy()
            else:
                reconstructed = self.ae_model(x_tensor)
                ae_scores = torch.mean((x_tensor - reconstructed)**2, dim=1).cpu().numpy()
                
        ae_decisions = (ae_scores >= self.threshold).astype(int)
        
        # 2. XGBoost Prediction
        xgb_probs = self.xgb_model.predict_proba(x_np)[:, 1]
        xgb_decisions = (xgb_probs >= 0.5).astype(int)
        
        return {
            'ae_anomaly_scores': ae_scores,
            'ae_decisions': ae_decisions,
            'xgb_probabilities': xgb_probs,
            'xgb_decisions': xgb_decisions
        }
