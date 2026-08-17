import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.autoencoder import Autoencoder

def test_autoencoder_shape():
    input_dim = 30
    latent_dim = 14
    batch_size = 32
    
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, dropout=0.2)
    
    # Create dummy input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Output shape should match input shape
    assert output.shape == (batch_size, input_dim), f"Expected shape {(batch_size, input_dim)}, got {output.shape}"

def test_autoencoder_components():
    input_dim = 30
    latent_dim = 14
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'decoder')
    
    # Check that the latent bottleneck is the correct size
    # The last linear layer of the encoder should output latent_dim
    assert model.encoder[-2].out_features == latent_dim
