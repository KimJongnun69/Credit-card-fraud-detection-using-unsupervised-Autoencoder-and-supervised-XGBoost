import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=14, dropout=0.2):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * 0.75)),
            nn.BatchNorm1d(int(input_dim * 0.75)),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.Linear(int(input_dim * 0.75), int(input_dim * 0.5)),
            nn.BatchNorm1d(int(input_dim * 0.5)),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.Linear(int(input_dim * 0.5), latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, int(input_dim * 0.5)),
            nn.BatchNorm1d(int(input_dim * 0.5)),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.Linear(int(input_dim * 0.5), int(input_dim * 0.75)),
            nn.BatchNorm1d(int(input_dim * 0.75)),
            nn.ReLU(True),
            nn.Dropout(dropout),
            
            nn.Linear(int(input_dim * 0.75), input_dim)
            # No activation at the end, assuming input is scaled and can be anything
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_autoencoder(model, train_loader, val_loader, config, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['training']['learning_rate'], 
                           weight_decay=float(config['training']['weight_decay']))
    
    epochs = config['training']['epochs']
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    # Enable mixed precision scaler for CUDA
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    train_losses = []
    val_losses = []
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x[0].to(device)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_x)
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    
                val_loss += loss.item() * batch_x.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}")
        
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    return model, train_losses, val_losses

def compute_reconstruction_error(model, dataloader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch_x in dataloader:
            batch_x = batch_x[0].to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    reconstructed = model(batch_x)
                    # MSE per sample across features
                    err = torch.mean((batch_x - reconstructed)**2, dim=1).cpu().numpy()
            else:
                reconstructed = model(batch_x)
                err = torch.mean((batch_x - reconstructed)**2, dim=1).cpu().numpy()
            errors.append(err)
            
    return np.concatenate(errors)
