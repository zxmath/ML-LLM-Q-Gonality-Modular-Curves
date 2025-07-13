import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os
import json
from datetime import datetime


# --- 1. Custom Dataset for Pre-processed Tensors ---
class MultiTypeDataset(Dataset):
    def __init__(self, targets, numerical_features=None, categorical_features=None, list_features=None):
        self.targets = targets
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features or {}
        self.list_features = list_features
        
        # Ensure all features have the same length
        self.length = len(targets)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Prepare feature dictionary
        features = {}
        
        # Add numerical features
        if self.numerical_features is not None:
            features['numerical'] = self.numerical_features[idx]
        
        # Add categorical features
        for cat_name, cat_tensor in self.categorical_features.items():
            features[cat_name] = cat_tensor[idx]
        
        # Add list features
        if self.list_features is not None:
            features['list'] = self.list_features[idx]
        
        return features, self.targets[idx]


# --- 2. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# ...existing code...
import torch.nn as nn # Ensure nn is imported

class FeatureTokenizer(nn.Module):
    def __init__(self, d_model, numerical_features, categorical_configs, list_feature_dim):
        super(FeatureTokenizer, self).__init__()
        self.d_model = d_model
        
        # Numerical feature processing
        if numerical_features > 0:
            self.numerical_linear = nn.Linear(numerical_features, d_model)
            # self.numerical_bn = nn.BatchNorm1d(d_model) # Replace BatchNorm
            self.numerical_norm = nn.LayerNorm(d_model)   # Use LayerNorm
        else:
            self.numerical_linear = None
            # self.numerical_bn = None
            self.numerical_norm = None

        # Categorical feature processing
        self.categorical_embeddings = nn.ModuleDict()
        if categorical_configs:
            for name, num_categories in categorical_configs.items():
                self.categorical_embeddings[name] = nn.Embedding(num_categories, d_model)
        
        # List feature processing (assuming list_feature_dim is the length of the padded list)
        if list_feature_dim > 0:
            self.list_linear = nn.Linear(list_feature_dim, d_model) # Project list features to d_model
            # self.list_bn = nn.BatchNorm1d(d_model) # Replace BatchNorm
            self.list_norm = nn.LayerNorm(d_model)   # Use LayerNorm
        else:
            self.list_linear = None
            # self.list_bn = None
            self.list_norm = None

    def forward(self, features):
        tokens = []
        
        # Process numerical features
        if self.numerical_linear and 'numerical' in features and features['numerical'] is not None:
            num_data = features['numerical']
            if num_data.ndim == 1: # Ensure it's at least 2D for linear layer
                num_data = num_data.unsqueeze(1)
            
            # Check if num_data has features before linear projection
            if num_data.size(1) > 0:
                num_tokens = self.numerical_linear(num_data) # Shape: (batch_size, d_model)
                # num_tokens = self.numerical_bn(num_tokens)   # Apply LayerNorm instead
                num_tokens = self.numerical_norm(num_tokens)
                tokens.append(num_tokens.unsqueeze(1)) # Shape: (batch_size, 1, d_model)
            elif num_data.size(0) > 0 : # Handle case where num_data is (batch_size, 0)
                 # Create a zero tensor of the correct shape if no numerical features but batch exists
                zero_tokens = torch.zeros(num_data.size(0), self.d_model, device=num_data.device)
                tokens.append(zero_tokens.unsqueeze(1))


        # Process categorical features
        for name, embedding_layer in self.categorical_embeddings.items():
            if name in features and features[name] is not None:
                cat_tokens = embedding_layer(features[name]) # Shape: (batch_size, d_model)
                tokens.append(cat_tokens.unsqueeze(1)) # Shape: (batch_size, 1, d_model)
        
        # Process list features
        if self.list_linear and 'list' in features and features['list'] is not None:
            list_data = features['list'] # Shape: (batch_size, list_feature_dim)
            if list_data.size(1) > 0: # Check if list_data has features
                list_tokens = self.list_linear(list_data) # Shape: (batch_size, d_model)
                # list_tokens = self.list_bn(list_tokens)   # Apply LayerNorm instead
                list_tokens = self.list_norm(list_tokens)
                tokens.append(list_tokens.unsqueeze(1)) # Shape: (batch_size, 1, d_model)
            elif list_data.size(0) > 0: # Handle case where list_data is (batch_size, 0)
                zero_tokens = torch.zeros(list_data.size(0), self.d_model, device=list_data.device)
                tokens.append(zero_tokens.unsqueeze(1))


        if tokens:
            # Concatenate all tokens along the sequence dimension
            return torch.cat(tokens, dim=1)  # Shape: (batch_size, seq_len, d_model)
        else:
            # Return empty tensor if no features, try to get batch_size from any input tensor
            batch_size_val = 1 # Default
            if 'numerical' in features and features['numerical'] is not None:
                batch_size_val = features['numerical'].size(0)
            elif self.categorical_embeddings:
                first_cat_key = next(iter(self.categorical_embeddings.keys()))
                if first_cat_key in features and features[first_cat_key] is not None:
                    batch_size_val = features[first_cat_key].size(0)
            elif 'list' in features and features['list'] is not None:
                 batch_size_val = features['list'].size(0)
            
            # Determine device from available tensors or default to CPU
            device_val = next((t.device for t in features.values() if isinstance(t, torch.Tensor) and t is not None), torch.device('cpu'))

            return torch.empty(batch_size_val, 0, self.d_model, device=device_val)
# ...existing code...


# --- 4. FT-Transformer Model ---
class FTTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, 
                 dropout=0.1, output_dim=1, numerical_features=0, 
                 categorical_configs=None, list_feature_dim=0, activation='relu'):
        super(FTTransformer, self).__init__()
        
        self.d_model = d_model
        self.feature_tokenizer = FeatureTokenizer(
            d_model, numerical_features, categorical_configs, list_feature_dim
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Important: batch_first=True for easier handling
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers with improved architecture
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                
    def forward(self, features):
        # Tokenize features
        tokens = self.feature_tokenizer(features)  # Shape: (batch_size, seq_len, d_model)
        
        if tokens.size(1) == 0:  # No features
            # Return zeros if no features
            batch_size = next(iter(features.values())).size(0)
            return torch.zeros(batch_size, 1, device=next(iter(features.values())).device)
        
        # Apply positional encoding
        tokens = tokens.transpose(0, 1)  # Shape: (seq_len, batch_size, d_model)
        tokens = self.pos_encoder(tokens)
        tokens = tokens.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(tokens)  # Shape: (batch_size, seq_len, d_model)
        
        # Global average pooling across sequence dimension
        pooled = torch.mean(encoded, dim=1)  # Shape: (batch_size, d_model)
        
        # Generate output
        output = self.output_layers(pooled)  # Shape: (batch_size, output_dim)
        
        return output


# --- 5. Training Functions ---
# Fix the train_ft_transformer function - remove verbose from ReduceLROnPlateau

def convert_tensors_to_numbers(data_list):
    """
    Convert tensor values in a list to Python numbers for JSON serialization.
    """
    converted = []
    for item in data_list:
        if torch.is_tensor(item):
            converted.append(item.item())
        else:
            converted.append(item)
    return converted

def train_ft_transformer(model, train_loader, val_loader=None, 
                         epochs=100, learning_rate=1e-3, weight_decay=1e-4,
                         device='cuda', patience=10, verbose=True, config=None, fold_idx=None):
    """
    Train FT-Transformer model with proper early stopping and checkpointing
    """
    model = model.to(device)
    model.train()
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use config for paths if provided, otherwise use defaults
    if config and 'info' in config:
        model_save_dir = os.path.join(config['info']['saved_model_path'], "FT_models")
        results_save_dir = "results/FT_results"
    else:
        model_save_dir = "saved_models/FT_models"
        results_save_dir = "results/FT_results"
    
    # Create directories
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Move data to device
            targets = targets.to(device)
            for key, value in features.items():
                features[key] = value.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)

            train_accuracy = (torch.floor(outputs.squeeze()+0.5)== targets).float().mean()
            if verbose:
                print(f"Batch {batch_idx+1}/{len(train_loader)}: Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}")

            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_accuracy += train_accuracy
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        avg_train_accuracy = epoch_train_accuracy / num_batches
        train_accuracies.append(avg_train_accuracy)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_accuracy = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for features, targets in val_loader:
                    targets = targets.to(device)
                    for key, value in features.items():
                        features[key] = value.to(device)
                    
                    outputs = model(features)
                    val_loss = criterion(outputs.squeeze(), targets)
                    val_accuracy = (torch.floor(outputs.squeeze()+0.5)== targets).float().mean()
                    
                    epoch_val_loss += val_loss.item()
                    epoch_val_accuracy += val_accuracy
                    val_batches += 1
            
            avg_val_loss = epoch_val_loss / val_batches
            avg_val_accuracy = epoch_val_accuracy / val_batches
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping and model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
                
                # Save best model with fold information
                if fold_idx is not None:
                    best_model_path = os.path.join(model_save_dir, f"best_ft_transformer_fold_{fold_idx}_{timestamp}.pth")
                else:
                    best_model_path = os.path.join(model_save_dir, f"best_ft_transformer_{timestamp}.pth")
                torch.save(best_model_state, best_model_path)
                if verbose:
                    print(f"Saved best model to: {best_model_path}")
            else:
                epochs_without_improvement += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            # No validation set - just print training loss
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}")
   
    # Load best model if we have validation
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    # Convert tensors to numbers for JSON serialization
    train_accuracies_converted = convert_tensors_to_numbers(train_accuracies)
    val_accuracies_converted = convert_tensors_to_numbers(val_accuracies)
    
    # Save training results with fold information
    results = {
        'timestamp': timestamp,
        'fold_idx': fold_idx,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies_converted,
        'val_accuracies': val_accuracies_converted,
        'best_val_loss': best_val_loss if val_loader is not None else None,
        'final_epoch': len(train_losses),
        'training_params': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'patience': patience,
            
        }
    }
    
    if fold_idx is not None:
        results_path = os.path.join(results_save_dir, f"training_results_fold_{fold_idx}_{timestamp}.json")
    else:
        results_path = os.path.join(results_save_dir, f"training_results_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"Saved training results to: {results_path}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def predict_ft_transformer(model, test_loader, device='cuda'):
    """
    Improved prediction function with better error handling
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch_features, _ in test_loader:
            # Move data to device
            batch_features = {k: v.to(device) for k, v in batch_features.items()}
            
            # Forward pass
            outputs = model(batch_features)
            predictions.append(outputs.cpu())
    
    return torch.cat(predictions, dim=0)


# --- 6. Utility Functions ---
def prepare_categorical_configs(df, categorical_columns):
    """
    Prepare categorical configurations for the model
    """
    configs = {}
    for col in categorical_columns:
        if col in df.columns:
            configs[col] = df[col].nunique()
    return configs


def create_ft_transformer_dataloaders(df, target_col, numerical_cols, categorical_cols, 
                                    list_cols, test_size=0.2, val_size=0.1, 
                                    batch_size=64, random_state=42):
    """
    Create train/val/test dataloaders for FT-Transformer
    """
    # This function is kept for backward compatibility but 
    # the notebook implementation is preferred for better control
    pass
