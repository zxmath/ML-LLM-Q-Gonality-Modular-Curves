import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
from datetime import datetime
import ast
from FT import train_ft_transformer, predict_ft_transformer
# Import FT-Transformer modules
import sys
# Add this after your other imports


# Add the project root to Python path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
import evaluation

try:
    # Try importing from models package
    from models.FT_Transformer import FT_data
    from models.FT_Transformer import FT
    from src.utils.get_config import load_toml_config
except ImportError:
    # Fallback to direct imports if running from within the directory
    import FT_data
    import FT
    # Import get_config from absolute path
    sys.path.insert(0, os.path.join(project_root, 'src', 'utils'))
    import get_config
    load_toml_config = get_config.load_toml_config
from FT_data import prepare_ft_data
from FT import MultiTypeDataset
from FT import FTTransformer
def main():
    print("Starting FT-Transformer OOT Experiment")
    print("=" * 60)
    
    # Load configuration
    config = load_toml_config("config/modular.toml")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, "results", "FT_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create saved models directory if it doesn't exist
    models_dir = os.path.join(project_root, "saved_models", "FT_models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Store results for all folds
    
    train_evaluation = {} 
    val_evaluation = {}
    test_evaluation = {}
    diff_evaluation = {}

    print(f"\n{'='*20} OOT {'='*20}")
        
        # Load fold data
    data_dir = os.path.join(project_root, "data", "Cross-Validation-Data")
    train_file = os.path.join(data_dir, f"train_small.csv")
    valid_file = os.path.join(data_dir, f"valid_small.csv")
    test_file = os.path.join(data_dir, f"level_large.csv")
    diff_file = os.path.join(data_dir, f"q_gonality_diff.csv")
    
    print(f"Loading training data from: {train_file}")
    print(f"Loading validation data from: {valid_file}")
    
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(valid_file)
    df_test = pd.read_csv(test_file)
    qonality_diff = pd.read_csv(diff_file)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Validation data shape: {df_val.shape}")
    
    # Prepare data using FT_data functions
    print("\nPreparing data...")
    
    # Get feature configurations from config
    numerical_columns = config['FT_Transformer']['numerical_features']
    categorical_columns = config['FT_Transformer']['catorigical_features']  # Note: typo in config
    list_columns = config['FT_Transformer']['list_features']
    target_column = config['FT_Transformer']['target']
    
    print(f"Numerical features: {numerical_columns}")
    print(f"Categorical features: {categorical_columns}")
    print(f"List features: {list_columns}")
    print(f"Target: {target_column}")
    print("Preparing training data...")


            # Prepare training data
    print("Preparing training data...")
    numerical_train, categorical_train, list_train, targets_train, scaler, encoders, max_list_len = prepare_ft_data(
        df_train[numerical_columns + categorical_columns + list_columns], 
        df_train[target_column], 
        categorical_columns, 
        list_columns
    )

    print(f"Training data prepared:")
    print(f"  Numerical shape: {numerical_train.shape if numerical_train is not None else 'None'}")
    print(f"  Categorical data: {list(categorical_train.keys()) if categorical_train else 'None'}")
    print(f"  List shape: {list_train.shape if list_train is not None else 'None'}")
    print(f"  Max list length: {max_list_len}")
    print(f"  Targets shape: {targets_train.shape}")

    # filepath: /home/zxmath/Machine_Learning_Arithmetic-_object/notebook/model_test_5.ipynb
    # ...existing code...
    # Add this cell to check for NaNs in preprocessed tensors
    print("Checking for NaNs in training tensors:")
    if numerical_train is not None:
        print(f"  numerical_train NaNs: {torch.isnan(numerical_train).any().item()}")
    for name, tensor in categorical_train.items():
        print(f"  categorical_train['{name}'] NaNs: {torch.isnan(tensor.float()).any().item()}") # Cast to float for isnan if long
    if list_train is not None:
        print(f"  list_train NaNs: {torch.isnan(list_train).any().item()}")
    print(f"  targets_train NaNs: {torch.isnan(targets_train).any().item()}")
    # Prepare validation and test data using fitted transformers
    print("Preparing validation data...")
    numerical_val, categorical_val, list_val, targets_val, _, _, _ = prepare_ft_data(
        df_val[numerical_columns + categorical_columns + list_columns], 
        df_val[target_column], 
        categorical_columns, 
        list_columns,
        fitted_scaler=scaler,
        fitted_encoders=encoders,
        max_list_len=max_list_len
    )

    print("Preparing test data...")
    numerical_test, categorical_test, list_test, targets_test, _, _, _ = prepare_ft_data(
        df_test[numerical_columns + categorical_columns + list_columns], 
        df_test[target_column], 
        categorical_columns, 
        list_columns,
        fitted_scaler=scaler,
        fitted_encoders=encoders,
        max_list_len=max_list_len
    )

    print("Preparing q_gonality_diff data...")
    numerical_diff, categorical_diff, list_diff, _, _, _, _ = prepare_ft_data(
        qonality_diff[numerical_columns + categorical_columns + list_columns], 
        pd.Series([0] * len(qonality_diff)),  # Dummy targets
        categorical_columns, 
        list_columns,
        fitted_scaler=scaler,
        fitted_encoders=encoders,
        max_list_len=max_list_len
    )

    print("All data prepared successfully!")

    # Create datasets and data loaders
    print("Creating datasets...")

    train_dataset = MultiTypeDataset(targets_train, numerical_train, categorical_train, list_train)
    val_dataset = MultiTypeDataset(targets_val, numerical_val, categorical_val, list_val)

    print("Creating datasets...")

    train_dataset = MultiTypeDataset(targets_train, numerical_train, categorical_train, list_train)
    val_dataset = MultiTypeDataset(targets_val, numerical_val, categorical_val, list_val)
    test_dataset = MultiTypeDataset(targets_test, numerical_test, categorical_test, list_test)
    diff_dataset = MultiTypeDataset(
        torch.zeros(len(qonality_diff)), 
        numerical_diff, 
        categorical_diff, 
        list_diff
    )

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_val = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    diff_loader = DataLoader(diff_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoaders created:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Q_gonality_diff samples: {len(diff_dataset)}")


    # Create FT-Transformer model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model configuration
    model_config = {
        'numerical_features': len([col for col in numerical_columns if col not in categorical_columns + list_columns]),
        'categorical_configs': {col: len(encoders[col].classes_) for col in categorical_columns if col in encoders},
        'list_feature_dim': max_list_len if max_list_len > 0 else 0
    }

    print(f"Model configuration: {model_config}")

    model = FTTransformer(
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.15,
        output_dim=1,
        numerical_features=model_config['numerical_features'],
        categorical_configs=model_config['categorical_configs'],
        list_feature_dim=model_config['list_feature_dim'],
        activation='gelu'
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")


    # ADD THIS CELL RIGHT AFTER MODEL CREATION (after cell eb8cbf2b)

    # 🚀 TRAIN THE FT-TRANSFORMER MODEL (MISSING STEP!)
    print("🚀 Starting FT-Transformer training...")
    print("This was the missing step - the model was never trained!")

# Train the model using your current setup
    train_losses, val_losses, train_accuracy, val_accuracy = train_ft_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=270,  # Start with 50 epochs, increase if needed
        learning_rate=5e-4,
        weight_decay=1e-4,
        device=device,
        patience=50,
        fold_idx='OOT',
        
    )

    print("✅ Training completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")

    # Plot training progress
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FT-Transformer Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Make predictions with FT-Transformer
    print("Making predictions with FT-Transformer...")

    # Ensure model is in evaluation mode

    predictions_train_ft = predict_ft_transformer(model, train_loader_val, device=device)

    predictions_val_ft  = predict_ft_transformer(model, val_loader, device=device)
    predictions_test_ft = predict_ft_transformer(model, test_loader, device=device)
    predictions_diff_ft = predict_ft_transformer(model, diff_loader, device=device)

    print(f"Predictions completed. Shapes:")
    print(f"  Train: {predictions_train_ft.shape}")
    print(f"  Val: {predictions_val_ft.shape}")
    print(f"  Test: {predictions_test_ft.shape}")
    print(f"  Diff: {predictions_diff_ft.shape}")

    # Convert to numpy for evaluation
    train_preds_np = predictions_train_ft.squeeze()
    val_preds_np = predictions_val_ft.squeeze()
    test_preds_np = predictions_test_ft.squeeze()

    # Convert to numpy if they are tensors
    if isinstance(train_preds_np, torch.Tensor):
        train_preds_np = train_preds_np.detach().cpu().numpy()
    if isinstance(val_preds_np, torch.Tensor):
        val_preds_np = val_preds_np.detach().cpu().numpy()
    if isinstance(test_preds_np, torch.Tensor):
        test_preds_np = test_preds_np.detach().cpu().numpy()

    print(f"\nConverted to numpy. Sample predictions:")
    print(f"  Train: {train_preds_np[:5]}")
    print(f"  Val: {val_preds_np[:5]}")
    print(f"  Test: {test_preds_np[:5]}")

    # Evaluate FT-Transformer results
    r2_train_ft, accuracy_train_ft, rmse_train_ft = evaluation.compute_r2_and_accuracy(
        train_preds_np, df_train[target_column].values
    )
    r2_val_ft, accuracy_val_ft, rmse_val_ft = evaluation.compute_r2_and_accuracy(
        val_preds_np, df_val[target_column].values
    )
    r2_test_ft, accuracy_test_ft, rmse_test_ft = evaluation.compute_r2_and_accuracy(
        test_preds_np, df_test[target_column].values
    )

    # Handle q_gonality_bounds
    import ast 
    def ensure_list(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return list(x)

    qonality_diff['q_gonality_bounds'] = qonality_diff['q_gonality_bounds'].apply(ensure_list).tolist()

    # Convert predictions_diff_ft to numpy if it's a tensor
    if isinstance(predictions_diff_ft, torch.Tensor):
        diff_preds_np = predictions_diff_ft.squeeze().detach().cpu().numpy()
    else:
        diff_preds_np = predictions_diff_ft.squeeze()

    accuracy_diff_ft = evaluation.compute_accuracy_in_bounds(
        diff_preds_np, qonality_diff['q_gonality_bounds']
    )

    print(f"\nFT-Transformer Results:")
    print(f"Train - R²: {r2_train_ft:.4f}, Accuracy: {accuracy_train_ft:.4f}, RMSE: {rmse_train_ft:.4f}")
    print(f"Val   - R²: {r2_val_ft:.4f}, Accuracy: {accuracy_val_ft:.4f}, RMSE: {rmse_val_ft:.4f}")
    print(f"Test  - R²: {r2_test_ft:.4f}, Accuracy: {accuracy_test_ft:.4f}, RMSE: {rmse_test_ft:.4f}")
    print(f"Diff  - Accuracy in bounds: {accuracy_diff_ft:.4f}")

    train_evaluation_OOT = {
        'r2': r2_train_ft,
        'accuracy': accuracy_train_ft,
        'rmse': rmse_train_ft
    }
    val_evaluation_OOT = {
        'r2': r2_val_ft,
        'accuracy': accuracy_val_ft,
        'rmse': rmse_val_ft
    }
    test_evaluation_OOT = {
        'r2': r2_test_ft,
        
        'accuracy': accuracy_test_ft,
        'rmse': rmse_test_ft
    }
    diff_evaluation_OOT  = {
        'accuracy': accuracy_diff_ft
    }


    # save train_evaluation, val_evaluation, test_evaluation, diff_evaluation to json
    with open(os.path.join(results_dir, f"train_evaluation_OOT.json"), "w") as f:
        json.dump(train_evaluation_OOT, f)
    with open(os.path.join(results_dir, f"val_evaluation_OOT.json"), "w") as f:
        json.dump(val_evaluation_OOT, f)
    with open(os.path.join(results_dir, f"test_evaluation_OOT.json"), "w") as f:
        json.dump(test_evaluation_OOT, f)
    with open(os.path.join(results_dir, f"diff_evaluation_OOT.json"), "w") as f:
        json.dump(diff_evaluation_OOT, f)


    
    
    
    



   
    
    
    
    




if __name__ == "__main__":
    main()
    
    
    
