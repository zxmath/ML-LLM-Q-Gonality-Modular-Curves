import pandas as pd
import numpy as np
import os
import json
import sys
from sklearn.metrics import mean_squared_error, r2_score
import ast

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils import get_config
from models.Tree_model.model_tree import train_xgb_model
from src.evaluation import compute_r2_and_accuracy, compute_accuracy_in_bounds

# Load configuration
config_path = '/home/zxmath/ML-LLM-Q-Gonality-Modular-Curves/config/modular.toml'
config = get_config.load_toml_config(config_path)

def load_and_preprocess_data(fold_num):
    """Load train, valid, and test data for a specific fold"""
    data_path = os.path.join(project_root, "data/Cross-Validation-Data")
    
    # Load data
    train_data = pd.read_csv(f"{data_path}/train_fold_{fold_num}.csv")
   
    valid_data = pd.read_csv(f"{data_path}/valid_fold_{fold_num}.csv")
  
    test_data = pd.read_csv(f"{data_path}/test_data.csv")  # Using the same test data for all folds
   
    
    # Load q_gonality_diff for bounds evaluation
    q_gonality_diff = pd.read_csv(f"{data_path}/q_gonality_diff.csv")
    q_gonality_diff['q_gonality_bounds'] = q_gonality_diff['q_gonality_bounds'].apply(ast.literal_eval)
    
    return train_data, valid_data, test_data, q_gonality_diff

def prepare_features(data, config):
    """Prepare features based on config settings"""
    # Make a copy so we don't mutate the config
    numerical_features = config['info']['numerical_features']

    return data[numerical_features]
    
    # Categorical features - convert to numeric
    

def train_and_evaluate_fold(fold_num, config):
    """Train XGBoost model and evaluate on train, valid, and test sets"""
    print(f"\n=== Training Fold {fold_num} ===")
    
    # Load data
    train_data, valid_data, test_data, q_gonality_diff = load_and_preprocess_data(fold_num)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Available columns: {list(train_data.columns)}")
    
    # Prepare features and target
    X_train = prepare_features(train_data, config)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_train columns: {list(X_train.columns)}")
    
    y_train = train_data[config.get('target', 'q_gonality')]
    
    X_valid = prepare_features(valid_data, config)
    y_valid = valid_data[config.get('target', 'q_gonality')]
    
    X_test = prepare_features(test_data, config)
    y_test = test_data[config.get('target', 'q_gonality')]

    X_bounds = prepare_features(q_gonality_diff, config)
    
    # Get XGBoost parameters from config
    xgb_params = config.get('params_xgb', {})
    
    # Train model
    print("Training XGBoost model...")
    model = train_xgb_model(X_train, y_train, xgb_params)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    y_pred_bounds = model.predict(X_bounds)
    
    # Evaluate on train set
    print("\n--- Train Set Evaluation ---")
    train_r2, train_accuracy, train_rmse = compute_r2_and_accuracy(y_train_pred, y_train)
    
    # Evaluate on validation set
    print("\n--- Validation Set Evaluation ---")
    valid_r2, valid_accuracy, valid_rmse = compute_r2_and_accuracy(y_valid_pred, y_valid)
    
    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_r2, test_accuracy, test_rmse = compute_r2_and_accuracy(y_test_pred, y_test)
    
    bounds_accuracy = compute_accuracy_in_bounds(y_pred_bounds, q_gonality_diff['q_gonality_bounds'])
    
    # Save results
    results = {
        'fold': fold_num,
        'train': {
            'r2': float(train_r2),
            'accuracy': float(train_accuracy),
            'rmse': float(train_rmse)
        },
        'valid': {
            'r2': float(valid_r2),
            'accuracy': float(valid_accuracy),
            'rmse': float(valid_rmse)
        },
        'test': {
            'r2': float(test_r2),
            'accuracy': float(test_accuracy),
            'rmse': float(test_rmse)
        },
        'bounds_accuracy': float(bounds_accuracy)
    }
    
    return results, model

def main():
    """Main training function"""
    # Create results directory
    results_dir = os.path.join(project_root, "results/xgb")
    os.makedirs(results_dir, exist_ok=True)
    
    # Train on all folds
    all_results = []
    models = {}
    
    for fold_num in range(5):  # Assuming 5 folds (fold_0 to fold_4)
        try:
            results, model = train_and_evaluate_fold(fold_num, config)
            all_results.append(results)
            models[f'fold_{fold_num}'] = model
            
            # Save individual fold results
            with open(f"{results_dir}/fold_{fold_num}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error training fold {fold_num}: {e}")
            continue
    
    # Calculate average results
    if all_results:
        avg_results = {
            'train': {
                'r2': np.mean([r['train']['r2'] for r in all_results]),
                'accuracy': np.mean([r['train']['accuracy'] for r in all_results]),
                'rmse': np.mean([r['train']['rmse'] for r in all_results])
            },
            'valid': {
                'r2': np.mean([r['valid']['r2'] for r in all_results]),
                'accuracy': np.mean([r['valid']['accuracy'] for r in all_results]),
                'rmse': np.mean([r['valid']['rmse'] for r in all_results])
            },
            'test': {
                'r2': np.mean([r['test']['r2'] for r in all_results]),
                'accuracy': np.mean([r['test']['accuracy'] for r in all_results]),
                'rmse': np.mean([r['test']['rmse'] for r in all_results])
            },
            'bounds_accuracy': np.mean([r['bounds_accuracy'] for r in all_results])
        }
        
        # Save average results
        with open(f"{results_dir}/average_results.json", 'w') as f:
            json.dump(avg_results, f, indent=2)
        
        # Print summary
        print("\n=== SUMMARY ===")
        print(f"Average Train R²: {avg_results['train']['r2']:.4f}")
        print(f"Average Train Accuracy: {avg_results['train']['accuracy']*100:.2f}%")
        print(f"Average Train RMSE: {avg_results['train']['rmse']:.4f}")
        print(f"Average Valid R²: {avg_results['valid']['r2']:.4f}")
        print(f"Average Valid Accuracy: {avg_results['valid']['accuracy']*100:.2f}%")
        print(f"Average Valid RMSE: {avg_results['valid']['rmse']:.4f}")
        print(f"Average Test R²: {avg_results['test']['r2']:.4f}")
        print(f"Average Test Accuracy: {avg_results['test']['accuracy']*100:.2f}%")
        print(f"Average Test RMSE: {avg_results['test']['rmse']:.4f}")
        print(f"Average Bounds Accuracy: {avg_results['bounds_accuracy']*100:.2f}%")
        
        # Save all results
        with open(f"{results_dir}/all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    main()






