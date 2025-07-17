import os, sys
import logging
from datetime import datetime
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from string_manipulation import make_func
from data_process_fun import compute_metrics
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import sys
from linear_boost_model import FastLinearRegression



def setup_logging(local_path: str, model_type: str) -> logging.Logger:
    logger_dir = os.path.join(local_path, 'logger')
    os.makedirs(logger_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"Boosting_{model_type}_{timestamp}.log"
    log_filepath = os.path.join(logger_dir, log_filename)
    logger = logging.getLogger(f'Boosting_{model_type}')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler (always works)
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with broken pipe protection
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except BrokenPipeError:
                # Ignore broken pipe errors when output is redirected/piped
                pass
            except Exception:
                # Ignore other stream errors to prevent script interruption
                pass
    
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class LoggerWrapper:
    def __init__(self, logger: logging.Logger, original_stdout):
        self.logger = logger
        self.original_stdout = original_stdout
    def write(self, message):
        try:
            self.original_stdout.write(message)
            self.original_stdout.flush()
        except BrokenPipeError:
            # Ignore broken pipe errors when output is redirected/piped
            pass
        message = message.rstrip('\n\r')
        if message:
            try:
                self.logger.info(message)
            except BrokenPipeError:
                # Ignore broken pipe errors in logging
                pass
    def flush(self):
        try:
            self.original_stdout.flush()
        except BrokenPipeError:
            # Ignore broken pipe errors when output is redirected/piped
            pass


#################################

def save_results(model_boost, accuracy_boost, 
                produced_features_dict, save_dir="Boosting_results", logger=None,
                df_train=None, df_val=None, df_test=None, 
                target_train=None, target_val=None, target_test=None, local_path=None):
    """Save all results"""
    # Use local_path if provided, following the same convention as model_llm_boosting.py
    if local_path is not None:
        save_dir = os.path.join(local_path, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save models (handle pickle issues with lambda functions)
    try:
        # Create serializable version of boosting model without lambda functions
        boosting_model_data = {
            'base_features': model_boost.base_features,
            'poly_degree': model_boost.poly_degree,
            'selected_features': model_boost.selected_features,
            'feature_bodies': model_boost.feature_bodies,
            'poly_feature_names': model_boost.poly_feature_names,
            'base_model_coef': model_boost.base_model.coef_ if model_boost.base_model else None,
            'base_model_intercept': model_boost.base_model.intercept_ if model_boost.base_model else None,
            'residual_models': [(fname, model.coef_, model.intercept_) for fname, model in model_boost.residual_models]
        }
        joblib.dump(boosting_model_data, os.path.join(save_dir, 'boosting_model.joblib'))
    except Exception as e:
        if logger:
            logger.warning(f"Could not save boosting model: {e}")


    
    # Save accuracy progress
    max_iterations = len(accuracy_boost)
    accuracy_df = pd.DataFrame({
        'iteration': range(max_iterations),
        'boosting_accuracy': accuracy_boost + [None] * (max_iterations - len(accuracy_boost))
    })
    accuracy_df.to_csv(os.path.join(save_dir, 'accuracy.csv'), index=False)
    
    # Evaluate on train, validation, and test sets
    if all(x is not None for x in [df_train, df_val, df_test, target_train, target_val, target_test]):
        evaluation_results = []
        
        for dataset_name, df, target in [('train', df_train, target_train), 
                                        ('val', df_val, target_val), 
                                        ('test', df_test, target_test)]:
            try:
                # Boosting model evaluation
                pred_boosting = model_boost.predict(df)
                r2_boosting, acc_boosting, mse_boosting = compute_metrics(pred_boosting, target, logger) # type: ignore
            except Exception as e:
                if logger:
                    logger.error(f"Error predicting with boosting model on {dataset_name}: {e}")
                r2_boosting, acc_boosting, mse_boosting = 0.0, 0.0, float('inf')
            
            
            evaluation_results.append({
                'dataset': dataset_name,
                'model': 'boosting',
                'r2': r2_boosting,
                'accuracy': acc_boosting,
                'mse': mse_boosting
            })
            
            
            if logger:
                logger.info(f"{dataset_name.upper()} SET RESULTS:")
                logger.info(f"  Boosting - R2: {r2_boosting:.4f}, Accuracy: {acc_boosting:.2f}%, MSE: {mse_boosting:.4f}")


        # Save evaluation results
        eval_df = pd.DataFrame(evaluation_results)
        eval_df.to_csv(os.path.join(save_dir, 'model_evaluation_metrics.csv'), index=False)
    
    # Save selected features
    max_features = len(model_boost.selected_features)
    features_padded = model_boost.selected_features + [None] * (max_features - len(model_boost.selected_features))

    features_df = pd.DataFrame({
        'boosting_features': features_padded,

    })
    features_df.to_csv(os.path.join(save_dir, 'selected_features.csv'), index=False)
    
    # Save coefficients
    boosting_coefs = model_boost.get_coefficients()

    with open(os.path.join(save_dir, 'boosting_coefficients.txt'), 'w') as f:
        for name, coef in boosting_coefs.items():
            f.write(f"{name}\t{coef:.18e}\n")
    
    # Save generated features
    features_data = []
    for code_str, info in produced_features_dict.items():
        features_data.append({
            'code_str': code_str,
            'reason': info['reason'],
            'body': info['body']
        })
    if features_data:
        features_df = pd.DataFrame(features_data)
        features_df.to_csv(os.path.join(save_dir, 'generated_features.csv'), index=False)
    
    if logger:
        logger.info(f"Results saved to {save_dir}/")

def plot_graph(accuracy_boosting, save_path="accuracy_graph.png", local_path=None):
    """Plot accuracy comparison"""
    plt.figure(figsize=(12, 6))
    
    # Handle different lengths by using the maximum length
    max_length = len(accuracy_boosting)
    iterations = list(range(1, max_length + 1))
    
    # Pad shorter list with the last value to match lengths
    boosting_padded = accuracy_boosting + [accuracy_boosting[-1]] * (max_length - len(accuracy_boosting))

    plt.plot(iterations, boosting_padded, 'b-', linewidth=2, marker='o', 
             markersize=6, label='Boosting')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Comparison: Boosting Regression', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate final values using original (non-padded) values
    if accuracy_boosting:
        plt.annotate(f'Boosting: {accuracy_boosting[-1]:.2f}%', 
                    xy=(len(accuracy_boosting), accuracy_boosting[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))
    
    plt.tight_layout()
    # Use local_path if provided, following the same convention as save_results
    if local_path is not None:
        save_path = os.path.join(local_path, "accuracy_results", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_checkpoint(state, checkpoint_path):
    """Save training state to checkpoint file"""
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # Convert non-serializable objects to serializable format
        serializable_state = {
            'iteration': state['iteration'],
            'accuracy_boosting': state['accuracy_boosting'],
            'produced_features_dict': {
                k: {
                    'reason': v['reason'],
                    'body': v['body'],
                    'code_str': v['code_str']
                } for k, v in state['produced_features_dict'].items()
            },
            'queue_features_boosting': state.get('queue_features_boosting', []),
            # Maintain backward compatibility with old checkpoints
            'queue_features': state.get('queue_features_boosting', []),  # Fallback for old checkpoint format
            'model_boosting_state': {
                'base_features': state['model_boosting'].base_features,
                'poly_degree': state['model_boosting'].poly_degree,
                'selected_features': state['model_boosting'].selected_features,
                'feature_bodies': state['model_boosting'].feature_bodies,
                'poly_feature_names': state['model_boosting'].poly_feature_names,
                'base_model_coef': state['model_boosting'].base_model.coef_.tolist() if state['model_boosting'].base_model and hasattr(state['model_boosting'].base_model, 'coef_') else None,
                'base_model_intercept': float(state['model_boosting'].base_model.intercept_) if state['model_boosting'].base_model and hasattr(state['model_boosting'].base_model, 'intercept_') else None,
                'residual_models': [(fname, model.coef_.tolist(), float(model.intercept_)) for fname, model in state['model_boosting'].residual_models],
                'feature_predictors': [(fname, predictor.coef_.tolist() if predictor else None, float(predictor.intercept_) if predictor else None) for fname, predictor in state['model_boosting'].feature_predictors] if hasattr(state['model_boosting'], 'feature_predictors') else [],
                'current_accuracy': state['model_boosting'].current_accuracy
            },
            
            # Add best model states for global best tracking
            'best_boosting_accuracy': state.get('best_boosting_accuracy', max(state['accuracy_boosting']) if state['accuracy_boosting'] else 0.0),
            'best_boosting_model_state': state.get('best_boosting_model_state', None),
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(serializable_state, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
        return False

def load_checkpoint(checkpoint_path, model_boost):
    """Load training state from checkpoint file"""
    try:
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
        
        # Restore produced_features_dict with function objects
        produced_features_dict = {}
        for k, v in state['produced_features_dict'].items():
            produced_features_dict[k] = {
                'reason': v['reason'],
                'body': v['body'],
                'code_str': v['code_str'],
                'func': make_func(v['body'])
            }
        
        # Restore boosting model state
        boosting_state = state['model_boosting_state']
        model_boost.base_features = boosting_state['base_features']
        model_boost.poly_degree = boosting_state['poly_degree']
        model_boost.selected_features = boosting_state['selected_features']
        model_boost.feature_bodies = boosting_state['feature_bodies']
        model_boost.poly_feature_names = boosting_state['poly_feature_names']
        model_boost.feature_functions = {name: make_func(body) for name, body in boosting_state['feature_bodies'].items()}
        
        # CRITICAL FIX: Restore PolynomialFeatures for linear model
        if model_boost.poly_feature_names:
            model_boost.poly = PolynomialFeatures(degree=model_boost.poly_degree, include_bias=False)
            # Fit on dummy data to restore the transformer state
            dummy_data = pd.DataFrame({feature: [0] for feature in model_boost.base_features})
            model_boost.poly.fit(dummy_data)
            print(f"Restored PolynomialFeatures for boosting model with degree {model_boost.poly_degree}")

        # Restore linear model
        if boosting_state.get('base_model_coef') is not None:
            model_boost.base_model = FastLinearRegression()
            model_boost.base_model.coef_ = np.array(boosting_state['base_model_coef'])
            model_boost.base_model.intercept_ = boosting_state['base_model_intercept']
            model_boost.base_model.fitted_ = True

        # Restore linear residual models
        model_boost.residual_models = []
        for fname, coef, intercept in boosting_state.get('residual_models', []):
            residual_model = FastLinearRegression()
            residual_model.coef_ = np.array(coef)
            residual_model.intercept_ = intercept
            residual_model.fitted_ = True
            model_boost.residual_models.append((fname, residual_model))
        
        # Restore linear feature predictors
        model_boost.feature_predictors = []
        for fname, coef, intercept in boosting_state.get('feature_predictors', []):
            if coef is not None:
                feature_predictor = FastLinearRegression()
                feature_predictor.coef_ = np.array(coef)
                feature_predictor.intercept_ = intercept
                feature_predictor.fitted_ = True
                model_boost.feature_predictors.append((fname, feature_predictor))
            else:
                model_boost.feature_predictors.append((fname, None))

        # Set current accuracy for boosting model
        boosting_current_accuracy = boosting_state.get('current_accuracy', 0.0)
        if boosting_current_accuracy == 0.0 and len(state['accuracy_boosting']) > 0:
            boosting_current_accuracy = state['accuracy_boosting'][-1]
        model_boost.set_current_accuracy(boosting_current_accuracy)


        # Check if the checkpoint contains best model states and restore to best if better than current
        best_boosting_accuracy = state.get('best_boosting_accuracy', 0.0)
        best_boosting_model_state = state.get('best_boosting_model_state', None)

        # If we have best model states and they're better than current, restore to best
        if (best_boosting_model_state is not None and 
            best_boosting_accuracy > model_boost.current_accuracy):
            print(f"Restoring boosting model to best state (accuracy: {best_boosting_accuracy:.2f}%)")
            restore_model_from_state(model_boost, best_boosting_model_state)
            model_boost.set_current_accuracy(best_boosting_accuracy)

        return {
            'iteration': state['iteration'],
            'accuracy_boosting': state['accuracy_boosting'],
            'produced_features_dict': produced_features_dict,
            'queue_features_boosting': state.get('queue_features_boosting', state.get('queue_features', [])),
            'model_boosting': model_boost,
            'best_boosting_accuracy': best_boosting_accuracy,
            'best_boosting_model_state': best_boosting_model_state
        }
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

def safe_log_message(logger, level, message):
    """Safely log a message, handling None logger"""
    if logger is not None:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)

def safe_log(logger, level, message):
    """Safely log a message, handling broken pipe errors"""
    try:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
    except BrokenPipeError:
        # Ignore broken pipe errors - this happens when output is redirected
        pass
    except Exception:
        # Ignore other logging errors to prevent script interruption
        pass

def capture_model_state(model):
    """Capture the current state of a model for saving as best state"""
    try:
        return {
            'base_features': model.base_features,
            'poly_degree': model.poly_degree,
            'selected_features': model.selected_features,
            'feature_bodies': model.feature_bodies,
            'poly_feature_names': model.poly_feature_names,
            'base_model_coef': model.base_model.coef_.tolist() if model.base_model and hasattr(model.base_model, 'coef_') else None,
            'base_model_intercept': float(model.base_model.intercept_) if model.base_model and hasattr(model.base_model, 'intercept_') else None,
            'residual_models': [(fname, model.coef_.tolist(), float(model.intercept_)) for fname, model in model.residual_models],
            'feature_predictors': [(fname, predictor.coef_.tolist() if predictor else None, float(predictor.intercept_) if predictor else None) for fname, predictor in model.feature_predictors] if hasattr(model, 'feature_predictors') else [],
            'current_accuracy': model.current_accuracy
            }
        

    except Exception as e:
        print(f"Failed to capture model state: {e}")
        return None

def restore_model_from_state(model, state):
    """Restore a model from a saved state"""
    try:
        # Common fields
        model.base_features = state['base_features']
        model.poly_degree = state['poly_degree']
        model.selected_features = state['selected_features']
        model.feature_bodies = state['feature_bodies']
        model.poly_feature_names = state['poly_feature_names']
        model.feature_functions = {name: make_func(body) for name, body in state['feature_bodies'].items()}
        model.current_accuracy = state.get('current_accuracy', 0.0)
        
        # CRITICAL FIX: Restore PolynomialFeatures for both model types
        if model.poly_feature_names:
            model.poly = PolynomialFeatures(degree=model.poly_degree, include_bias=False)
            # Fit on dummy data to restore the transformer state
            dummy_data = pd.DataFrame({feature: [0] for feature in model.base_features})
            model.poly.fit(dummy_data)
            print(f"Restored PolynomialFeatures for model with degree {model.poly_degree}")
        
        # Restore linear base model
        if state.get('base_model_coef') is not None:
            model.base_model = FastLinearRegression()
            model.base_model.coef_ = np.array(state['base_model_coef'])
            model.base_model.intercept_ = state['base_model_intercept']
            model.base_model.fitted_ = True
            
        # Restore linear residual models
        model.residual_models = []
        for fname, coef, intercept in state.get('residual_models', []):
            residual_model = FastLinearRegression()
            residual_model.coef_ = np.array(coef)
            residual_model.intercept_ = intercept
            residual_model.fitted_ = True
            model.residual_models.append((fname, residual_model))
                
        # Restore linear feature predictors
        model.feature_predictors = []
        for fname, coef, intercept in state.get('feature_predictors', []):
            if coef is not None:
                feature_predictor = FastLinearRegression()
                feature_predictor.coef_ = np.array(coef)
                feature_predictor.intercept_ = intercept
                feature_predictor.fitted_ = True
                model.feature_predictors.append((fname, feature_predictor))
            else:
                model.feature_predictors.append((fname, None))
        
        return True
    except Exception as e:
        print(f"Failed to restore model from state: {e}")
        return False