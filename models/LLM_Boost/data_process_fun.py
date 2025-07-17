import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
from typing import Tuple



def compute_metrics(y_pred: np.ndarray, y_test: np.ndarray, logger: logging.Logger) -> Tuple[float, float, float]:
    try:
        if hasattr(y_pred, 'dtype') and y_pred.dtype == 'object':
            y_pred = pd.to_numeric(y_pred, errors='coerce').fillna(0) # type: ignore
        y_pred = np.asarray(y_pred, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        y_pred_rounded = np.floor(y_pred + 0.5)
        r2 = r2_score(y_test, y_pred)
        accuracy = np.mean(y_pred_rounded == y_test) * 100
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, accuracy, rmse
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return 0.0, 0.0, float('inf')
def compute_accuracy_in_bounds(y_pred, y_bounds):
    
    y_pred = np.floor(y_pred+0.5)
    # Check if the predicted values are within the bounds
    in_bounds = (y_pred >= y_bounds.map(lambda b: b[0])) & (y_pred <= y_bounds.map(lambda b: b[1]))
    # Calculate accuracy
    accuracy = np.mean(in_bounds) * 100
    # Print the results
    print(f"Accuracy within bounds: {accuracy:.2f}%")
    return accuracy

def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values and data type conversions
    df_clean = df_raw.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            except:
                pass
    df_clean = df_clean.fillna(0)
    return df_clean