import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ast

def safe_eval_list(x):
    """Safely evaluate string representations of lists"""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(x, list):
        return x
    else:
        return []

def prepare_ft_data(X_df, y_series, categorical_columns, list_columns, fitted_scaler=None, fitted_encoders=None, max_list_len=None):
    """Prepare data for FT-Transformer"""
    
    # Get numerical columns (exclude categorical and list columns)
    numerical_cols = [col for col in X_df.columns if col not in categorical_columns + list_columns]
    
    # Handle numerical features
    numerical_data = None
    scaler = fitted_scaler
    if numerical_cols:
        if fitted_scaler is None:
            scaler = StandardScaler()
            numerical_data = torch.tensor(scaler.fit_transform(X_df[numerical_cols]), dtype=torch.float32)
        else:
            numerical_data = torch.tensor(scaler.transform(X_df[numerical_cols]), dtype=torch.float32)
    
    # Handle categorical features
    categorical_data_dict = {}
    encoders = fitted_encoders or {}
    
    for col in categorical_columns:
        if col in X_df.columns:
            if col not in encoders:
                le = LabelEncoder()
                encoded = le.fit_transform(X_df[col].astype(str))
                encoders[col] = le
            else:
                encoded = encoders[col].transform(X_df[col].astype(str))
            categorical_data_dict[col] = torch.tensor(encoded, dtype=torch.long)
    
    # Handle list features
    list_data = None
    actual_max_len = 0
    
    if list_columns:
        all_lists = []
        for _, row in X_df.iterrows():
            row_lists = []
            
            # Process canonical_conjugator
            if 'canonical_conjugator' in list_columns and 'canonical_conjugator' in row:
                canonical = safe_eval_list(row['canonical_conjugator'])
                row_lists.extend(canonical)
            
            # Process conductor (handle nested structure)
            if 'conductor' in list_columns and 'conductor' in row:
                conductor = safe_eval_list(row['conductor'])
                for item in conductor:
                    if isinstance(item, list):
                        row_lists.extend(item)
                    else:
                        row_lists.append(item)
            
            all_lists.append(row_lists)
        
        # Find max length
        if all_lists:
            actual_max_len = max(len(lst) for lst in all_lists) if all_lists else 0
            if max_list_len is not None:
                actual_max_len = max_list_len
            
            # Pad/truncate to same length
            padded_lists = []
            for lst in all_lists:
                if len(lst) > actual_max_len:
                    padded = lst[:actual_max_len]  # Truncate
                else:
                    padded = lst + [0] * (actual_max_len - len(lst))  # Pad
                padded_lists.append(padded)
            
            list_data = torch.tensor(padded_lists, dtype=torch.float32)
    
    # Prepare targets
    targets = torch.tensor(y_series.values, dtype=torch.float32)
    
    return numerical_data, categorical_data_dict, list_data, targets, scaler, encoders, actual_max_len

print("Data preparation functions defined")