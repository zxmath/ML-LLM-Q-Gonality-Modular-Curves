import pandas as pd
import numpy as np # For np.nan
import os

def create_data_by_feature_target(df, feature, target):
   
    # Example implementation (replace with your actual logic):
    selected_columns = []
    if isinstance(feature, list):
        selected_columns.extend(feature)
    elif isinstance(feature, str): # Check if it's a string and not None
        selected_columns.append(feature)


    if isinstance(target, list):
        selected_columns.extend(target)
    elif isinstance(target, str): # Check if it's a string and not None
        selected_columns.append(target)

    # Ensure all selected columns exist in the DataFrame
    # If not, it could lead to issues or NaNs later
    # Remove None from selected_columns if they were added due to feature/target being None
    selected_columns = [col for col in selected_columns if col is not None]
    
    columns_to_use = [col for col in selected_columns if col in df.columns]
    
    if len(columns_to_use) != len(selected_columns):
        missing_cols = set(selected_columns) - set(columns_to_use)
        print(f"Warning: Not all feature/target columns found in DataFrame. Missing: {missing_cols}. Found: {columns_to_use}, Requested: {selected_columns}")


    if not columns_to_use:
        print("Error: No valid feature or target columns found in the DataFrame after filtering.")
        return pd.DataFrame() # Return empty DataFrame to avoid errors

    # It's good practice to make a copy if you're subsetting
    data_subset = df[columns_to_use].copy()
    
    return data_subset


def combine_data_by_feature_target(feature, target, data_path, range_start, range_end):
    data_combine = pd.DataFrame()
    
    # Define common string values that should be treated as NaN
    common_na_strings = ['None', 'null', 'N/A', 'NaN', ''] 

    for i in range(range_start, range_end):
        file_path = os.path.join(data_path, f'NamedlmfdbDataRanks{i}.csv')
        if os.path.exists(file_path):
            print(f"\n--- Processing file: {file_path} ---")
            try:
                # Read CSV, treat common_na_strings as NaN, handle spaces
                df = pd.read_csv(
                    file_path,
                    sep=';',
                    skipinitialspace=True, # Handles spaces after delimiter
                    low_memory=False,
                    na_values=common_na_strings # Converts exact matches to np.nan
                )
                print(f"Shape after pd.read_csv (file {i}): {df.shape}")
                # print(f"NaNs immediately after read_csv (file {i}):\n{df.isnull().sum()}") 

                # Clean column names
                df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
                
                # Strip whitespace from all string cell values
                for c in df.select_dtypes(include="object").columns:
                    df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
                
                # --- ADDED STEP: Explicitly replace 'None' strings with np.nan ---
                # This catches 'None' strings that might have been formed after stripping,
                # if they weren't caught by na_values (e.g., if they were ' None ' in the CSV)
                print(f"Checking for 'None' strings to convert to np.nan after stripping (file {i})...")
                for c in df.select_dtypes(include="object").columns:
                    # Count 'None' strings before replacement for debugging
                    # none_string_count_before = df[c].eq('None').sum()
                    # if none_string_count_before > 0:
                    #     print(f"   Column '{c}': Found {none_string_count_before} 'None' strings before replacement.")
                    df[c] = df[c].replace('None', np.nan) # Replace string 'None' with actual NaN
                    # none_string_count_after = df[c].eq('None').sum()
                    # if none_string_count_before > 0 : # Only print if there was something to replace
                    #    print(f"   Column '{c}': Found {none_string_count_after} 'None' strings after replacement.")

                print(f"NaNs after explicit 'None' string replacement (file {i}):\n{df.isnull().sum()}")


                data_by_feature_target = create_data_by_feature_target(df, feature, target)
                
                if data_by_feature_target.empty and not df.empty:
                    print(f"Warning: create_data_by_feature_target returned an empty DataFrame for non-empty input from file {i}.")

                # Column alignment before concat
                if not data_combine.empty and not data_by_feature_target.empty:
                    if set(data_combine.columns) != set(data_by_feature_target.columns):
                        print(f"Warning: Column mismatch before concat for file {i}!")
                        print(f"  data_combine columns: {data_combine.columns.tolist()}")
                        print(f"  data_by_feature_target columns: {data_by_feature_target.columns.tolist()}")
                        all_cols = data_combine.columns.union(data_by_feature_target.columns)
                        data_combine = data_combine.reindex(columns=all_cols)
                        data_by_feature_target = data_by_feature_target.reindex(columns=all_cols)


                if not data_by_feature_target.empty:
                    data_combine = pd.concat([data_combine, data_by_feature_target], axis=0, ignore_index=True) 
                else:
                    print(f"Skipping concatenation for file {i} as data_by_feature_target is empty.")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print("\n--- Final Combined Data (Before Final NaN Drop) ---")
    print(f"Shape: {data_combine.shape}")
    if not data_combine.empty:
        print(f"Total NaNs per column:\n{data_combine.isnull().sum()}")
        if data_combine.isnull().any().any():
            print("Sample of rows with NaNs (before final drop):")
            print(data_combine[data_combine.isnull().any(axis=1)].head())
    else:
        print("Combined data is empty before final NaN drop.")

    # --- Option to drop NaNs on the FINAL combined data before returning ---
    columns_for_nan_check = []
    if isinstance(feature, list):
        columns_for_nan_check.extend(feature)
    elif isinstance(feature, str): 
        columns_for_nan_check.append(feature)

    if isinstance(target, list):
        columns_for_nan_check.extend(target)
    elif isinstance(target, str):
        columns_for_nan_check.append(target)
    
    columns_for_nan_check = [col for col in columns_for_nan_check if col is not None]

    if columns_for_nan_check: 
        columns_for_nan_check = list(dict.fromkeys(columns_for_nan_check)) 
        columns_for_nan_check = [col for col in columns_for_nan_check if col in data_combine.columns]
        
        if columns_for_nan_check: 
            print(f"\nApplying: Drop rows where NaN exists in any of these specific columns: {columns_for_nan_check}...")
            data_combine.dropna(subset=columns_for_nan_check, how='any', inplace=True)
        else:
            print("\nWarning: Specified feature/target columns for NaN check were not found in the combined DataFrame's columns. No subset-based NaN drop performed.")
    else:
        print("\nNo specific feature/target columns were defined for the NaN drop. No subset-based NaN drop performed.")


    print(f"\n--- Final Combined Data (After Potential NaN Drop) ---")
    print(f"Shape to be returned: {data_combine.shape}")
    if not data_combine.empty:
        print(f"NaNs per column to be returned:\n{data_combine.isnull().sum()}")
        # Final check for any remaining 'None' strings in the log_conductor column if it exists
        if 'log_conductor' in data_combine.columns:
            if data_combine['log_conductor'].eq('None').any():
                print("CRITICAL WARNING: String 'None' still exists in 'log_conductor' in the final DataFrame!")
            else:
                print("'log_conductor' column does not contain 'None' strings in the final DataFrame.")

        if data_combine.isnull().any().any():
            print("WARNING: NaNs still present after final drop. Review dropna conditions or data integrity.")
        else:
            print("Final combined data appears to be clean of specified NaNs.")
    else:
        print("Combined data is empty after potential NaN drop.")

    return data_combine



