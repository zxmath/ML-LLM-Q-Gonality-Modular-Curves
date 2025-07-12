import pandas as pd
import os
import toml
from sklearn.model_selection import KFold, train_test_split
from utils.data_process import data_process


def create_cross_validation_data(data_path, config_path, output_dir, n_splits=5, test_size=0.1, random_state=42):
    """
    Create cross-validation data splits and save them to files.
    
    Args:
        data_path: Path to the combined_data_7.h5 file
        config_path: Path to the config file with features
        output_dir: Directory to save the cross-validation data
        n_splits: Number of folds for KFold (default 5 for 5-fold cross-validation)
        test_size: Size of test set (default 0.1)
        random_state: Random state for reproducibility
    """
    
    # Read config file
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Get features for dropping duplicates
    features = config['info']['features']
    
    all_features = features 
    
    print(f"Features for duplicate removal: {all_features}")
    
    # Load and process data
    print("Loading and processing data...")
    q_gonality_same, q_gonality_diff = data_process(data_path)
    
    # Drop duplicates based on features
    print("Dropping duplicates...")
    print(f"q_gonality_same shape before dropping duplicates: {q_gonality_same.shape}")
    q_gonality_same_dedup = q_gonality_same.drop_duplicates(subset=all_features)
    print(f"q_gonality_same shape after dropping duplicates: {q_gonality_same_dedup.shape}")
    
    print(f"q_gonality_diff shape before dropping duplicates: {q_gonality_diff.shape}")
    q_gonality_diff_dedup = q_gonality_diff.drop_duplicates(subset=all_features)
    print(f"q_gonality_diff shape after dropping duplicates: {q_gonality_diff_dedup.shape}")
    
    # Split q_gonality_same into test and train+valid (0.1 test, 0.9 train+valid)
    print("Creating test split...")
    train_valid_indices, test_indices = train_test_split(
        q_gonality_same_dedup.index, 
        test_size=test_size, 
        random_state=random_state,
        #stratify=q_gonality_same_dedup['q_gonality'] if 'q_gonality' in q_gonality_same_dedup.columns else None
    )
    
    train_valid_data = q_gonality_same_dedup.loc[train_valid_indices]
    test_data = q_gonality_same_dedup.loc[test_indices]
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Train+Valid data shape: {train_valid_data.shape}")
    
    # Save test data (this doesn't change across folds)
    os.makedirs(output_dir, exist_ok=True)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    q_gonality_diff_dedup.to_csv(os.path.join(output_dir, 'q_gonality_diff.csv'), index=False)
    
    print("Saved test_data.csv and q_gonality_diff.csv")
    
    # Create KFold splits for train+valid data with different random states
    print(f"Creating {n_splits}-fold cross-validation splits...")
    
    for fold_idx in range(n_splits):
        print(f"Processing fold {fold_idx + 1}/{n_splits}")
        
        # Use different random state for each fold
        fold_random_state = random_state + fold_idx
        
        # Create train/valid split for this fold
        train_indices, valid_indices = train_test_split(
            train_valid_data.index,
            test_size=1/9,  # 1/9 ≈ 0.111 for validation, 8/9 ≈ 0.889 for training (8:1 ratio)
            random_state=fold_random_state,
            #stratify=train_valid_data['q_gonality'] if 'q_gonality' in train_valid_data.columns else None
        )
        
        train_fold = train_valid_data.loc[train_indices]
        valid_fold = train_valid_data.loc[valid_indices]
        
        print(f"  Train fold shape: {train_fold.shape}")
        print(f"  Valid fold shape: {valid_fold.shape}")
        print(f"  Random state for fold {fold_idx}: {fold_random_state}")
        
        # Save fold data
        train_file = os.path.join(output_dir, f'train_fold_{fold_idx}.csv')
        valid_file = os.path.join(output_dir, f'valid_fold_{fold_idx}.csv')
        
        train_fold.to_csv(train_file, index=False)
        valid_fold.to_csv(valid_file, index=False)
        
        print(f"  Saved train_fold_{fold_idx}.csv and valid_fold_{fold_idx}.csv")
    
    print("Cross-validation data creation completed!")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Original q_gonality_same shape: {q_gonality_same.shape}")
    print(f"After deduplication: {q_gonality_same_dedup.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train+Valid data shape: {train_valid_data.shape}")
    print(f"q_gonality_diff shape: {q_gonality_diff_dedup.shape}")
    print(f"Number of CV folds: {n_splits}")
    print(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_path = os.path.join(project_root, 'data', 'combined_data_7.h5')
    config_path = os.path.join(project_root, 'config', 'modular.toml')
    output_dir = os.path.join(project_root, 'data', 'Cross-Validation-Data')
    
    # Create cross-validation data
    create_cross_validation_data(
        data_path=data_path,
        config_path=config_path,
        output_dir=output_dir,
        n_splits=5,  # 5-fold cross-validation
        test_size=0.1,
        random_state=42
    ) 

