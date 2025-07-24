# Machine Learning and LLM-Boost Symbolic Regression for Predicting Q-Gonality of Modular Curves

This repository accompanies the paper:  
[Machine Learning and LLM-Boost Symbolic Regression for Predicting $Q$-Gonality of Modular Curves](https://openreview.net/pdf?id=LDcFa3E5vJ).

It provides a comprehensive framework for predicting the $Q$-gonality of modular curves using a variety of machine learning models, including neural networks, tree-based models, FT-Transformers, and LLM-boosted symbolic regression.

## Table of Contents
- Project Structure
- Installation
- Configuration
- Model Overview
- Usage
- Evaluation
- Results


---

## Project Structure

```
.
├── config/                # Configuration files (TOML)
├── models/                # Model implementations
│   ├── FT_Transformer/    # FT-Transformer model
│   ├── LLM_Boost/         # LLM-boosted symbolic regression
│   ├── NN_model/          # Neural network models
│   └── Tree_model/        # Tree-based models (e.g., XGBoost)
├── results/               # Output results and plots
├── saved_models/          # Saved model checkpoints
├── src/                   # Source code and utilities
│   └── utils/             # Utility scripts
├── pyproject.toml         # Python dependencies (Poetry)
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ML-LLM-Q-Gonality-Modular-Curves
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```
   Or, using pip:
   ```bash
   pip install -r requirements.txt
   ```
   *(If requirements.txt is not present, use the dependencies listed in `pyproject.toml`.)*

3. **Python version:**  
   Requires Python 3.12 or higher.

## Configuration

All model and experiment settings are managed via `config/modular.toml`.  
This includes feature selection, model hyperparameters, and training options for each model type.

## Model Overview

### 1. Neural Network (NN)
- Located in `models/NN_model/`
- Main script: `NN_Classical_Train_and_Main.py`
- Implements feedforward neural networks for regression.
- Configurable via `[FFN]` section in `modular.toml`.

### 2. Tree-based Models
- Located in `models/Tree_model/`
- Main script: `train_xgb.py`
- Uses XGBoost for regression.
- Hyperparameters set in `[params_xgb]` in `modular.toml`.

### 3. FT-Transformer
- Located in `models/FT_Transformer/`
- Main script: `FT_experiment.py`
- Implements a Feature Tokenizer Transformer for tabular data.
- Configurable via `[FT_Transformer]` in `modular.toml`.

### 4. LLM-Boosted Symbolic Regression
- Located in `models/LLM_Boost/`
- Main script: `main.py`
- Combines linear boosting with LLM-based feature engineering and symbolic regression.
- Configurable via `[LLM]` in `modular.toml`.

## Usage

### Training and Evaluation

Each model can be trained and evaluated independently.  
Example commands (from project root):

- **Neural Network:**
  ```bash
  python models/NN_model/NN_Classical_Train_and_Main.py
  ```

- **Tree-based Model:**
  ```bash
  python models/Tree_model/train_xgb.py
  ```

- **FT-Transformer:**
  ```bash
  python models/FT_Transformer/FT_experiment.py
  ```

- **LLM-Boosted Regression:**
  ```bash
  python models/LLM_Boost/main.py
  ```

### Cross-Validation

- Cross-validation splits can be created using:
  ```bash
  python src/create_CV.py
  ```

## Evaluation

- Standard regression metrics are used: R², RMSE, and accuracy.
- Evaluation utilities are in `src/evaluation.py`.

## Results

- Results, plots, and model checkpoints are saved in the `results/` and `saved_models/` directories.
- Example plots: neural network performance, feature importance, residuals, etc.



**Note:**  
This README omits details about the data and its preparation, as requested. For more information about the dataset, please refer to the original paper or contact the authors. 