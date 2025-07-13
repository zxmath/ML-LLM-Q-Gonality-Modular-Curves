import itertools

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# import torch.nn.functional as F
import torch.optim as optim
from jedi.api import file_name
from sympy.strategies.core import switch
# from jupyter_server.auth import passwd
# from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import namedtuple

try:
    from sklearn.model_selection import train_test_split  # for data splitting
except ImportError:
    raise ImportError("Please install scikit-learn package using: pip install scikit-learn")
import os
import re
from pathlib import Path
import sys
# import argparse # for accept args from commands
import ast  # for safely evaluating string expressions
import random
import matplotlib.pyplot as plt  # plot the loss graphs
import tables
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from src.utils.get_config import load_toml_config


#### Data preprocessing ####
#### The following is for splitting data with bounds [a,a] ####
def Classical_Exact_Data_Filter(files = {"File_Exact_Train": "",
                                         "File_Exact_Valid": "",
                                         "File_Exact_Test": "",
                                        "File_Bounded": "",
                                        "File_Bounded_Train": "",
                                        "File_Bounded_Valid": "",
                                        "File_Bounded_Test": ""},
                      X_col_names=["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor","coarse_class_num", "coarse_level"],
                      Y_col_names=['q_gonality_bounds'],
                    folder_name = "data"):
    """
        Loads multiple CSVs with Y columns containing interval strings "[a, b]".
        Keeps only rows where all Y columns are exactly "[a, a]",
        and extracts scalar `a` from them.

        :param
        files: dictionary of file paths
        X_col_names: list of column names for X
        Y_col_names: list of column names for Y

        Returns:
            X_train, Y_train, X_val, Y_val, X_test, Y_test (pandas DataFrames)
    """

    """
        read the data from files
    """
    usecols = X_col_names + Y_col_names
    f_train = files["File_Exact_Train"] # this should be a os.path
    f_val = files["File_Exact_Valid"] # this should be another os.path
    f_test = files["File_Exact_Test"] # same as above


    CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "modular.toml"
    root_path = Path(__file__).resolve().parent.parent.parent
    config = load_toml_config(CONFIG_PATH)
    data_path = os.path.join(root_path, config["FFN"]["nn_data_path"])
    base_path = data_path
    f_train_path = os.path.join(base_path, f_train)
    df_train = pd.read_csv(f_train_path) # data frame for exact training
    f_val_path = os.path.join(base_path, f_val)
    df_val = pd.read_csv(f_val_path) # data frame for valid data
    f_test_path = os.path.join(base_path, f_test)
    print(f"exact test path: {f_test_path}")
    df_test = pd.read_csv(f_test_path) # data frame for (exact) test data
    print(f"dataframe for exact test data:")
    print(df_test.columns)


    # Just in case, we use the 4 entries
    if "canonical_conjugator" in X_col_names:
        X_col_names = X_col_names + ["canonical_conjugator_0",
                                     "canonical_conjugator_1",
                                     "canonical_conjugator_2",
                                     "canonical_conjugator_3"]
        X_col_names.remove("canonical_conjugator")

    # Extract the scalar a from "[a, a]"
    def extract_scalar(s) -> float:
        if type(s) != str:
            return s
        else:
            a, b = ast.literal_eval(s)
            return a  # a == b guaranteed

    # Change the bounded data to a value
    if 'q_gonality_bounds' in usecols:
        df_train['q_gonality_bounds'] = df_train['q_gonality_bounds'].apply(extract_scalar)
        df_val['q_gonality_bounds'] = df_val['q_gonality_bounds'].apply(extract_scalar)
        df_test['q_gonality_bounds'] = df_test['q_gonality_bounds'].apply(extract_scalar)

    # Get X and Y's
    X_train = df_train[X_col_names]
    X_val = df_val[X_col_names]
    X_test = df_test[X_col_names]
    Y_train = df_train[Y_col_names]
    Y_val = df_val[Y_col_names]
    Y_test = df_test[Y_col_names]

    return X_train.astype(np.float32), Y_train.astype(np.float32), X_val.astype(np.float32), Y_val.astype(np.float32), X_test.astype(np.float32), Y_test.astype(np.float32)


### just return X_bd and Y_bd
def Classical_Coarse_Bounded_Data_Filter(files = {"File_Exact_Train": "",
                                         "File_Exact_Valid": "",
                                         "File_Exact_Test": "",
                                        "File_Bounded": "",
                                        "File_Bounded_Train": "",
                                        "File_Bounded_Valid": "",
                                        "File_Bounded_Test": ""},
                      X_col_names=["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor","coarse_class_num", "coarse_level"],
                      Y_col_names=['q_gonality_bounds'],
                    folder_name = "data"):
    usecols = X_col_names + Y_col_names
    f_bd = files["File_Bounded"]
    f_bd_train = files["File_Bounded_Train"]
    f_bd_val = files["File_Bounded_Valid"]
    f_bd_test = files["File_Bounded_Test"]
    CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "modular.toml"
    root_path = Path(__file__).resolve().parent.parent.parent
    config = load_toml_config(CONFIG_PATH)
    data_path = os.path.join(root_path, config["FFN"]["nn_data_path"])
    base_path = data_path
    f_bd_path = os.path.join(base_path, f_bd)
    df_bd = pd.read_csv(f_bd_path)

    # Just in case, we use the 4 entries
    if "canonical_conjugator" in X_col_names:
        X_col_names = X_col_names + ["canonical_conjugator_0",
                                     "canonical_conjugator_1",
                                     "canonical_conjugator_2",
                                     "canonical_conjugator_3"]
        X_col_names.remove("canonical_conjugator")

    # Convert Y columns from string to list [a, b]
    def to_list(s: str):
        return list(ast.literal_eval(s))

    # Change the bounded data to a value
    if 'q_gonality_bounds' in usecols:
        df_bd['q_gonality_bounds'] = df_bd['q_gonality_bounds'].apply(to_list)

    # Get X and Y's
    X_bd = df_bd[X_col_names]
    Y_bd = df_bd[Y_col_names]

    X_bd = X_bd.astype(np.float32)
    Y_bd = Y_bd.map(lambda lst: np.array(lst, dtype=np.float32))


    return X_bd, Y_bd



def Classical_Bounded_Data_Filter(files = {"File_Exact_Train": "",
                                           "File_Exact_Valid": "",
                                           "File_Exact_Test": "",
                                           "File_Bounded":"",
                                           "File_Bounded_Train": "",
                                           "File_Bounded_Valid": "",
                                           "File_Bounded_Test": ""},
                      X_col_names=["genus", "rank",
                                   "cusps", "rational_cusps",
                                   "level", "log_conductor",
                                   "coarse_class_num", "coarse_level"],
                      Y_col_names=['q_gonality_bounds'],
                      folder_name = "data"):
    """
        Loads multiple CSVs with Y columns containing interval strings "[a, b]".
        Keeps only rows where all Y columns are exactly "[a, a]",
        and extracts scalar `a` from them.

        :param
        files: dictionary of file paths
        X_col_names: list of column names for X
        Y_col_names: list of column names for Y

        Returns:
            X_bd, Y_bd, X_bd_train, Y_bd_train, X_bd_val, Y_bd_val, X_bd_test, Y_bd_test (pandas DataFrames)
    """

    """
        read the data from files
    """
    usecols = X_col_names + Y_col_names
    f_bd = files["File_Bounded"]
    f_bd_train = files["File_Bounded_Train"]
    f_bd_val = files["File_Bounded_Valid"]
    f_bd_test = files["File_Bounded_Test"]
    CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "modular.toml"
    root_path = Path(__file__).resolve().parent.parent.parent
    config = load_toml_config(CONFIG_PATH)
    data_path = os.path.join(root_path, config["FFN"]["nn_data_path"])
    base_path = data_path
    f_bd_path = os.path.join(base_path, folder_name, f_bd)
    df_bd = pd.read_csv(f_bd_path)
    f_bd_train_path = os.path.join(base_path, folder_name, f_bd_train)
    df_bd_train = pd.read_csv(f_bd_train_path)
    f_bd_val_path = os.path.join(base_path, folder_name,  f_bd_val)
    df_bd_val = pd.read_csv(f_bd_val_path)
    f_bd_test_path = os.path.join(base_path, folder_name, f_bd_test)
    df_bd_test = pd.read_csv(f_bd_test_path)


    # Just in case, we use the 4 entries
    if "canonical_conjugator" in X_col_names:
        X_col_names = X_col_names + ["canonical_conjugator_0",
                                     "canonical_conjugator_1",
                                     "canonical_conjugator_2",
                                     "canonical_conjugator_3"]
        X_col_names.remove("canonical_conjugator")


    # Convert Y columns from string to list [a, b]
    def to_list(s: str):
        return list(ast.literal_eval(s))

    # Change the bounded data to a value
    if 'q_gonality_bounds' in usecols:
        df_bd['q_gonality_bounds'] = df_bd['q_gonality_bounds'].apply(to_list)
        df_bd_train['q_gonality_bounds'] = df_bd_train['q_gonality_bounds'].apply(to_list)
        df_bd_val['q_gonality_bounds'] = df_bd_val['q_gonality_bounds'].apply(to_list)


    # Get X and Y's
    X_bd = df_bd[X_col_names]
    Y_bd = df_bd[Y_col_names]
    X_bd_train = df_bd_train[X_col_names]
    X_bd_val = df_bd_val[X_col_names]
    X_bd_test = df_bd_test[X_col_names]
    Y_bd_train = df_bd_train[Y_col_names]
    Y_bd_val = df_bd_val[Y_col_names]
    Y_bd_test = df_bd_test[Y_col_names]

    X_bd = X_bd.astype(np.float32)
    X_bd_train = X_bd_train.astype(np.float32)
    X_bd_val = X_bd_val.astype(np.float32)
    X_bd_test = X_bd_test.astype(np.float32)
    #Y_bd = Y_bd.astype(object)
    Y_bd = Y_bd.map(lambda lst: np.array(lst, dtype = np.float32))
    #Y_bd_train = Y_bd_train.astype(object)
    Y_bd_train = Y_bd_train.map(lambda lst: np.array(lst, dtype = np.float32))
    #Y_bd_val = Y_bd_val.astype(object)
    Y_bd_val = Y_bd_val.map(lambda lst: np.array(lst, dtype = np.float32))
    #Y_bd_test = Y_bd_test.astype(object)
    Y_bd_test = Y_bd_test.map(lambda lst: np.array(lst, dtype = np.float32))

    return X_bd, Y_bd, X_bd_train, Y_bd_train, X_bd_val, Y_bd_val, X_bd_test, Y_bd_test



############ A Simple Function to compute the bounds accuracy ##############
def bounds_accuracy(X_test, Y_test, model, device, is_equal):
    """
    :param X_test: Test data input X
    :param Y_test: Test data output Y, which are [a, b]'s or just a's
    :param model: the model to be tested
    :param device: the device to be used for computation
    :param is_equal: if True, then we are in [a,a] case,
                    the bounds accuracy is calculated by comparing the predicted bounds to the test data;
    :return: the bounds accuracy of the model on the test data
    """
    # Transform the np data to torch data
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    # get the prediction, also translate back to np data
    Y_pred_test = model.forward(X_test_tensor).cpu().detach().numpy()
    # reduce the predictions to the closest integer
    Y_pred = np.floor(Y_pred_test + 0.5)
    if is_equal == False:
        return np.mean((Y_pred >= Y_test.map(lambda b: b[0])) & (Y_pred <= Y_test.map(lambda b: b[1])))
    else:
        return np.mean(Y_test == Y_pred)



############ compute all the related performances ##############
def performence_all(model = None, X_train =None, Y_train=None, X_val=None, Y_val=None, X_test=None, Y_test=None,X_bd_test=None, Y_bd_test=None, device = torch.device('cuda')):
    """
    :param model:
    :param X_train:
    :param Y_train:
    :param Y_val:
    :param Y_test:
    :return: the accuracy, R^2, RMSE, for four different subsets
    """
    # Transform the np data to torch data
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    # get the prediction, also translate back to np data
    Y_pred_train = model.forward(X_train_tensor).cpu().detach().numpy()
    # reduce the predictions to the closest integer
    Y_pred_train = np.floor(Y_pred_train + 0.5)

    accuracy_train = np.mean(Y_train == Y_pred_train)
    Y_train_np = Y_train.values

    r2_train = 1.0 - (np.sum(((Y_train_np - Y_pred_train) ** 2) / (np.sum((Y_train_np - np.mean(Y_train_np)) ** 2))))
    #print(f"R2 train: {r2_train}")
    rmse_train = np.sqrt(np.mean((Y_train - Y_pred_train) ** 2))


    # Transform the np data to torch data
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    # get the prediction, also translate back to np data
    Y_pred_val = model.forward(X_val_tensor).cpu().detach().numpy()
    # reduce the predictions to the closest integer
    Y_pred_val = np.floor(Y_pred_val + 0.5)

    accuracy_val = np.mean(Y_val == Y_pred_val)
    Y_val_np = Y_val.values
    r2_val = 1.0- (np.sum( ((Y_val_np - Y_pred_val) ** 2) / (np.sum((Y_val_np - np.mean(Y_val_np)) ** 2) )) )
    #print(f"R2 val: {r2_val}")
    rmse_val = np.sqrt(np.mean((Y_val - Y_pred_val) ** 2))

    # Transform the np data to torch data
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    # get the prediction, also translate back to np data
    Y_pred_test = model.forward(X_test_tensor).cpu().detach().numpy()
    # reduce the predictions to the closest integer
    Y_pred_test = np.floor(Y_pred_test + 0.5)

    accuracy_test = np.mean(Y_test == Y_pred_test)
    Y_test_np = Y_test.values
    r2_test = 1.0 - (np.sum(((Y_test_np - Y_pred_test) ** 2) / (np.sum((Y_test_np - np.mean(Y_test_np)) ** 2))))
    #print(f"R2 test: {r2_test}")
    rmse_test = np.sqrt(np.mean((Y_test - Y_pred_test) ** 2))

    def to_list(s: str):
        return list(ast.literal_eval(s))

    # str->list
    # print(f"shape of Y_bd_test: {Y_bd_test.shape}, type of Y_bd_test: {type(Y_bd_test)}")
    if type(Y_bd_test['q_gonality_bounds'][0]) == str:
        Y_bd_test = Y_bd_test.apply(to_list)

    # Transform the np data to torch data
    X_bd_test_tensor = torch.tensor(X_bd_test.values, dtype=torch.float32).to(device)
    # get the prediction, also translate back to np data
    Y_pred_bd_test = model.forward(X_bd_test_tensor).cpu().detach().numpy()
    # reduce the predictions to the closest integer
    Y_pred_bd_test = np.floor(Y_pred_bd_test + 0.5)

    accuracy_bd_test = np.mean((Y_pred_bd_test >= Y_bd_test.map(lambda b: b[0])) & (Y_pred_bd_test <= Y_bd_test.map(lambda b: b[1])))

    return accuracy_train, r2_train, rmse_train, accuracy_val, r2_val, rmse_val, accuracy_test, r2_test, rmse_test, accuracy_bd_test


# from Transformer_Network_and_Dataset_Classes import predict_ft_transformer
# def bounds_accuracy_transfomer(test_dataloader, model, device, Y_test = None, is_equal = True):
#     """
#     :param test_dataloader: Test data, now we have more torch package layers
#     :param model: the model to be tested, should be the network for FT transformer
#     :param device: the device to be used for computation
#     :param Y_test
#     :param is_equal: if True, then we are in [a,a] case,
#                     the bounds accuracy is calculated by comparing the predicted bounds to the test data;
#     :return: the bounds accuracy of the model on the test data
#     """
#     #model.eval()
#     Y_pred_test = predict_ft_transformer(model, test_dataloader, device)
#     Y_pred_test = Y_pred_test.squeeze()
#     Y_pred_test = Y_pred_test.cpu().detach().numpy()
#     #Y_pred_test = Y_pred_test.squeeze()
#     Y_pred = np.floor(Y_pred_test + 0.5)
#     Y_pred = Y_pred.astype(int)
#
#
#     #Y_test_a = Y_test_a.squeeze()
#     #Y_test = pd.DataFrame(Y_test.cpu().detach().numpy())
#     Y_test = Y_test.squeeze()
#
#
#     def to_list(s: str):
#         return list(ast.literal_eval(s))
#
#     # str->list
#     if type(Y_test[0]) == str:
#         Y_test = Y_test.apply(to_list)
#
#     if is_equal == False:
#         print(f"type of Y_pred: {type(Y_pred[0])}, type of Y_test: {type(Y_test[0])}")
#         return np.mean((Y_pred >= Y_test.map(lambda b: b[0])) & (Y_pred <= Y_test.map(lambda b: b[1])))
#     else:
#         return np.mean( Y_test.map(lambda b: b[0]) == Y_pred)




"""
We need another versions of data filter function for weakly supervise situation
"""
def Weakly_Data_Filter(files = {"File_Exact_Train": "",
                                "File_Exact_Valid": "",
                                "File_Exact_Test": "",
                                "File_Bounded": "",
                                "File_Bounded_Train": "",
                                "File_Bounded_Valid": "",
                                "File_Bounded_Test": ""},
                        X_col_names=["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor", "coarse_class_num", "coarse_level"],
                        Y_col_names=['q_gonality_bounds'],
                        folder_name="data"):
    """
    Function split the function to X, Y's in weakly supervised situation
    :param files:
    :param X_col_names:
    :param Y_col_names:
    :return: X_w_train, Y_w_train, X_w_val, Y_w_val, X_w_test, Y_w_test
    """

    usecols = X_col_names + Y_col_names
    f_train = files["File_Exact_Train"]  # this should be a os.path
    f_val = files["File_Exact_Valid"]  # this should be another os.path
    f_test = files["File_Exact_Test"]  # same as above
    base_path = os.path.dirname(os.path.abspath(__file__))
    f_train_path = os.path.join(base_path, folder_name, f_train)
    df_train = pd.read_csv(f_train_path)  # data frame for exact training
    f_val_path = os.path.join(base_path, folder_name, f_val)
    df_val = pd.read_csv(f_val_path)  # data frame for valid data
    f_test_path = os.path.join(base_path, folder_name, f_test)
    df_test = pd.read_csv(f_test_path)  # data frame for (exact) test data

    f_bd = files["File_Bounded"]
    f_bd_train = files["File_Bounded_Train"]
    f_bd_val = files["File_Bounded_Valid"]
    f_bd_test = files["File_Bounded_Test"]
    base_path = os.path.dirname(os.path.abspath(__file__))
    f_bd_path = os.path.join(base_path, "data", f_bd)
    df_bd = pd.read_csv(f_bd_path)
    f_bd_train_path = os.path.join(base_path, folder_name, f_bd_train)
    df_bd_train = pd.read_csv(f_bd_train_path)
    f_bd_val_path = os.path.join(base_path, folder_name, f_bd_val)
    df_bd_val = pd.read_csv(f_bd_val_path)
    f_bd_test_path = os.path.join(base_path, folder_name, f_bd_test)
    df_bd_test = pd.read_csv(f_bd_test_path)

    ### put the two different sets together for train, val and test
    df_train_comb = pd.concat([df_train, df_bd_train])
    df_val_comb = pd.concat([df_val, df_bd_val])
    df_test_comb = pd.concat([df_test, df_bd_test])

    # Just in case, we use the 4 entries
    if "canonical_conjugator" in X_col_names:
        X_col_names = X_col_names + ["canonical_conjugator_0",
                                     "canonical_conjugator_1",
                                     "canonical_conjugator_2",
                                     "canonical_conjugator_3"]
        X_col_names.remove("canonical_conjugator")
        usecols = X_col_names + Y_col_names

    # Convert Y columns from string to list [a, b]
    def to_list(s: str):
        return list(ast.literal_eval(s))

    # Change the bounded data to a value
    if 'q_gonality_bounds' in usecols:
        #df_bd['q_gonality_bounds'] = df_bd['q_gonality_bounds'].apply(to_list)
        df_train_comb['q_gonality_bounds'] = df_train_comb['q_gonality_bounds'].apply(to_list)
        df_val_comb['q_gonality_bounds'] = df_val_comb['q_gonality_bounds'].apply(to_list)
        df_test_comb['q_gonality_bounds'] = df_test_comb['q_gonality_bounds'].apply(to_list)
        # need to modify the column names related to lower bound and upper bound
        Y_col_names+= ['q_gonality_lower', 'q_gonality_upper']
        Y_col_names.remove('q_gonality_bounds')
        usecols+=['q_gonality_lower', 'q_gonality_upper']
        usecols.remove('q_gonality_bounds')

    # now, need to split the Y into lower bound and upper bounds, that will be q_gonality_lower, q_gonality_upper
    df_train_comb['q_gonality_lower'] = df_train_comb['q_gonality_bounds'].map(lambda lst: lst[0])
    df_train_comb['q_gonality_upper'] = df_train_comb['q_gonality_bounds'].map(lambda lst: lst[1])
    df_val_comb['q_gonality_lower'] = df_val_comb['q_gonality_bounds'].map(lambda lst: lst[0])
    df_val_comb['q_gonality_upper'] = df_val_comb['q_gonality_bounds'].map(lambda lst: lst[1])
    df_test_comb['q_gonality_lower'] = df_test_comb['q_gonality_bounds'].map(lambda lst: lst[0])
    df_test_comb['q_gonality_upper'] = df_test_comb['q_gonality_bounds'].map(lambda lst: lst[1])

    # Just in case, if we lazily only put a col name = "canonical_conjugator", we replace it by the 4 entries
    if "canonical_conjugator" in X_col_names:
        X_col_names = X_col_names + ["canonical_conjugator_0",
                                     "canonical_conjugator_1",
                                     "canonical_conjugator_2",
                                     "canonical_conjugator_3"]
        X_col_names.remove("canonical_conjugator")

    X_w_train = df_train_comb[X_col_names]
    Y_w_train = df_train_comb[Y_col_names]
    X_w_val = df_val_comb[X_col_names]
    Y_w_val = df_val_comb[Y_col_names]
    X_w_test = df_test_comb[X_col_names]
    Y_w_test = df_test_comb[Y_col_names]

    return X_w_train.astype(np.float32), Y_w_train.astype(np.float32), X_w_val.astype(np.float32), Y_w_val.astype(np.float32), X_w_test.astype(np.float32), Y_w_test.astype(np.float32)


