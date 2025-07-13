import csv
import itertools
import json
import time

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
import os
import sys
# Add the project root to Python path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
from src.utils.get_config import load_toml_config


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

"""
import the prep classes and functions we did
"""

from NN_Preparation_Functions import device_verification, create_directory
from NN_Plotter_and_Log_classes import ExperimentResultPlotter
from NN_Accuracy import Classical_Exact_Data_Filter, Classical_Bounded_Data_Filter, Classical_Coarse_Bounded_Data_Filter, bounds_accuracy, performence_all
from NN_Saving_Classes import Saving_Experiments
from NN_Network_and_Dataset_Classes import InMemoryDataset, BasicNeuralNetwork







####### Want to use the following to print logs to both screen and log file
class TeeLogger:
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()



######### A Midware function for shorter implementation to save torch data #######
def save_torch_X_and_Y(X = None, Y = None, torch_data_path = None, data_file_name = "torch_data", mode = "train"):
    print(f"loading train data...")
    data = InMemoryDataset(X=X, Y=Y)
    print(f"saving torch train data...")
    current_data_path = os.path.join(torch_data_path, "torch_data")
    if os.path.exists(current_data_path) == False:
        os.mkdir(current_data_path)
    data_name = "torch_data" + "_" + mode + ".pt"
    torch.save(data, os.path.join(current_data_path, data_name))
    return data, os.path.join(current_data_path, data_name)





def classical_train(base_path,
                    batch_size,
                    optimizer,
                    learning_rate,
                    hidden_sizes,
                    activation_function,
                    num_epochs,
                    train_ratio,
                    valid_test_ratio,
                    criterion,
                    X_col_names=["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor","coarse_class_num", "coarse_level"],
                    Y_col_names=["q_gonality_bounds"],
                    drop_col_names = ["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor",
                                 "coarse_class_num", "coarse_level"],
                    data_dir_name = None,
                    data_split_idx=0,
                    is_loading_existing_train_data=False,
                    is_loading_existing_valid_data=False,
                    is_loading_existing_test_data=False,
                    is_loading_existing_bounds_data=False,
                    file_names=["combined_data_7.h5"],
                    is_testing=False,
                    is_experiment_for_different_split = True):
    """
    We train the model by this function,
    and we can set up hyperparameters for the model
    as well as the saving/loading paths
    :param base_path: the base path for the specific file
    :param batch_size:
    :param optimizer:
    :param learning_rate:
    :param hidden_sizes:
    :param activation_function:
    :param num_epochs:
    :param train_ratio: if train_ratio = 0.9, then 10% data will be valid and test data
    :param valid_test_ratio: if 10% is valid and test data, and valid_test_ratio = 0.5,
            then 5% of data will be valid and 5% of data will be test
    :param is_loading_existing_train_data: if True, then we will load the existing torch train data
    :param is_loading_existing_valid_data: if True, then we will load the existing torch valid data
    :param is_loading_existing_test_data: if True, then we will load the existing torch test data
    :param is_loading_existing_bounds_data: if True, then we will load the existing torch bounds data
    :param is_testing: if testing, we consider test accuracy, bounded accuract and save such models
    :return: train_losses, valid_losses, test_accuracy, bounded_accuracy
    """

    ##### check the devices and print the device information #####
    device = device_verification()
    ##### Get the base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    ##### Lists of Hyperparameters #####
    batch_size = batch_size
    optimizer = optimizer
    learning_rate = learning_rate
    hidden_sizes = hidden_sizes
    activation_function = activation_function
    num_epochs = num_epochs
    train_ratio = train_ratio
    valid_test_ratio = valid_test_ratio
    criterion = criterion
    X_col_names = X_col_names
    Y_col_names = Y_col_names

    ##### Build a dictionary for hyperparams
    hyperparam = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_sizes': hidden_sizes,
        'optimizer': optimizer,
        'criterion': criterion,
        'num_epochs': num_epochs,
        'activation_function': activation_function,
        'X_col_names': X_col_names,  #
        'Y_col_names': Y_col_names,
        'drop_col_names': drop_col_names,
        'train_ratio': train_ratio,
        'valid_test_ratio': valid_test_ratio
    }

    ##### Build a class to handle the saving jobs, and also save the hyperparams in a separated folder
    saver = Saving_Experiments(
        save_dir_name="Experiment_Classical",
        hyperparam = hyperparam,
        model = None,
        accuracy_list=[],
        bounded_accuracy_list=[],
        data_split_idx= data_split_idx,
        is_experiment_for_different_split = is_experiment_for_different_split
    )

    ##### For the h5 data reading from base_path/data
    if data_dir_name is None:
        data_dir_name = "data"
    data_read_path = os.path.join(base_path, data_dir_name)

    # Where to find all the train-valid-test data for X and Y's
    file_names_for_train_valid_test = {}
    if is_experiment_for_different_split:
        file_names_for_train_valid_test = {"File_Exact_Train": "train_fold_"+str(data_split_idx)+".csv",
            "File_Exact_Valid": "valid_fold_"+str(data_split_idx)+".csv",
            "File_Exact_Test": "test_data.csv",
            "File_Bounded": "q_gonality_diff.csv",
            "File_Bounded_Train": "",
            "File_Bounded_Valid": "",
            "File_Bounded_Test": ""}
    else:
        file_names_for_train_valid_test = {"File_Exact_Train": "train_small.csv",
            "File_Exact_Valid": "valid_small.csv",
            "File_Exact_Test": "level_large.csv",
            "File_Bounded": "q_gonality_diff.csv",
            "File_Bounded_Train": "",
            "File_Bounded_Valid": "",
            "File_Bounded_Test": ""
        }

    X_train, Y_train, X_val, Y_val, X_test, Y_test = Classical_Exact_Data_Filter(files = file_names_for_train_valid_test,
                                                                                X_col_names=X_col_names,
                                                                                Y_col_names=Y_col_names)
    X_bd, Y_bd = Classical_Coarse_Bounded_Data_Filter(files = file_names_for_train_valid_test,
                                                                X_col_names=X_col_names,
                                                                Y_col_names=Y_col_names)

    ##### Find the folder to save things for current experiment
    save_path  = saver.experiment_dir_path

    """
        Set up log path
    """
    # Set up tee logger
    log_file_path = os.path.join(save_path, "log.txt")
    with open(log_file_path, 'w') as f:
        f.write(f"Log created...\n")

    sys.stdout = TeeLogger(log_file_path)

    # init our model
    if 'canonical_conjugator' in X_col_names:
        input_len = len(X_col_names) + 3
    else:
        input_len = len(X_col_names)
    model = BasicNeuralNetwork(input_size=input_len,
                               hidden_sizes=hidden_sizes,
                               output_size=len(Y_col_names),
                               activation_function=activation_function)
    model.to(device)

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer must be Adam or SGD.")

    ##### We need to put data into pytorch, and also save
    train_data, torch_train_data_path = save_torch_X_and_Y(X = X_train,
                                               Y = Y_train,
                                               torch_data_path=save_path,
                                               data_file_name = "torch_data",
                                               mode = "train")
    valid_data, torch_valid_data_path = save_torch_X_and_Y(X = X_val,
                                               Y= Y_val,
                                               torch_data_path=save_path,
                                               data_file_name = "torch_data",
                                               mode = "valid")
    test_data, torch_test_data_path = save_torch_X_and_Y(X = X_test,
                                              Y= Y_test,
                                              torch_data_path=save_path,
                                              data_file_name = "torch_data",
                                              mode = "test")
    # bd_test_data, torch_bd_test_data_path = save_torch_X_and_Y(X = X_bd,
    #                                                            Y = Y_bd,
    #                                                            torch_data_path=save_path,
    #                                                            data_file_name = "torch_data",
    #                                                            mode = "bd_test")
    bd_test_data = X_bd # an ugly replacement of the above buggy code


    ######### Find the lengths of datasets #######
    train_size = len(train_data)
    valid_size = len(valid_data)
    test_size = len(test_data)
    bd_test_size = len(bd_test_data)
    print(f"train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}, bounded_test_size: {bd_test_size}")


    ######### Start the training ############
    print(f"Start to train the model:")
    print(f"number_epoch: "
          f"num_epochs: {str(num_epochs)}, "
          f"batch_size: {str(batch_size)}, "
          f"learning_rate: {str(learning_rate)}, "
          f"hidden_sizes: {str(hidden_sizes)}, "
          f"train_ratio: {str(train_ratio)}, "
          f"optimizer: {optimizer.__class__.__name__}, "
          f"criterion: {criterion.__class__.__name__}, "
          f"activation_function: {activation_function}, "
          f"X_col_names: {X_col_names}, "
          f"Y_col_names: {Y_col_names}. ")
    #### Lists to save losses and accuracies ####
    train_losses = []
    val_losses = []
    train_accuracies = []
    valid_accuracies = []
    bd_accuracies = []
    #### Set-up Dataloaders
    train_loader = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

    valid_loader = DataLoader(dataset=valid_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}... ")
        train_loss = model.train_one_epoch(DataLoader=train_loader,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           device=device)
        print(f"Now compute the losses for the epoch {str(epoch + 1)}...")
        val_loss = model.evaluate(data = valid_loader,
                                  criterion=criterion,
                                  device=device)
        ######### Model Saving ############
        if (len(val_losses) > 0 and val_loss < min(val_losses)) or (epoch==0):
            print(f"Best Valid Loss in Epoch {epoch + 1}...")
            print(f"Saving the best model...")
            model_save_path_best = os.path.join(save_path, "best_valid_models")
            if os.path.exists(model_save_path_best) == False:
                os.mkdir(model_save_path_best)
            best_valid_model_path = saver.save_model_custom_path(model = model,
                                                                 path = "best_valid_models",
                                                                 model_name="best_valid_model",
                                                                 epoch_idx=epoch+1)
            #print(f"Best Valid Model saved at {best_valid_model_path}")

        model_save_path = os.path.join(save_path, "all_models")
        if os.path.exists(model_save_path) == False:
            os.mkdir(model_save_path)
        model_save_path = saver.save_model_custom_path(model = model,
                                                       path = "all_models",
                                                       model_name="model",
                                                       epoch_idx=epoch+1)
        #print(f"Model saved at {model_save_path}")

        ##### After saving current best valid loss model, add data
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        ############ Accuracies ############
        print(f"Now compute the accuracies for the epoch {str(epoch + 1)}...")
        train_accuracy = bounds_accuracy(X_test=X_train, Y_test=Y_train, model=model, device=device, is_equal=True)
        train_accuracies.append(train_accuracy)
        valid_accuracy = bounds_accuracy(X_test=X_val, Y_test=Y_val, model=model, device=device, is_equal=True)
        valid_accuracies.append(valid_accuracy)
        bd_accuracy = bounds_accuracy(X_test=X_bd, Y_test=Y_bd, model=model, device=device,
                                           is_equal=False)
        bd_accuracies.append(bd_accuracy)
        print(f"Epoch {epoch + 1} "
              f"train loss: {train_loss:.4f}, "
              f"valid loss: {val_loss:.4f}, "
              f"train accuracy: {train_accuracy * 100:.4f}%, "
              f"valid accuracy: {valid_accuracy * 100:.4f}%, "
              f"bounds accuracy: {bd_accuracy * 100:.4f}%\n")
        ####### End of the loop #######

    end_times =  time.time()

    ######## End Training: Save loss and model trained above #########
    test_accuracy = bounds_accuracy(X_test=X_test,
                                    Y_test=Y_test,
                                    model=model,
                                    device=device,
                                    is_equal=True)
    bd_test_accuracy = bounds_accuracy(X_test=X_bd,
                                       Y_test=Y_bd,
                                       model=model,
                                       device=device,
                                       is_equal=False)
    print(f"END OF TRAIN: accuracy: {test_accuracy * 100: .2f}%; bounds_accuracy: {bd_test_accuracy * 100:.2f}%\n")



    ######## After the training: save everything to correct path ##################
    final_model_path = saver.save_model(model = model,
                                        model_name = "final_model",
                                        epoch_idx = num_epochs)
    loss_list_path = saver.save_train_valid_losses(train_losses = train_losses,
                                                   val_losses = val_losses,
                                                   loss_file_name= "loss")
    test_accuracies = []
    for i in range(num_epochs):
        model = torch.load(os.path.join(save_path, "all_models", "model_"+str(i+1)+".pt"), weights_only=False)
        #y_test_pred = model.predict(X_test, device=device)
        test_accuracy = bounds_accuracy(X_test=X_test, Y_test=Y_test, model=model, device=device, is_equal=True)
        test_accuracies.append(test_accuracy)
    accuracy_list_path = saver.save_accuracy_lists(train_accuracy_list =train_accuracies,
                                                   valid_accuracy_list=valid_accuracies,
                                                   accuracy_list=test_accuracies,
                                                   bounded_accuracy_list=bd_accuracies,
                                                   accuracy_name="accuracy")

    # choose the model with best valid accuracy, then evaluate its performaces
    # choose the idx in valid_accuracies with highest value
    best_valid_idx = valid_accuracies.index(max(valid_accuracies))
    best_valid_model_path = os.path.join(save_path, "best_valid_models", "best_valid_model_"+str(best_valid_idx+1)+".pt")
    best_valid_model = torch.load(best_valid_model_path, weights_only = False)
    accuracy_train, r2_train, rmse_train, accuracy_val, r2_val, rmse_val, accuracy_test, r2_test, rmse_test, accuracy_bd_test = performence_all(model = best_valid_model,
                                                                                                                                                X_train=X_train,
                                                                                                                                                Y_train = Y_train,
                                                                                                                                                X_val = X_val,
                                                                                                                                                Y_val = Y_val,
                                                                                                                                                X_test = X_test,
                                                                                                                                                Y_test = Y_test,
                                                                                                                                                X_bd_test = X_bd,
                                                                                                                                                Y_bd_test = Y_bd,
                                                                                                                                                device=device)

    # Add this code after the performance metrics are calculated
    performances = {
        "train": {
            "accuracy": float(accuracy_train),
            "r2": float(r2_train),
            "rmse": float(rmse_train)
        },
        "validation": {
            "accuracy": float(accuracy_val),
            "r2": float(r2_val),
            "rmse": float(rmse_val)
        },
        "test": {
            "accuracy": float(accuracy_test),
            "r2": float(r2_test),
            "rmse": float(rmse_test)
        },
        "bounded_test": {
            "accuracy": float(accuracy_bd_test)
        },
        "running_times": {
            "training_time": float(end_times - start_time)
        }
    }

    # Save to JSON file
    performance_file_path = os.path.join(save_path, "performance_metrics.json")
    with open(performance_file_path, 'w') as f:
        json.dump(performances, f, indent=4)



    print(f"Plot the graphs after the training...\n")
    plotter, plot_path = saver.save_pictures(model= model,
                                             plot_name = "plot",
                                             train_losses = train_losses,
                                             valid_losses = val_losses)
    print(f"Plot done.")


###############################################################################
###############################################################################
######################## Some Sample Experiments ##############################
###############################################################################
###############################################################################

def experiment_1():
    """
    Test case 1: we test the classical train on exactly one hyperparameters
    :return:
    """
    ##### The following are different setups of hyperparams to test ########
    hyperparameters = {
        'batch_size': [32],
        'learning_rate': [0.0001],
        'hidden_sizes': [[128, 32]],
        'optimizer': ['Adam'],
        'criterion': [nn.MSELoss()],
        'num_epochs': [2000],
        'activation_function': [nn.LeakyReLU(negative_slope=0.01)],
        'X_col_names': [["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor", "coarse_class_num", "coarse_level",
             "canonical_conjugator"]],  #
        'Y_col_names': [['q_gonality_bounds']],
        'drop_col_names': [["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor", "coarse_class_num", "coarse_level"]],
        'train_ratio': [0.8],
        'valid_test_ratio': [0.5]
    }
    # Get a list of hyperparameters
    param_list = [dict(zip(hyperparameters.keys(), values)) for values in itertools.product(*hyperparameters.values())]
    # Test the hyperparameter parse
    # for hyperparameter in param_list:
    #     print(hyperparameter)
    #     print("\n")

    CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "modular.toml"
    root_path = Path(__file__).resolve().parent.parent.parent
    config = load_toml_config(CONFIG_PATH)
    data_path = os.path.join(root_path, config["FFN"]["nn_data_path"])

    for hyperparameter in param_list:
        batch_size = hyperparameter['batch_size']
        learning_rate = hyperparameter['learning_rate']
        hidden_sizes = hyperparameter['hidden_sizes']
        activation_function = hyperparameter['activation_function']
        num_epochs = hyperparameter['num_epochs']
        criterion = hyperparameter['criterion']
        train_ratio = hyperparameter['train_ratio']
        valid_test_ratio = hyperparameter['valid_test_ratio']
        base_path = data_path
        optimizer = hyperparameter['optimizer']

        for i in range(5):
            print(f"Current Hyperparameters: {hyperparameters}" + f"\n")
            print(f"Currently, Experiment for: {i}th Data Split" + f"\n")
            classical_train(batch_size=batch_size,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                        hidden_sizes=hidden_sizes,
                        activation_function=activation_function,
                        num_epochs=num_epochs,
                        criterion=criterion,
                        X_col_names=hyperparameter['X_col_names'],
                        Y_col_names=hyperparameter['Y_col_names'],
                        drop_col_names = hyperparameter['drop_col_names'],
                        base_path=base_path,
                        train_ratio=train_ratio,
                        valid_test_ratio=valid_test_ratio,
                        data_split_idx = i,
                        is_experiment_for_different_split = True
                        )

    return None

def experiment_2():
    """
    We need to do an experiment for training only on smaller level data,
    and test only on larger level data
    :return: None
    """
    ##### The following are different setups of hyperparams to test ########
    hyperparameters = {
        'batch_size': [32],
        'learning_rate': [0.0001],
        'hidden_sizes': [[128, 32]],
        'optimizer': ['Adam'],
        'criterion': [nn.MSELoss()],
        'num_epochs': [2000],
        'activation_function': [nn.LeakyReLU(negative_slope=0.01)],
        'X_col_names': [
            ["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor", "coarse_class_num", "coarse_level",
             "canonical_conjugator"]],  #
        'Y_col_names': [['q_gonality_bounds']],
        'drop_col_names': [
            ["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor", "coarse_class_num", "coarse_level"]],
        'train_ratio': [0.8],
        'valid_test_ratio': [0.5]
    }
    # Get a list of hyperparameters
    param_list = [dict(zip(hyperparameters.keys(), values)) for values in itertools.product(*hyperparameters.values())]
    # Test the hyperparameter parse
    # for hyperparameter in param_list:
    #     print(hyperparameter)
    #     print("\n")

    CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "modular.toml"
    root_path = Path(__file__).resolve().parent.parent.parent
    config = load_toml_config(CONFIG_PATH)
    data_path = os.path.join(root_path, config["FFN"]["nn_data_path"])

    for hyperparameter in param_list:
        batch_size = hyperparameter['batch_size']
        learning_rate = hyperparameter['learning_rate']
        hidden_sizes = hyperparameter['hidden_sizes']
        activation_function = hyperparameter['activation_function']
        num_epochs = hyperparameter['num_epochs']
        criterion = hyperparameter['criterion']
        train_ratio = hyperparameter['train_ratio']
        valid_test_ratio = hyperparameter['valid_test_ratio']
        base_path = data_path
        optimizer = hyperparameter['optimizer']

        print(f"Current Hyperparameters: {hyperparameters}" + f"\n")

        classical_train(batch_size=batch_size,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                        hidden_sizes=hidden_sizes,
                        activation_function=activation_function,
                        num_epochs=num_epochs,
                        criterion=criterion,
                        X_col_names=hyperparameter['X_col_names'],
                        Y_col_names=hyperparameter['Y_col_names'],
                        drop_col_names=hyperparameter['drop_col_names'],
                        base_path=base_path,
                        train_ratio=train_ratio,
                        valid_test_ratio=valid_test_ratio,
                        data_split_idx=0,
                        is_experiment_for_different_split=False
                        )




if __name__ == "__main__":
    experiment_1() # loop for 5 different split experiment
    experiment_2() # train for small levels, test for large levels

