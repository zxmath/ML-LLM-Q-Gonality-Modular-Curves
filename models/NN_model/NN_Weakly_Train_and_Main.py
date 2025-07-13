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
import copy

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
from NN_Data_Processing_Accuracy import Classical_Exact_Data_Filter, Classical_Bounded_Data_Filter, bounds_accuracy, Weakly_Data_Filter
from NN_Saving_Reading_Classes import Saving_Experiments
from NN_Network_and_Dataset_Classes import InMemoryDataset, IntervalDataset, BasicNeuralNetwork, interval_loss


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
######### this is a variation for interval/weak supervised learning #########
def save_torch_X_and_a_b(X = None, a = None, b = None, torch_data_path = None, data_file_name = "torch_data", mode = "train"):
    print(f"loading train data...")
    data = IntervalDataset(X=X, a=a, b=b)
    print(f"saving torch train data...")
    current_data_path = os.path.join(torch_data_path, "torch_data_interval")
    if os.path.exists(current_data_path) == False:
        os.mkdir(current_data_path)
    data_name = "torch_data" + "_" + mode + ".pt"
    torch.save(data, os.path.join(current_data_path, data_name))
    return data, os.path.join(current_data_path, data_name)





###### For testing the implementation: print the model shapes
def print_model_shapes(model, input_size):
    """
    Print the shapes of the model's layers for an input of given size.

    Args:
        model: The neural network model
        input_size: The size of the input tensor (int or tuple)
    """
    # If input_size is just a number, convert it to a tuple
    if isinstance(input_size, int):
        input_size = (1, input_size)  # Batch size of 1
    else:
        # Ensure batch dimension
        if len(input_size) == 1:
            input_size = (1, input_size[0])

    # Create a dummy input tensor
    x = torch.zeros(input_size).to(next(model.parameters()).device)

    print(f"\n{'=' * 50}")
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"{'=' * 50}")

    # Get all modules in order
    if hasattr(model, 'layers'):
        # For models with a sequential container called 'layers'
        modules = model.layers
        print(f"Input shape: {tuple(x.shape)}")

        # Pass through each layer and print shape
        for i, layer in enumerate(modules):
            x = layer(x)
            print(f"Layer {i}: {layer.__class__.__name__} → Output shape: {tuple(x.shape)}")
    else:
        # For models with individual layer attributes
        print(f"Input shape: {tuple(x.shape)}")

        # Assuming your BasicNeuralNetwork structure has input_layer, hidden_layers, and output_layer
        if hasattr(model, 'input_layer'):
            x = model.input_layer(x)
            print(f"Input Layer: {model.input_layer.__class__.__name__} → Output shape: {tuple(x.shape)}")

        if hasattr(model, 'hidden_layers'):
            for i, layer in enumerate(model.hidden_layers):
                x = layer(x)
                print(f"Hidden Layer {i}: {layer.__class__.__name__} → Output shape: {tuple(x.shape)}")

        if hasattr(model, 'output_layer'):
            x = model.output_layer(x)
            print(f"Output Layer: {model.output_layer.__class__.__name__} → Output shape: {tuple(x.shape)}")

    print(f"{'=' * 50}")

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'=' * 50}\n")

    return x.shape  # Return the final output shape


####### training using the weak loss function #########
def weakly_train(base_path,
                    batch_size,
                    optimizer,
                    learning_rate,
                    hidden_sizes,
                    activation_function,
                    num_epochs,
                    train_ratio,
                    valid_test_ratio,
                    criterion = interval_loss,
                    X_col_names=["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor","coarse_class_num", "coarse_level"],
                    Y_col_names=["q_gonality_bounds"],
                    drop_col_names = ["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor",
                                 "coarse_class_num", "coarse_level"],
                    data_dir_name = None,
                    is_trained = False,
                    trained_model = None):
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
    :param is_trained: bool, using trained model by classical train function or not
    :return: train_losses, valid_losses, test_accuracy, bounded_accuracy
    """

    ##### check the devices and print the device information #####
    device = device_verification()
    ##### Get the base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    ##### Lists of Hyperparameters #####
    batch_size_1 = batch_size
    optimizer_1 = optimizer
    learning_rate_1 = learning_rate
    hidden_sizes_1 = hidden_sizes
    activation_function_1 = activation_function
    num_epochs_1 = num_epochs
    train_ratio_1 = train_ratio
    valid_test_ratio_1 = valid_test_ratio
    criterion_1 = criterion
    X_col_names_1 = copy.deepcopy(X_col_names)
    Y_col_names_1 = copy.deepcopy(Y_col_names)
    print(f"Y_col_names check 0: {Y_col_names}")

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
        bounded_accuracy_list=[]
    )

    ##### For the h5 data reading from base_path/data
    if data_dir_name is None:
        data_dir_name = "data"
    data_read_path = os.path.join(base_path, data_dir_name)

    # Where to find all the train-valid-test data for X and Y's
    file_names_for_train_valid_test = {"File_Exact_Train": "q_gonality_same_train.csv",
            "File_Exact_Valid": "q_gonality_same_val.csv",
            "File_Exact_Test": "q_gonality_same_test.csv",
            "File_Bounded": "q_gonality_diff.csv",
            "File_Bounded_Train": "q_gonality_diff_train.csv",
            "File_Bounded_Valid": "q_gonality_diff_val.csv",
            "File_Bounded_Test": "q_gonality_diff_test.csv"}

    X_w_train, Y_w_train, X_w_val, Y_w_val, X_w_test, Y_w_test = Weakly_Data_Filter(files = file_names_for_train_valid_test,
                                                                                    X_col_names=X_col_names,
                                                                                    Y_col_names=Y_col_names)

    """
    Despite of using the combined train-val-test sets, 
    for accuracies, we still need the splitted train-val-test sets in classical settings
    So let's do it below
    """
    print(f"Y_col_names check 1: {Y_col_names_1}")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = Classical_Exact_Data_Filter(files = file_names_for_train_valid_test,
                                                                                X_col_names=X_col_names_1,
                                                                              Y_col_names=Y_col_names_1)
    print(f"Y_col_names check 2: {Y_col_names_1}")
    X_bd, Y_bd, X_bd_train, Y_bd_train, X_bd_val, Y_bd_val, X_bd_test, Y_bd_test = Classical_Bounded_Data_Filter(files = file_names_for_train_valid_test,
                                                                                                                 X_col_names=X_col_names_1,
                                                                                                                 Y_col_names=Y_col_names_1)


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

    # either init a model or use the trained model by classical supervised learning
    if is_trained == False:
        model = BasicNeuralNetwork(input_size=input_len,
                               hidden_sizes=hidden_sizes,
                               output_size=len(Y_col_names),
                               activation_function=activation_function)
    else:
        if trained_model is None:
            raise ValueError("Please provide the trained model for the next training")
        else:
            model = trained_model
    model.to(device)

    # set up optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer must be Adam or SGD.")

    ##### We need to put data into pytorch, and also save
    ##### Note that the save torch function below can also return the torch Dataset
    train_data, torch_train_data_path = save_torch_X_and_a_b(X = X_w_train,
                                               a = Y_w_train["q_gonality_lower"],
                                               b = Y_w_train["q_gonality_upper"],
                                               torch_data_path=save_path,
                                               data_file_name = "torch_data_interval",
                                               mode = "train")
    valid_data, torch_valid_data_path = save_torch_X_and_a_b(X = X_w_val,
                                               a = Y_w_val['q_gonality_lower'],
                                               b = Y_w_val['q_gonality_upper'],
                                               torch_data_path=save_path,
                                               data_file_name = "torch_data_interval",
                                               mode = "valid")
    test_data, torch_test_data_path = save_torch_X_and_a_b(X = X_w_test,
                                              a= Y_w_test['q_gonality_lower'],
                                              b = Y_w_test['q_gonality_upper'],
                                              torch_data_path=save_path,
                                              data_file_name = "torch_data_interval",
                                              mode = "test")

    ######### Find the lengths of datasets #######
    train_size = len(train_data)
    valid_size = len(valid_data)
    test_size = len(test_data)
    print(f"train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}")

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
    test_accuracies = []
    bd_test_accuracies = []

    # Dataloaders for train and val
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

    ######## Now we start the train loop
    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}... ")
        train_loss = model.train_one_epoch_interval(DataLoader=train_loader,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           device=device)
        print(f"Now compute the losses for the epoch {str(epoch + 1)}...")
        val_loss = model.evaluate_interval(data = valid_loader,
                                  criterion=criterion,
                                  device=device)
        ######### Model Saving ############
        if len(val_losses) > 0 and val_loss < min(val_losses):
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
        model_save_path = saver.save_model_custom_path(model=model,
                                                       path="all_models",
                                                       model_name="model",
                                                       epoch_idx=epoch + 1)

        ##### After saving current best valid loss model, add data
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        ############ Accuracies ############
        print(f"Now compute the accuracies for the epoch {str(epoch + 1)}...")
        test_accuracy = bounds_accuracy(X_test=X_test, Y_test=Y_test, model=model, device=device, is_equal=True)
        test_accuracies.append(test_accuracy)
        bd_test_accuracy = bounds_accuracy(X_test=X_bd_test, Y_test=Y_bd_test, model=model, device=device,
                                           is_equal=False)
        bd_test_accuracies.append(bd_test_accuracy)
        print(f"Epoch {epoch + 1} train loss: {train_loss:.4f}, valid loss: {val_loss:.4f}, test accuracy: {test_accuracy * 100:.4f}%, bounds test accuracy: {bd_test_accuracy * 100:.4f}%\n")
        ##### End Loop #####

    ######## After the training: save everything to correct path ##################
    final_model_path = saver.save_model(model = model,
                                        model_name = "final_model",
                                        epoch_idx = num_epochs)
    loss_list_path = saver.save_train_valid_losses(train_losses = train_losses,
                                                   val_losses = val_losses,
                                                   loss_file_name= "loss")
    accuracy_list_path = saver.save_accuracy_lists(accuracy_list=test_accuracies,
                                                   bounded_accuracy_list=bd_test_accuracies,
                                                   accuracy_name="accuracy")

    print(f"Plot the graphs after the training...\n")
    plotter, plot_path = saver.save_pictures(model= model,
                                             plot_name = "plot",
                                             train_losses = train_losses,
                                             valid_losses = val_losses)
    print(f"Plot done.")

def experiment_2():
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
        'criterion': [interval_loss],
        'num_epochs': [2],
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

    for hyperparameter in param_list:
        batch_size = hyperparameter['batch_size']
        learning_rate = hyperparameter['learning_rate']
        hidden_sizes = hyperparameter['hidden_sizes']
        activation_function = hyperparameter['activation_function']
        num_epochs = hyperparameter['num_epochs']
        criterion = hyperparameter['criterion']
        train_ratio = hyperparameter['train_ratio']
        valid_test_ratio = hyperparameter['valid_test_ratio']
        base_path = os.path.dirname(os.path.abspath(__file__))
        optimizer = hyperparameter['optimizer']


        print(f"Current Hyperparameters: {hyperparameters}" + f"\n")

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "_Experiment_Classical20250616_005951", "model", "final_model_2000.pt")
    model = torch.load(model_path, weights_only=False)

    print_model_shapes(model = model, input_size = 12)

    weakly_train(batch_size=batch_size,
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
                is_trained=True,
                trained_model=model)


    return None

if __name__ == "__main__":
    experiment_2()
