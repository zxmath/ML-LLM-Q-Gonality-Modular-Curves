import itertools

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
# import torch.nn.functional as F
import torch.optim as optim
from jedi.api import file_name
from numba.pycc.platform import get_configs
from sympy.strategies.core import switch
# from jupyter_server.auth import passwd
# from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import namedtuple
from NN_Plotter_and_Log_classes import ExperimentResultPlotter ## use it to plot and save graphs

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
from Find_Root_Path import find_project_root


class Saving_Experiments:
    def __init__(self,
        save_dir_name = "Experiment_Classical",
        hyperparam:dict = {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'hidden_sizes': [128, 16],
        'optimizer': 'Adam',
        'criterion': nn.MSELoss(),
        'num_epochs': 2000,
        'activation_function': nn.LeakyReLU(negative_slope=0.1),
        'X_col_names': ["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor", "coarse_class_num", "coarse_level",
             "canonical_conjugator"],  #
        'Y_col_names': ['q_gonality_bounds'],
        'drop_col_names': ["genus", "rank", "cusps", "rational_cusps", "level", "log_conductor", "coarse_class_num", "coarse_level"],
        'train_ratio': 0.9,
        'valid_test_ratio': 0.5
        },
        model = None,
        accuracy_list = [],
        bounded_accuracy_list = [],
        data_split_idx = 0,
        is_experiment_for_different_split = True):


        self.accuracy_list = accuracy_list # list of accuracy for each epoch
        self.bounded_accuracy_list = bounded_accuracy_list # list of bounded accuracy for each epoch

        self.train_losses = []
        self.valid_losses = []

        self.model = model

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # we save each combination of hyperparams to one separated folder

        # get current time to avoid repetitive names of folders
        if is_experiment_for_different_split:
            experiment_dir_name = save_dir_name + timestamp + "_split_" + str(data_split_idx)
        else:
            experiment_dir_name = save_dir_name + timestamp + "Small_Large"
        print(f"Current File Path: {Path(__file__).resolve()}")

        CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "modular.toml"
        root_path = Path(__file__).resolve().parent.parent.parent
        config = load_toml_config(CONFIG_PATH)

        # # current absolute path
        # if "__file__" in globals():
        #     here = os.path.dirname(os.path.abspath(__file__))
        # else:
        #     # fallback for notebooks or REPL
        #     here = os.getcwd()

        save_path = config["FFN"]["nn_result_path"]

        experiment_dir_path = os.path.join(root_path, save_path, experiment_dir_name)
        os.makedirs(experiment_dir_path)

        self.batch_size = hyperparam['batch_size']
        self.learning_rate = hyperparam['learning_rate']
        self.hidden_sizes = hyperparam['hidden_sizes']
        self.optimizer = hyperparam['optimizer']
        self.criterion = hyperparam['criterion']
        self.num_epochs = hyperparam['num_epochs']
        self.activation_function = hyperparam['activation_function']
        self.X_col_names = hyperparam['X_col_names']
        self.Y_col_names = hyperparam['Y_col_names']
        self.train_ratio = hyperparam['train_ratio']
        self.valid_test_ratio = hyperparam['valid_test_ratio']
        self.experiment_dir_path = experiment_dir_path
        #self.here = here
        self.hyperparam = hyperparam

        data = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_sizes': self.hidden_sizes,
            'optimizer': self.optimizer,
            'criterion': str(self.criterion.__class__.__name__),
            'num_epochs': self.num_epochs,
            'activation_function': str(self.activation_function.__class__.__name__),
            'X_col_names': self.X_col_names,
            'Y_col_names': self.Y_col_names,
            'train_ratio': self.train_ratio,
            'valid_test_ratio': self.valid_test_ratio}


        with open(os.path.join(experiment_dir_path, "hyperparams.csv"), "w") as f:
            json.dump(data, f, indent  = 4)

        if model is not None:
            model_path = os.path.join(experiment_dir_path,"model", "model.pt")
            if os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model, model_path)
            print(f"Model saved to {model_path}")
        ### end of for loop

    # the following is to return the current folder path
    def folder_path_for_current_hyperparams(self):
        return self.experiment_dir_path

    def save_model(self, model, model_name = "model", epoch_idx = None):
        """
        :param model: the model "xxx.pt" to save
        :param experiment_dir_name:
        :param model_name: the name of model to use, default is "model"
        :param epoch_idx: the index of epoch, default is None, this will be added to the model name if it is not None
        :return:
        """
        experiment_dir_path = os.path.join(self.experiment_dir_path, "model")
        # create a folder if it doesn't exist' (a dir for models)
        if not os.path.exists(experiment_dir_path):
            os.makedirs(experiment_dir_path)
        if epoch_idx is not None:
            model_name = model_name + "_" + str(epoch_idx)

        model_path = os.path.join(experiment_dir_path, model_name + ".pt")
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")
        return model_path

    def save_model_custom_path(self, model, path = "best_valid_models" , model_name = "model", epoch_idx = None):
        if path is None:
            raise ValueError("The path should not be None")
        if epoch_idx is not None:
            model_name = model_name + "_" + str(epoch_idx)
        model_path = os.path.join(self.experiment_dir_path, path, model_name + ".pt")
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")
        return model_path


    def save_accuracy_lists(self,
                            train_accuracy_list = None,
                            valid_accuracy_list = None,
                            accuracy_list = None,
                            bounded_accuracy_list = None,
                            accuracy_name = "accuracy"):
        csv_file_name = accuracy_name + ".csv"
        accuracy_path = os.path.join(self.experiment_dir_path, csv_file_name)
        l = len(accuracy_list)

        if l != self.num_epochs:
            raise ValueError("The length of accuracy list should be equal to the number of epochs")

        if train_accuracy_list is not None:
            self.train_accuracy_list = train_accuracy_list
        if valid_accuracy_list is not None:
            self.valid_accuracy_list = valid_accuracy_list
        if accuracy_list is not None:
            self.accuracy_list = accuracy_list
        if bounded_accuracy_list is not None:
            self.bounded_accuracy_list = bounded_accuracy_list

        data = {
            "Epoch": range(self.num_epochs),
            "Train accuracies": self.train_accuracy_list,
            "Valid accuracies": self.valid_accuracy_list,
            "Exact accuracies": self.accuracy_list,
            "Bounded accuracies": self.bounded_accuracy_list
        }

        print(
            f"length of accuracy list: {len(self.accuracy_list)}, number of epochs: {len(range(self.num_epochs))}, the length of bd acc list: {len(self.bounded_accuracy_list)}\n")

        df = pd.DataFrame(data)
        df.to_csv(accuracy_path, index=False)
        print(f"Accuracy list saved to {accuracy_path}")
        return accuracy_path

    def save_train_valid_losses(self, train_losses = None, val_losses = None, loss_file_name = "loss"):
        cvs_file_name = loss_file_name + ".csv"
        loss_path = os.path.join(self.experiment_dir_path, cvs_file_name)
        if train_losses is None:
            raise ValueError("The train losses should not be None")
        if val_losses is None:
            raise ValueError("The valid losses should not be None")
        l = len(train_losses)
        if l != self.num_epochs:
            raise ValueError("The length of train losses list should be equal to the number of epochs")
        data = {
            "Epoch" : range(self.num_epochs),
            "Train losses": train_losses,
            "Valid losses": val_losses
        }
        df = pd.DataFrame(data)
        df.to_csv(loss_path, index=False)

    def save_pictures(self, model = None, plot_name = None, train_losses = None, valid_losses = None):
        """
        :param model: default is None, this will be used to draw the graphs of the model
        :param plotter: the plotter to save, should be already set-up, then we save the plotted pictures to correct path
                        as above models/accuracies/hyperparams
        :param plot_name: the name of plot to use, default is None, this will be added to the plot name if it is not None
        :param train_losses: the train losses to use, default is None, this will be added to the plot name if it is not None
        :param valid_losses: the valid losses to use, default is None, this will be added to the plot name if it is not None

        :return: plot_path: the path of saved pictures, which is a folder named "plots" under the experiment folder path
        """

        self.train_losses = train_losses

        self.valid_losses = valid_losses

        plotter = ExperimentResultPlotter(
            model= model,
            hyperparams=self.hyperparam,
            train_losses=self.train_losses,
            valid_losses=self.valid_losses,
            test_accuracies=self.accuracy_list,
            test_bounds_accuracies=self.bounded_accuracy_list
        )

        plot_dir = os.path.join(self.experiment_dir_path, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        if plot_name is not None:
            plot_name = "plot"
        plotter.plot_all(save_dir=plot_dir, file_name=plot_name, show=True)
        print(f"Plots saved to {plot_dir}")
        return plotter, plot_dir


def test_Save_Class_1():
    Save_Class = Saving_Experiments()
    print(Save_Class.experiment_dir_path)

if __name__ == "__main__":
    test_Save_Class_1()