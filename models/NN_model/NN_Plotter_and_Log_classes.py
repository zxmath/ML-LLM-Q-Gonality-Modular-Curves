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








########### PLot Functions #############
# TODO: write a class for all necessary plots
class ExperimentResultPlotter:
    """
    A utility class for plotting neural network training and evaluation results
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 hyperparams: Optional[Dict[str, Any]] = None,
                 train_losses: Optional[List[float]] = None,
                 valid_losses: Optional[List[float]] = None,
                 test_losses: Optional[List[float]] = None,
                 train_accuracies: Optional[List[float]] = None,
                 valid_accuracies: Optional[List[float]] = None,
                 test_accuracies: Optional[List[float]] = None,
                 train_bounds_accuracies: Optional[List[float]] = None,
                 valid_bounds_accuracies: Optional[List[float]] = None,
                 test_bounds_accuracies: Optional[List[float]] = None,
                 train_samples: Optional[np.ndarray] = None,
                 valid_samples: Optional[np.ndarray] = None,
                 test_samples: Optional[np.ndarray] = None,
                 train_bd_samples: Optional[np.ndarray] = None,
                 valid_bd_samples: Optional[np.ndarray] = None,
                 test_bd_samples: Optional[np.ndarray] = None
                 ) -> None:
        """
        Initialize the plotter with experiment data
        """
        print(f"init Plotter")
        self.model = model
        self.hyperparams = hyperparams
        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.test_losses = test_losses
        self.train_accuracies = train_accuracies
        self.valid_accuracies = valid_accuracies
        self.test_accuracies = test_accuracies
        self.train_bounds_accuracies = train_bounds_accuracies
        self.valid_bounds_accuracies = valid_bounds_accuracies
        self.test_bounds_accuracies = test_bounds_accuracies
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.test_samples = test_samples
        self.train_bd_samples = train_bd_samples
        self.valid_bd_samples = valid_bd_samples
        self.test_bd_samples = test_bd_samples

    def plot_losses(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training and validation losses over epochs"""
        print(f"Plotting losses")
        if not (self.train_losses or self.valid_losses):
            print("No loss data available for plotting")
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1) if self.train_losses else []

        if self.train_losses:
            plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if self.valid_losses:
            plt.plot(epochs, self.valid_losses, 'r-', label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_accuracies(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training and validation accuracies over epochs"""
        print(f"Plotting accuracies")
        if not (self.train_accuracies or self.valid_accuracies or self.test_accuracies):
            print("No accuracy data available for plotting")
            return

        plt.figure(figsize=(10, 6))

        if self.train_accuracies:
            epochs = range(1, len(self.train_accuracies) + 1)
            plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')

        if self.valid_accuracies:
            epochs = range(1, len(self.valid_accuracies) + 1)
            plt.plot(epochs, self.valid_accuracies, 'r-', label='Validation Accuracy')

        if self.test_accuracies:
            epochs = range(1, len(self.test_accuracies) + 1)
            plt.plot(epochs, self.test_accuracies, 'g-', label='Test Accuracy')

        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_bounds_accuracies(self, save_path: Optional[str] = None, show: bool = True):
        """Plot bound prediction accuracies for training, validation, and test sets"""
        print(f"Plotting bounds accuracies")
        if not (self.train_bounds_accuracies or self.valid_bounds_accuracies or self.test_bounds_accuracies):
            print("No bounds accuracy data available for plotting")
            return

        plt.figure(figsize=(10, 6))

        if self.train_bounds_accuracies:
            epochs = range(1, len(self.train_bounds_accuracies) + 1)
            plt.plot(epochs, self.train_bounds_accuracies, 'b-', label='Training Bounds Accuracy')

        if self.valid_bounds_accuracies:
            epochs = range(1, len(self.valid_bounds_accuracies) + 1)
            plt.plot(epochs, self.valid_bounds_accuracies, 'r-', label='Validation Bounds Accuracy')

        if self.test_bounds_accuracies:
            epochs = range(1, len(self.test_bounds_accuracies) + 1)
            plt.plot(epochs, self.test_bounds_accuracies, 'g-', label='Test Bounds Accuracy')

        plt.title('Bounds Prediction Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_predictions(self, save_path: Optional[str] = None, show: bool = True):
        """Plot actual vs predicted values for a sample of data points"""
        # This is a placeholder for implementing prediction visualization
        pass

    def plot_model_parameters(self, model, save_path: Optional[str] = None, show: bool = True):
        """
        Visualize the weight matrices of the neural network model

        Args:
            model: The PyTorch neural network model
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        print(f"Plotting model parameters")
        if model is None:
            print("No model provided for parameter visualization")
            return

        plt.figure(figsize=(15, 10))

        # Create subplots based on number of layers
        num_layers = len([m for m in model.modules() if isinstance(m, torch.nn.Linear)])
        if num_layers == 0:
            print("No linear layers found in the model")
            return

        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))
        if num_layers == 1:
            axes = [axes]  # Make it iterable for single layer

        # Plot each layer's weight matrix
        layer_idx = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                ax = axes[layer_idx]
                weights = param.data.cpu().numpy()

                # Create heatmap
                im = ax.imshow(weights, cmap='viridis')
                ax.set_title(f'Layer {layer_idx + 1} Weights\n{weights.shape[0]}×{weights.shape[1]}')

                # Add colorbar
                plt.colorbar(im, ax=ax)

                # Only set x and y ticks if the matrix is small enough
                if weights.shape[0] < 20 and weights.shape[1] < 20:
                    ax.set_xticks(np.arange(weights.shape[1]))
                    ax.set_yticks(np.arange(weights.shape[0]))
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])

                layer_idx += 1

        plt.tight_layout()
        plt.suptitle("Neural Network Weight Matrices", fontsize=16)
        plt.subplots_adjust(top=0.88)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all(self, save_dir, file_name="xxx", show: bool = True):
        """Plot all available metrics"""

        # Plot losses
        loss_path = os.path.join(save_dir, file_name + 'losses.png') if save_dir else None
        self.plot_losses(save_path=loss_path, show=show)

        # Plot accuracies
        acc_path = os.path.join(save_dir, file_name + 'accuracies.png') if save_dir else None
        self.plot_accuracies(save_path=acc_path, show=show)

        # Plot bounds accuracies
        bounds_path = os.path.join(save_dir, file_name + 'bounds_accuracies.png') if save_dir else None
        self.plot_bounds_accuracies(save_path=bounds_path, show=show)

        # Plot model
        model_plot_path = os.path.join(save_dir, file_name + 'model_parameters.png') if save_dir else None
        self.plot_model_parameters(save_path=model_plot_path, model=self.model, show=show)

