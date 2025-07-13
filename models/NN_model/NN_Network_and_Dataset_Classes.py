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


class InMemoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]




"""
The follwowing is for Y in the format bound [a, b], a!=b 
"""
class IntervalDataset(Dataset):
    def __init__(self, X, a, b):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.a = torch.tensor(a.values, dtype=torch.float32)
        self.b = torch.tensor(b.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.a[idx], self.b[idx]


"""
Interval Loss function class
"""
class interval_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y_pred, a, b):
        """
        :param Y_pred: model output
        :param a: lower bound
        :param b: upper bound
        :return:
        """
        lower_violation = torch.relu(a-Y_pred)
        upper_violation = torch.relu(Y_pred-b)
        return torch.mean(lower_violation**2 + upper_violation**2)



"""
The following is an implementation of classical forwarding neural network
"""


class BasicNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function=nn.ReLU,
                 apply_final_activation=False):
        """
        Initialize the neural network.
        Args:
            input_size (int): Number of input features
            hidden_sizes (list of int): List defining the number of neurons in each hidden layer
            output_size (int): Number of output units
            activation_function (nn.Module): Activation function to use in hidden layers, default is ReLU, but we can also use Tanh or Sigmoid, etc.
                                            For our goal, I feel that sigmoid is not a bad choice
            apply_final_activation (bool): Whether to apply activation function to the final output layer, if false, apply ReLU after each layer
        """
        super(BasicNeuralNetwork, self).__init__()
        # initiate the activation nonlinear function
        self.activation_function = activation_function
        # The list of size of each layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create layers with the sizes specified by layer_sizes
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(self.activation_function)
            # self.layers.append(nn.Dropout(p=0.1))

        # Note: in above, the output of the current latyer should be the same as the input size for the next layer

    def forward(self, x):
        """
        here layer below record the matrix/functions in each layer
        x = layer(x) actually means Y = NN_layer_i(X), where NN_layer_i is the ith layer of our network
        basically input x, the return is the output after the data go through the current network
        :param x: input data
        :return: output data
        """
        for i, layer in enumerate(self.layers):
            # apply the current ith layer on the input and get the output as the input of the next layer
            x = layer(x)
            # if i < len(self.layers) - 1:
            #     x = self.activation_function(x)
        return x

    def evaluate(self, data, criterion=nn.MSELoss, device=torch.device('cpu'), mode="train"):
        """
            FORWARD pass of the neural network.
            Args:
                data (Tensor): Dataloader
                criterion (nn.Module): Loss function to optimize.
                device (torch.device): Device to use for computation. Default: CPU.
                mode: "train" or "valid" or "test"
            Returns:
                Tensor: Output prediction of the network
        """
        # Set the model to evaluation mode
        # There are some T/F features of the class of nn.model,
        # here set the one for evaluation to True; set the one for train to False
        self.eval()
        # init an empty list for the predict result
        # predictions = []
        totalLoss = 0.0

        # disables gradient tracking, save memory, faster, avoid compute gradient automatically
        with torch.no_grad():
            if mode in ["train", "valid"]:
                # data = data.shuffle(0)
                for batchInputs, batchOuputs in data:  # Only inputs are needed for prediction
                    batchInputs = batchInputs.to(device,
                                                 non_blocking=True)  # Move the tensor X_batch to a specific device (e.g., CPU or GPU)
                    batchOuputs = batchOuputs.to(device, non_blocking=True).view(-1, 1)
                    batchPredictions = self.forward(batchInputs)  # compute the batch prediction

                    # add the result of this batch to the result list
                    loss = criterion(batchPredictions, batchOuputs)  # compute the loss for this batch
                    totalLoss += loss.item()  # loss is a more complicated structure, take loss.item() get a usual float
                return totalLoss / len(data)
            elif mode == "test":
                total_in_range = 0
                total_samples = 0
                for batchInputs, bounds in data:  # bounds is a tensor of shape [batch_size, 2]
                    batchInputs = batchInputs.to(device, non_blocking=True)
                    bounds = bounds.to(device, non_blocking=True)  # shape: [batch_size, 2]

                    predictions = self.forward(batchInputs).view(-1)  # shape: [batch_size]
                    lower = bounds[:, 0]
                    upper = bounds[:, 1]

                    in_range_mask = ((predictions >= lower) & (predictions <= upper)).float()
                    total_in_range += in_range_mask.sum().item()
                    total_samples += predictions.numel()

                percent_in_range = (total_in_range / total_samples) * 100 if total_samples > 0 else 0.0
                print(f"Test Accuracy: {percent_in_range:.2f}% of predictions fall within [a, b]")
                return percent_in_range
            print(
                f"We cannot return the loss in the mode you choose. Please choose mode to be either 'train', 'valid' or 'test'.")
            return None

    ##### We define another variation of evaluation function, for weakly supervised learning
    def evaluate_interval(self, data, criterion = interval_loss, device=torch.device('cpu'), mode="train"):
        # Set the model to evaluation mode
        # There are some T/F features of the class of nn.model,
        # here set the one for evaluation to True; set the one for train to False
        self.eval()
        # init an empty list for the predict result
        # predictions = []
        totalLoss = 0.0

        # disables gradient tracking, save memory, faster, avoid compute gradient automatically
        with torch.no_grad():
            for batchInputs, batchOuputs_a, batchOuputs_b in data:  # Only inputs are needed for prediction
                batchInputs = batchInputs.to(device,
                                             non_blocking=True)  # Move the tensor X_batch to a specific device (e.g., CPU or GPU)
                batchOuputs_a = batchOuputs_a.to(device, non_blocking=True).view(-1, 1)
                batchOuputs_b = batchOuputs_b.to(device, non_blocking=True).view(-1, 1)
                batchPredictions = self.forward(batchInputs)  # compute the batch prediction

                # add the result of this batch to the result list
                loss_func = criterion()
                loss = loss_func.forward(batchPredictions, batchOuputs_a, batchOuputs_b)  # compute the loss for this batch
                totalLoss += loss.item()  # loss is a more complicated structure, take loss.item() get a usual float
            return totalLoss / len(data)
        return None


    def train_one_epoch(self, DataLoader, criterion, optimizer, device):
        """
            Train the model for one epoch using batch data.
            Args:chat
                DataLoader: PyTorch DataLoader for the training data.
                criterion: The loss function to optimize.
                optimizer: The optimizer to update weights.
                device: The device to use for computation.
            Returns:
                float: The average loss for this training epoch.
        """

        # Set the model to training (T/F setting), again, some Boolean features in the class
        self.train()
        epoch_loss = 0.0

        for batch_X, batch_Y in DataLoader:
            batch_X, batch_Y = batch_X.to(device, non_blocking=True), batch_Y.to(device, non_blocking=True).view(-1,
                                                                                                                 1)  # Move the tensor X_batch to a specific device (e.g., CPU or GPU)
            # Forward pass
            # predictions = self.forward(batch_X)
            predictions = self.forward(batch_X)
            loss = criterion(predictions, batch_Y)
            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients, otherwise accumulative
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            epoch_loss += loss.item()
        # Concatenate all batch predictions
        return epoch_loss / len(DataLoader)


    def train_one_epoch_interval(self, DataLoader, optimizer, device, criterion = interval_loss):
        """
            Train the model for one epoch using batch data. This train version is for weakly supervised learning
            Args:chat
                DataLoader: PyTorch DataLoader for the training data.
                criterion: The loss function to optimize.
                optimizer: The optimizer to update weights.
                device: The device to use for computation.
            Returns:
                float: The average loss for this training epoch.
        """

        # Set the model to training (T/F setting), again, some Boolean features in the class
        self.train()
        epoch_loss = 0.0

        for batch_X, batch_Y_a, batch_Y_b in DataLoader:
            batch_X = batch_X.to(device, non_blocking=True)  # Move the tensor X_batch to a specific device (e.g., CPU or GPU)
            batch_Y_a = batch_Y_a.to(device, non_blocking=True).view(-1, 1)
            batch_Y_b = batch_Y_b.to(device, non_blocking=True).view(-1, 1)
            # print(f"Data Shape Check 1: X shape: {batch_X.shape}, Y_a shape: {batch_Y_a.shape}, Y_b shape: {batch_Y_b.shape}")
            # Forward pass
            # predictions = self.forward(batch_X)
            predictions = self.forward(batch_X)
            loss_func = criterion()
            loss = loss_func.forward(Y_pred = predictions, a = batch_Y_a, b = batch_Y_b)
            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients, otherwise accumulative
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            epoch_loss += loss.item()
        # Concatenate all batch predictions
        return epoch_loss / len(DataLoader)

