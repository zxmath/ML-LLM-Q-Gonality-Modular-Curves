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



def device_verification():
    ######################### Prepare the devices, paths, hyper-params ##########################
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / 1e6:.2f} MB")
            print(f"  - Memory Cached: {torch.cuda.memory_reserved(i) / 1e6:.2f} MB")
            print(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("No GPU available!")

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    return device


def create_directory(path: str or Path) -> bool:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}", file=sys.stderr)
        return False