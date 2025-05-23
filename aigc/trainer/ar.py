import sys

sys.path.append("..")


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from aigc.load_data import load_data
from aigc.model.unet import Unet
from utils import load_checkpoint, save_checkpoint, set_device, set_seed

device = set_device()

set_seed()


def train(train_loader, batch_size, epochs, lr, model_path=None):
    pass
