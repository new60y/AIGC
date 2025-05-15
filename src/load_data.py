import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision

import transformers

import os

def load_data(dataset_name="MNIST"):
    if dataset_name:
        if dataset_name=="MNIST":
            transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: 2*(x-0.5)),
            ]
            )
            train_set=torchvision.datasets.MNIST(
                root=os.path.join("../data"), 
                download=True, 
                transform=transform,
            )
    
    return train_set
