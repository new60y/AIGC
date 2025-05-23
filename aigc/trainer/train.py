import sys

sys.path.append("..")

import argparse
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from aigc.load_data import load_data
from aigc.model.unet import Unet
from aigc.schedulers.ddpm import DDPMScheduler
from aigc.trainer.ar import train as ar_train
from aigc.trainer.ddpm import train as ddpm_train
from aigc.trainer.flow import train as flow_train
from aigc.trainer.gan import train as gan_train
from utils import load_checkpoint, save_checkpoint, set_device, set_seed

device = set_device()

set_seed()


def train(algorithm, train_loader, batch_size, epochs, lr, config):
    if algorithm == "ddpm":
        train = ddpm_train
        model_path = "../checkpoint/ddpm_unet.pth"
    elif algorithm == "flow":
        train = flow_train
        model_path = "../checkpoint/flow_unet.pth"
    elif algorithm == "gan":
        train = gan_train
        model_path = "../checkpoint/gan.pth"
    elif algorithm == "ar":
        train = ar_train
        model_path = "../checkpoint/ar.pth"
    train(train_loader, batch_size, epochs, lr, config, model_path)
