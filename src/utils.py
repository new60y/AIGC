import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision

import transformers
from transformers import AutoTokenizer,AutoModel,AutoConfig

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import os
import json
import random
import math

def set_device():
    device="cuda" if torch.cuda.is_available() else "cpu"
    return device

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model,path):
    torch.save(model.state_dict(),path)

def load_checkpoint(model,path):
    model.load_state_dict(torch.load(path))