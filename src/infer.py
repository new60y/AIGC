import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import os
import json
import random
import math
import argparse

import ddpm
from unet import Unet
from scheduler import linear_beta_schedule
from utils import set_device,set_seed,save_checkpoint,load_checkpoint
from load_data import load_data

device=set_device()

set_seed()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ag",default="ddpm")
    parser.add_argument("--net",default="unet")
    #parser.add_argument("--data",default="MNIST")
    parser.add_argument("--ckpt",default="../checkpoint/ddpm_unet.pth")
    #parser.add_argument("--bs",default=128)
    #parser.add_argument("--ep",default=10)
    #parser.add_argument("--lr",default=1e-3)
    parser.add_argument("--tm",default=200)
    parser.add_argument("--sc",default="linear")
    parser.add_argument("--sn",default=3)
    args=parser.parse_args()

    if args.ag=="ddpm":
        infer=ddpm.infer
    if args.net=="unet":
        model=Unet(1).to(device)
    if args.ckpt:
        load_checkpoint(model,args.ckpt)
    if args.sc=="linear":
        schedule=linear_beta_schedule
        
    infer(model,args.tm,schedule,args.sn)
    
    
