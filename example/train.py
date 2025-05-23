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

from aigc.model.mlp import Generator
from aigc.model.mlp_discriminator import Discriminator

# from aigc.load_data import load_data
from aigc.model.unet import Unet
from aigc.schedulers.ddpm import DDPMScheduler
from aigc.trainer.train import train
from utils import load_checkpoint, save_checkpoint, set_device, set_seed

device = set_device()

set_seed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag", default="ddpm")
    parser.add_argument("--gen", default="unet")
    parser.add_argument("--data", default="MNIST")
    # parser.add_argument("--ckpt", default="../checkpoint/ddpm_unet.pth")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--ep", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--tm", default=200)
    # parser.add_argument("--sc",default="linear")
    args = parser.parse_args()

    # train_set=load_data(args.data)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: 2 * (x - 0.5)),
        ]
    )
    train_set = torchvision.datasets.MNIST(
        root=os.path.join("../data"),
        download=True,
        transform=transform,
    )
    # print(train_set[0])
    train_loader = DataLoader(train_set, args.bs, True)

    config = {}  # config里设置模型，可以用框架实现的，也可以自定义
    config["timesteps"] = 200
    config["sample_num"] = 3
    config["channel"] = 1
    if args.gen == "unet":
        config["gen"] = Unet(config["channel"])
    if args.ag == "gan":
        config["gen"] = Generator()
        config["disc"] = Discriminator()

    """
    if args.ag == "ddpm":
        train = ddpm_train
        model_path = "../checkpoint/ddpm_unet.pth"
    elif args.ag == "flow":
        train = flow_train
        model_path = "../checkpoint/flow_unet.pth"
    """
    # if args.net == "unet":
    #    model = Unet(1).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # criterion = nn.MSELoss(reduction="sum")
    train(args.ag, train_loader, args.bs, args.ep, args.lr, config)

    # load_checkpoint(args.ckpt)
    # infer(3, args.ckpt)
