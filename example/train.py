import sys
sys.path.append("..")

from aigc.schedulers.ddpm import DDPMScheduler
from utils import set_device,set_seed,save_checkpoint,load_checkpoint
#from aigc.load_data import load_data
from aigc.model.unet import Unet

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

device=set_device()

set_seed()

def train(train_set,model,optimizer,criterion,batch_size,epochs,lr,scheduler,model_path=None):
    train_loader=DataLoader(train_set,batch_size,True)

    model.train()

    #print(schedule)

    #betas,alphas,hat_alphas=schedule(timesteps)

    #scheduler=DDPMScheduler()

    timesteps=scheduler.timesteps
    print(timesteps)

    for epoch in range(epochs):
        loss_sum=0
        print("epoch",epoch)
        for img,label in tqdm(train_loader):
            #img=img.to(device)
            #noise,noisy_img,t=q_sample(img,hat_alphas,timesteps)
            noise=torch.randn_like(img)
            t=torch.randint(0,timesteps,(img.shape[0],))
            noisy_img=scheduler.add_noise(img,noise,t)
            #print(noisy_img.device,t.device)
            prd_noise=model(noisy_img,t).to("cpu")
            loss=criterion(prd_noise,noise)
            loss_sum+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model_path:
                save_checkpoint(model,model_path)
            #break
        print(loss_sum/len(train_set))

def infer(model,scheduler,sample_num):
    #betas,alphas,hat_alphas=schedule(timesteps)

    model.eval()

    timesteps=scheduler.timesteps
    print(timesteps)

    with torch.no_grad():
        for i in range(sample_num):
            x=torch.randn((1,1,32,32))
            for t in range(timesteps-1,-1,-1):
                #x=p_sample(model,x,t,betas,alphas,hat_alphas)
                t=torch.tensor([t])
                #print(t)
                noise=model(x,t).to("cpu")
                x=scheduler.step(noise,t,x)
            #print(x)
            plt.imshow(x.squeeze().cpu(),cmap="gray")
            plt.show()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ag",default="ddpm")
    parser.add_argument("--net",default="unet")
    parser.add_argument("--data",default="MNIST")
    parser.add_argument("--ckpt",default="../checkpoint/ddpm_unet.pth")
    parser.add_argument("--bs",default=128)
    parser.add_argument("--ep",default=10)
    parser.add_argument("--lr",default=1e-3)
    parser.add_argument("--tm",default=300)
    #parser.add_argument("--sc",default="linear")
    args=parser.parse_args()

    #train_set=load_data(args.data)
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

    if args.ag=="ddpm":
        scheduler=DDPMScheduler(timesteps=args.tm)
    if args.net=="unet":
        model=Unet(1).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    criterion=nn.MSELoss(reduction="sum")
    train(train_set,model,optimizer,criterion,args.bs,args.ep,args.lr,scheduler,args.ckpt)

    load_checkpoint(model,args.ckpt)
    infer(model,scheduler,3)