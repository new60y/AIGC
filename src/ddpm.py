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

from unet import Unet
from scheduler import linear_beta_schedule
from utils import set_device,set_seed,save_checkpoint,load_checkpoint
from load_data import load_data

device=set_device()

set_seed()

def q_sample(x,hat_alphas,timesteps):
    t=torch.randint(1,timesteps+1,(x.shape[0],))
    hat_alpha=hat_alphas[t-1][:,None,None,None]
    noise=torch.randn_like(x)
    x_t=torch.sqrt(hat_alpha)*x+torch.sqrt(1.-hat_alpha)*noise
    return noise,x_t,t

def p_sample(model,x,t,betas,alphas,hat_alphas):
    t=torch.tensor([t])
    z=torch.randn_like(x)
    hat_alpha=hat_alphas[t-1]
    alpha=alphas[t-1]
    beta=betas[t-1]
    pred_noise=model(x,t).to("cpu")
    return 1/torch.sqrt(alpha)*(x-(1-alpha)/torch.sqrt(1-hat_alpha)*pred_noise)+torch.sqrt(beta)*z

def train(train_set,model,optimizer,criterion,batch_size,epochs,lr,timesteps,schedule,model_path=None):
    train_loader=DataLoader(train_set,batch_size,True)

    model.train()

    #print(schedule)

    betas,alphas,hat_alphas=schedule(timesteps)

    for epoch in range(epochs):
        loss_sum=0
        print("epoch",epoch)
        for img,label in tqdm(train_loader):
            #img=img.to(device)
            noise,noisy_img,t=q_sample(img,hat_alphas,timesteps)
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

def infer(model,timesteps,schedule,sample_num):
    betas,alphas,hat_alphas=schedule(timesteps)

    model.eval()

    with torch.no_grad():
        for i in range(sample_num):
            x=torch.randn((1,1,32,32))
            for t in range(timesteps,0,-1):
                x=p_sample(model,x,t,betas,alphas,hat_alphas)
            #print(x)
            plt.imshow(x.squeeze().cpu(),cmap="gray")
            plt.show()

def ddim_infer(model,timesteps,schedule,sample_num,ddim_steps=10):
    betas,alphas,hat_alphas=schedule(timesteps)

    model.eval()

    #ts=torch.tensor(range(timesteps,0,-1))

    ts=torch.linspace(1,timesteps,ddim_steps).to(torch.long)

    print(ts)

    with torch.no_grad():
        for i in range(sample_num):
            x=torch.randn((1,1,32,32))
            for t in range(ddim_steps-1,0,-1):
                #x=ddim_p_sample(model,x,t,betas,alphas,hat_alphas)
                t2=ts[t]
                t1=ts[t-1]
                #print(t,t-1,t2,t1)
                z=torch.randn_like(x)
                hat_alpha2=hat_alphas[t2-1]
                hat_alpha1=hat_alphas[t1-1]
                #alpha=alphas[t-1]
                beta=betas[t2-1]
                #var=torch.sqrt(beta)
                var=0
                pred_noise=model(x,t2.unsqueeze(dim=0)).to("cpu")
                #x=1/torch.sqrt(alpha)*(x-(1-alpha)/torch.sqrt(1-hat_alpha)*pred_noise)
                x=torch.sqrt(hat_alpha1)/torch.sqrt(hat_alpha2)*(x-torch.sqrt(1-hat_alpha2)*pred_noise)+torch.sqrt(1-hat_alpha1-var**2)*pred_noise+var*z

            #print(x)
            plt.imshow(x.squeeze().cpu(),cmap="gray")
            plt.show()

if __name__=="__main__":
    '''
    transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2*(x-0.5)),
    ]
    )
    train_set=torchvision.datasets.MNIST(root="../data/", download=True, transform=transform)
    '''
    train_set=load_data("MNIST")
    batch_size=128
    train_loader=DataLoader(train_set,batch_size,True)

    model=Unet(1)
    #print(model)
    model.to(device)
    lr=1e-3
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    #criterion=nn.CrossEntropyLoss()
    criterion=nn.MSELoss(reduction="sum")

    epochs=10

    timesteps=300
    
    model_path="../checkpoint/ddpm_unet.pth"

    train(train_loader,model,optimizer,criterion,epochs,timesteps,model_path)

    infer(model)

    
