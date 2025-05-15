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
from abc import abstractmethod

#model_path="microsoft/deberta-v3-small"
model_path="prajjwal1/bert-small"

seed=42
random.seed(seed)
np.random.seed(seed)
torch.random.seed()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device="cuda" if torch.cuda.is_available() else "cpu"

class Down_sample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,3,1,1)
        self.conv2=nn.Conv2d(out_channel,out_channel,3,1,1)
        self.conv3=nn.Conv2d(out_channel,out_channel,3,1,1)
        self.pool=nn.MaxPool2d(2)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm2d(out_channel)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.bn3=nn.BatchNorm2d(out_channel)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        res=x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x+=res
        x=self.relu(x)
        '''
        x=self.conv3(x)
        x=self.bn3(x)
        x+=res
        x=self.relu(x)
        '''
        return x,self.pool(x)

class Up_sample(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channel,in_channel,2,2)
        self.conv1=nn.Conv2d(in_channel,in_channel,3,1,1)
        self.conv2=nn.Conv2d(in_channel,in_channel,3,1,1)
        self.conv3=nn.Conv2d(in_channel,out_channel,3,1,1)
        self.relu=nn.ReLU()
        #self.pool=nn.MaxPool2d(2)
        self.bn1=nn.BatchNorm2d(in_channel)
        self.bn2=nn.BatchNorm2d(in_channel)
        self.bn3=nn.BatchNorm2d(out_channel)

    def forward(self,skip,x):
        res=x=skip+self.up(x)
        '''
        x=self.conv1(x)
        x=self.bn1(x)
        x+=res
        res=x=self.relu(x)
        '''
        x=self.conv2(x)
        x=self.bn2(x)
        x+=res
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self,img_channel,hidden_channels=[16,32]):
        super().__init__()
        self.img_channel=img_channel
        self.hidden_channels=hidden_channels
        
        self.down=nn.ModuleList()
        self.down.append(Down_sample(img_channel,hidden_channels[0]))
        for i in range(1,len(hidden_channels)):
            in_channel=hidden_channels[i-1]
            out_channel=hidden_channels[i]
            #print("down:",in_channel,out_channel)
            self.down.append(Down_sample(in_channel,out_channel))

        self.up=nn.ModuleList()
        for i in range(len(hidden_channels)-1,0,-1):
            in_channel=hidden_channels[i]
            out_channel=hidden_channels[i-1]
            #print("up:",in_channel,out_channel)
            self.up.append(Up_sample(in_channel,out_channel))
        self.up.append(Up_sample(hidden_channels[0],img_channel))

        self.mid_conv=nn.Conv2d(hidden_channels[-1],hidden_channels[-1],3,1,1)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm2d(hidden_channels[-1])

        self.fc=nn.Linear(16,1)
        self.last_conv=nn.Conv2d(img_channel,img_channel,3,1,1)

    def time_embedding(self,t):
        t=torch.tensor(t).to(device)
        #dim=self.hidden_channels[0]
        dim=16
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)

        embed=torch.cat([sin_emb, cos_emb], dim=-1)
        #print("embed:",sin_emb.shape,cos_emb.shape,embed.shape)
        return embed

    def forward(self,x,t):
        x=x.to(device)
        #x=torch.tensor(x,dtype=torch.float32)
        t_embed=self.time_embedding(t)
        #print(x.shape,t_embed.shape)
        x=x+self.fc(t_embed)[:,:,None,None]
        skips=[]
        for conv in self.down:
            skip,x=conv(x)
            #print("down:",x.shape,skip.shape)
            skips.append(skip)

        x=self.mid_conv(x)
        x=self.bn1(x)
        x=self.relu(x)

        skips=skips[::-1]

        '''
        for skip in skips:
            print("skip:",skip.shape)
        '''

        for i in range(len(skips)):
            #print("up:",i)
            skip=skips[i]
            #print(skip.shape,x.shape)
            conv=self.up[i]
            #print(conv.up)
            x=conv(skip,x)
        
        x=self.last_conv(x)

        return x

def linear_beta_schedule(timesteps):
    beta_min=1e-4
    beta_max=2e-2
    return torch.linspace(beta_min,beta_max,timesteps)

def diffusion_forward(img,t,hat_alphas):
    hat_alpha=hat_alphas[t]
    noise=torch.randn_like(img)
    noisy_img=torch.sqrt(hat_alpha)*img+torch.sqrt(1.-hat_alpha)*noise
    return noisy_img

if __name__=="__main__":
    transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2*(x-0.5)),
    ]
    )
    train_set=torchvision.datasets.MNIST(root="./data/", download=True, transform=transform)
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
    
    betas=linear_beta_schedule(timesteps).to(device)
    print(len(betas))
    alphas=1.-betas
    #print(alphas)
    hat_alphas=torch.cumprod(alphas,dim=0).to(device)
    print(hat_alphas)


    for epoch in range(epochs):
        loss_sum=0
        print("epoch",epoch)
        for img,label in tqdm(train_loader):
            #plt.imshow(img.squeeze(),cmap="gray")
            #plt.show()
            img=img.to(device)
            #img=img.reshape(-1,784)
            t=torch.randint(1,timesteps+1,(img.shape[0],)).to(device)
            hat_alpha=hat_alphas[t-1][:,None,None,None]
            #print(t,hat_alpha)
            noise=torch.randn_like(img).to(device)
            #print(img.shape,noise.shape,hat_alpha.shape)
            noisy_img=torch.sqrt(hat_alpha)*img+torch.sqrt(1.-hat_alpha)*noise
            #print(hat_alpha.shape,img.shape,noisy_img.shape)
            '''
            print(t)
            plt.imshow(noisy_img.squeeze().cpu(),cmap="gray")
            plt.show()
            '''
            prd_noise=model(noisy_img,t)
            loss=criterion(prd_noise,noise)
            loss_sum+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #break
        print(loss_sum/len(train_set))

    model.eval()
    with torch.no_grad():
        x=torch.randn((1,1,32,32)).to(device)
        #x=x.reshape(784)
        #x=x.unsqueeze(dim=0)
        #plt.imshow(x[0].cpu(),cmap="gray")
        #plt.show()
        #print(x)
        for t in range(timesteps,0,-1):
            t=torch.tensor([t]).to(device)
            #print(t)
            z=torch.randn_like(x).to(device)
            #print(hat_alphas,t)
            hat_alpha=hat_alphas[t-1]
            alpha=alphas[t-1]
            pred_noise=model(x,t)
            #print(pred_noise)
            #break
            #print(t,hat_alpha,alpha)
            x=1/torch.sqrt(alpha)*(x-(1-alpha)/torch.sqrt(1-hat_alpha)*pred_noise)
        #print(x)
        plt.imshow(x.squeeze().cpu(),cmap="gray")
        plt.show()

    
