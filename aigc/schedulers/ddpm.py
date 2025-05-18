import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision

def linear_beta_schedule(timesteps):
    beta_min=1e-4
    beta_max=2e-2
    betas=torch.linspace(beta_min,beta_max,timesteps)
    alphas=1.-betas
    hat_alphas=torch.cumprod(alphas,dim=0)
    return betas,alphas,hat_alphas

class DDPMScheduler():
    def __init__(self,
        timesteps=200,
        beta_min=1e-4,
        beta_max=2e-2,

    ):
        self.timesteps=timesteps
        self.beta_min=beta_min
        self.beta_max=beta_max
        self.betas=torch.linspace(beta_min,beta_max,timesteps)
        self.alphas=1.-self.betas
        self.cumprod_alphas=torch.cumprod(self.alphas,dim=0)
        
    def add_noise(self,x,noise,t):
        #t=torch.randint(0,self.timesteps,(x.shape[0],))
        cumprod_alpha=self.cumprod_alphas[t][:,None,None,None]
        #noise=torch.randn_like(x)
        x_t=torch.sqrt(cumprod_alpha)*x+torch.sqrt(1.-cumprod_alpha)*noise
        return x_t

    def step(self,noise,t,x):
        #t=torch.tensor(t).reshape(1,-1)
        z=torch.randn_like(x)
        cumprod_alpha=self.cumprod_alphas[t]
        alpha=self.alphas[t]
        beta=self.betas[t]
        return 1/torch.sqrt(alpha)*(x-(1-alpha)/torch.sqrt(1-cumprod_alpha)*noise)+torch.sqrt(beta)*z
    
    
