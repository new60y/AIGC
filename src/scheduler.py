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