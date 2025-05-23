import sys

sys.path.append("..")


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aigc.model.mlp import Generator
from aigc.model.mlp_discriminator import Discriminator
from aigc.model.unet import Unet
from utils import load_checkpoint, save_checkpoint, set_device, set_seed

device = set_device()

set_seed()


def train(train_loader, batch_size, epochs, lr, config, model_path=None):
    noise_dim = 100
    img_channel = config.get("channel", 1)
    sample_num = config.get("sample_num", 3)
    # G = Generator().to(device)
    # print(G)
    G = config["gen"].to(device)
    # print(G)
    # D = Discriminator().to(device)
    # print(D)
    D = config["disc"].to(device)
    # print(D)
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        loss_sum = 0
        batch_count = 0
        for real_imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            z = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = G(z)
            D_real = D(real_imgs)
            D_fake = D(fake_imgs.detach())
            loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            z = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = G(z)
            D_fake = D(fake_imgs)
            loss_G = criterion(D_fake, real_labels)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            loss_sum += (loss_D + loss_G).item()
            batch_count += 1

        print(loss_sum / batch_count)

        if model_path:
            save_checkpoint(G, model_path + "_G.pth")
            save_checkpoint(D, model_path + "_D.pth")

    G.eval()
    with torch.no_grad():
        z = torch.randn(sample_num, noise_dim, device=device)
        samples = G(z)
        for i in range(sample_num):
            plt.imshow(samples[i].cpu().squeeze(), cmap="gray")
            plt.title(f"Sample {i + 1} After Training")
            plt.show()
    G.train()
