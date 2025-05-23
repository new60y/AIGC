import sys

sys.path.append("..")


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from aigc.load_data import load_data
from aigc.model.unet import Unet
from aigc.schedulers.ddpm import DDPMScheduler
from utils import load_checkpoint, save_checkpoint, set_device, set_seed

device = set_device()

set_seed()


def train(train_loader, batch_size, epochs, lr, config, model_path=None):
    # train_loader = DataLoader(train_set, batch_size, True)

    # print(next(iter(train_loader))[0].shape)

    # model = Unet(config["channel"]).to(device)
    model = config["gen"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    # print(schedule)

    # betas,alphas,hat_alphas=schedule(timesteps)

    scheduler = DDPMScheduler(config["timesteps"])

    timesteps = scheduler.timesteps
    print(timesteps)

    for epoch in range(epochs):
        loss_sum = 0
        print("epoch", epoch)
        for img, label in tqdm(train_loader):
            # img=img.to(device)
            # noise,noisy_img,t=q_sample(img,hat_alphas,timesteps)
            noise = torch.randn_like(img)
            t = torch.randint(0, timesteps, (img.shape[0],))
            noisy_img = scheduler.add_noise(img, noise, t)
            # print(noisy_img.device,t.device)
            prd_noise = model(noisy_img, t).to("cpu")
            loss = criterion(prd_noise, noise)
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model_path:
                save_checkpoint(model, model_path)
            # break
        print(loss_sum / len(train_loader))

    model.eval()

    timesteps = scheduler.timesteps
    print(timesteps)

    sample_num = config["sample_num"]

    with torch.no_grad():
        for i in range(sample_num):
            x = torch.randn((1, 1, 32, 32))
            for t in range(timesteps - 1, -1, -1):
                # x=p_sample(model,x,t,betas,alphas,hat_alphas)
                t = torch.tensor([t])
                # print(t)
                noise = model(x, t).to("cpu")
                x = scheduler.step(noise, t, x)
            # print(x)
            plt.imshow(x.squeeze().cpu(), cmap="gray")
            plt.show()
