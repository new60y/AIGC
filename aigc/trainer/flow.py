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

# from aigc.load_data import load_data
from aigc.model.unet import Unet
from utils import load_checkpoint, save_checkpoint, set_device, set_seed

device = set_device()

set_seed()


def train(train_loader, batch_size, epochs, lr, config, model_path=None):
    # train_loader = DataLoader(train_set, batch_size, True)

    # model = Unet(config["channel"]).to(device)
    model = config["gen"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    # print(schedule)

    # betas,alphas,hat_alphas=schedule(timesteps)

    timesteps = config["timesteps"]
    print(timesteps)

    for epoch in range(epochs):
        loss_sum = 0
        print("epoch", epoch)
        for img, label in tqdm(train_loader):
            # img=img.to(device)
            # noise,noisy_img,t=q_sample(img,hat_alphas,timesteps)
            x1 = img
            x0 = torch.randn_like(img)
            t = torch.rand(img.shape[0])
            # print(x1.shape,t.shape)
            xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
            # print(xt.shape)
            pred_v = model(xt, t).to("cpu")
            gt_v = x1 - x0
            loss = criterion(pred_v, gt_v)
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model_path:
                save_checkpoint(model, model_path)
            # break
        print(loss_sum / len(train_loader))

    model.eval()

    timesteps = config["timesteps"]
    print(timesteps)

    delta = 1.0 / timesteps

    sample_num = config["sample_num"]

    with torch.no_grad():
        for i in range(sample_num):
            t = torch.tensor([0.0])
            x = torch.randn((1, 1, 32, 32))
            # for t in range(timesteps-1,-1,-1):
            for i in range(timesteps):
                # t=torch.tensor([t])
                vt = model(x, t).to("cpu")
                t += delta
                x += vt * delta

            # print(x)
            plt.imshow(x.squeeze().cpu(), cmap="gray")
            plt.show()
