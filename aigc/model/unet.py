import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision

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
        freqs=torch.pow(10000,torch.linspace(0,1,dim//2)).to(device)
        sin_emb=torch.sin(t[:,None]/freqs)
        cos_emb=torch.cos(t[:,None]/freqs)

        embed=torch.cat([sin_emb,cos_emb],dim=-1)
        #print("embed:",sin_emb.shape,cos_emb.shape,embed.shape)
        return embed

    def forward(self,x,t):
        x=x.to(device)
        #t=torch.tensor(t).reshape(x.shape[0],-1).to(device)
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