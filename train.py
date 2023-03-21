#!/usr/bin/env python
# coding: utf-8

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.distributions
# from torchsummary import summary
# import tqdm as notebook_tqdm
# from tqdm import tqdm as tq
import numpy as np
import matplotlib.pyplot as plt



class slidingWindow(Dataset):
    def __init__(self, data, L):
        self.data = data
        self.L = L

    def __getitem__(self, index):
        if self.data.shape[1] - index >= self.L:            
            x = self.data[:,index:index+self.L]        
            return x
        
    def __len__(self):
        return self.data.shape[1] - self.L


# In[3]:


### Cost function
def criterion(recon_x, x, mu, logvar):
    ### reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    ### KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    ### total loss
    loss = recon_loss + kld_loss
    return loss

### Train function
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(device)
        optimizer.zero_grad()
        x_rec, mu, logvar = model(data)

        loss = criterion(x_rec, data[:,:,0], mu, logvar)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

### Test Function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x_rec, mu, logvar = model(data)
            
            ### sum up batch loss
            loss = criterion(x_rec, data[:,:,0], mu, logvar)
            test_loss += loss.item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))