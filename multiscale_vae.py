#!/usr/bin/env python
# coding: utf-8

# In[8]:


# from dataGen import Gen
# from vae import VariationalAutoencoder

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.distributions


import numpy as np
import matplotlib.pyplot as plt


# In[186]:

def lin_size(n, kernels):
    for k in kernels:
        n = 1+ ((n-k)//2)
    return n
        
class slidingWindow(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        if self.data.shape[1] - index >= self.window:            
            x = self.data[:,index:index+self.window]        
            return x
        
    def __len__(self):
        return self.data.shape[1] - self.window


# In[176]:


class shortEncoder(nn.Module):
    def __init__(self, n_channels, L, latent_dims):
        super(shortEncoder, self).__init__()        
        
         ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(n_channels, n_channels*2, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*2),
            
            nn.Conv1d(n_channels*2, n_channels*4, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*4),   
            
            nn.Conv1d(n_channels*4, n_channels*8, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*8),
        )       
        
        self.n = lin_size(L, [2,2,2])
        
        ### Linear section: mean
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.n * n_channels*8, 32*latent_dims),
            nn.ReLU(True),
            nn.Linear(32*latent_dims, latent_dims)
        )
        
        self.encoder_var_log = nn.Sequential(
            nn.Linear(self.n * n_channels*8, 32*latent_dims),
            nn.ReLU(True),
            nn.Linear(32*latent_dims, latent_dims)
        )
         

    def forward(self, x):
        ### CNN
        x = self.encoder_cnn(x) 
        x = x.view(x.size(0),-1)
        
        ### MLP
        mu = self.encoder_mu(x)  
        var_log = self.encoder_var_log(x)
        
        return mu, var_log
    
class longEncoder(nn.Module):
    def __init__(self, n_channels, L, latent_dims):
        super(longEncoder, self).__init__()      
                                
         ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(n_channels, n_channels*2, kernel_size=15, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*2),
            
            nn.Conv1d(n_channels*2, n_channels*4, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*4),   
            
            nn.Conv1d(n_channels*4, n_channels*8, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*8),
        )       
        
        self.n = lin_size(L, [15,2,2])
        ### Linear section: mean
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.n * n_channels*8, 32*latent_dims),
            nn.ReLU(True),
            nn.Linear(32*latent_dims, latent_dims)
        )
        
        self.encoder_var_log = nn.Sequential(
            nn.Linear(self.n * n_channels*8, 32*latent_dims),
            nn.ReLU(True),
            nn.Linear(32*latent_dims, latent_dims)
        )
         

    def forward(self, x):
        ### CNN
        x = self.encoder_cnn(x) 
        x = x.view(x.size(0),-1)
        
        ### MLP
        mu = self.encoder_mu(x)  
        var_log = self.encoder_var_log(x)
        
        return mu, var_log
    
class Decoder(nn.Module):
    def __init__(self, n_channels, latent_dims):
        super(Decoder, self).__init__()        
        
         ### Linear section
        self.dec_lin = nn.Sequential(
            nn.Linear(2*latent_dims, 8*latent_dims),
            nn.ReLU(True),
            nn.Linear(8*latent_dims, 16*latent_dims),
            nn.ReLU(True),
            nn.Linear(16*latent_dims, 32*latent_dims),
            nn.ReLU(True),
            nn.Linear(32*latent_dims, n_channels),
        )      

    def forward(self, x):
        ### CNN
        x = self.dec_lin(x)  
        return x     


# In[177]:


class vae(nn.Module):
    def __init__(self, n_channels, L, latent_dims):
        super(vae, self).__init__()
        
        self.sencoder = shortEncoder(n_channels, L, latent_dims)
        self.lencoder = longEncoder(n_channels, L, latent_dims)
        self.decoder = Decoder(n_channels, latent_dims)
        
    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        smu, slogvar = self.sencoder(x)        
        lmu, llogvar = self.lencoder(x)

        mu = torch.cat((smu, lmu), axis=1)
        logvar = torch.cat((slogvar, llogvar), axis=1)
        
        z = self.reparametrization_trick(mu, logvar)
       
        return self.decoder(z), mu, logvar   


# In[243]:


### Cost function
def criterion(recon_x, x, mu, logvar):
    ### reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
#     recon_loss = nn.MSELoss(recon_x, x, reduction='sum')

    ### KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    ### total loss
    loss = recon_loss + kld_loss
    return loss

### Train function
def train(v, train_loader, criterion, optimizer, device, epoch):
    v.train()
    train_loss = 0
#     rec_bach = torch.tensor([])
#     for i in range(0, params["T"]-L):
    for batch_idx, data in enumerate(train_loader):
#         W = torch.tensor(serie.T[i:i+L]).float()
        
        data = data.to(device)
        optimizer.zero_grad()
        x_rec, mu, logvar = v(data)
#         rec_bach = torch.cat((rec_bach, x_rec), 0)
#     print(serie.T[:params["T"]-L].shape)
#     loss = criterion(rec_bach, serie.T[:params["T"]-L].squeeze(), mu, logvar)
        
#         recon_batch, mu, logvar = vae(W)
#         print(x_rec.shape)
#         print(data.shape)
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
def test(vae, test_loader, criterion, device):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = vae(data)
            
            ### sum up batch loss
            loss = criterion(recon_batch, data, mu, logvar)
            test_loss += loss.item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))








