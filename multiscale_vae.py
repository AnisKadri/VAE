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
            nn.Linear(self.n * n_channels*8, latent_dims)
#             nn.Linear(self.n * n_channels*8, 32*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(32*latent_dims, latent_dims)
        )
        
        self.encoder_var_log = nn.Sequential(
            nn.Linear(self.n * n_channels*8, latent_dims)
#             nn.Linear(self.n * n_channels*8, 32*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(32*latent_dims, latent_dims)
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
            nn.Linear(self.n * n_channels*8, latent_dims)
#             nn.Linear(self.n * n_channels*8, 32*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(32*latent_dims, latent_dims)
        )
        
        self.encoder_var_log = nn.Sequential(
            nn.Linear(self.n * n_channels*8, latent_dims)
#             nn.Linear(self.n * n_channels*8, 32*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(32*latent_dims, latent_dims)
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
    def __init__(self, n_channels, L, latent_dims):
        super(Decoder, self).__init__()        
        
         ### Linear section
#         self.dec_lin = nn.Sequential(
#             nn.Linear(2*latent_dims, 8*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(8*latent_dims, 16*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(16*latent_dims, 32*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(32*latent_dims, n_channels),
#         )  
        self.n = lin_size(L, [2,2,2])
        self.dec_lin = nn.Sequential(
            nn.Linear(2*latent_dims, n_channels*8)
#             nn.Linear(2*latent_dims, 32*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(32*latent_dims, n_channels*8)
#             nn.Linear(8*latent_dims, 16*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(16*latent_dims, 32*latent_dims),
#             nn.ReLU(True),
#             nn.Linear(32*latent_dims, n_channels),
        )  
        
        ### Convolutional section
        self.dec_cnn = nn.Sequential(
            nn.ConvTranspose1d(n_channels*8, n_channels*4, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*4),
            
            nn.ConvTranspose1d(n_channels*4, n_channels*2, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(n_channels*2),   
            
            nn.ConvTranspose1d(n_channels*2, n_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),

        )

    def forward(self, x):
        ### CNN
#         print(x.shape)
        x = self.dec_lin(x)  
#         print(x.shape)
        x = self.dec_cnn(x.view(x.shape[0],x.shape[1],1))
#         print(x.shape)
        return x[:,:,0]     


# In[177]:


class vae(nn.Module):
    def __init__(self, n_channels, L, latent_dims):
        super(vae, self).__init__()
        
        self.sencoder = shortEncoder(n_channels, L, latent_dims)
        self.lencoder = longEncoder(n_channels, L, latent_dims)
        self.decoder = Decoder(n_channels, L, latent_dims)
        
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
