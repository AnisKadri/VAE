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
from Encoders import LongShort_TCVAE_Encoder, RnnEncoder
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder


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

### Cost function
def criterion_tol(x_recon, x, mu, logvar, tol = 1):
    ### reconstruction loss

    recon_loss = torch.where( torch.abs(x_recon -x) < tol, 0, F.mse_loss(x_recon, x, reduction='sum')).sum()
    
#     recon_loss = F.mse_loss(x_recon, x, reduction='sum')


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
#         print(data.shape)
        x_rec, mu, logvar = model(data)
#         print(x_rec.shape)
#         print(data[:,:,-1].shape)
#         print(data[:,:,-1])

        loss = criterion(x_rec, data[:,:,-1], mu, logvar)
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
    return test_loss

def objective(trial, model, train_data, test_data, criterion_fcn, train_fcn, test_fcn, n_channels, L, epochs):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-1)
    num_layers = trial.suggest_int('num_layers', 3, 5)
    latent_dims = trial.suggest_int('latent_dims', 3, 15)
    first_kernel = trial.suggest_int('first_kernel', 15, 30)
    slope = trial.suggest_int('slope', 0, 0.4)

    ### Init Model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    v = model(input_size=n_channels,
               hidden_size=30,
               num_layers=num_layers,
               latent_dims=latent_dims,
               v_encoder=LongShort_TCVAE_Encoder,  # RnnEncoder, LongShort_TCVAE_Encoder,
               v_decoder=LongShort_TCVAE_Decoder,  # RnnDecoder, LongShort_TCVAE_Decoder,
               L=L,
               slope=slope,
               first_kernel=first_kernel)
    # Define the loss function and optimizer
    optimizer = optim.Adam(v.parameters(), lr=learning_rate)

    for epoch in range(1, epochs):
        train_fcn(v, train_data, criterion_fcn, optimizer, device, epoch)

    test_loss = test_fcn(v, test_data, criterion_fcn, device)
    print(test_loss)

    # Return the validation accuracy as the objective value
    return test_loss