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
    loss = recon_loss + 1*kld_loss
    return loss


def sample_mean(model, batch, n):
    batch_size = batch.shape[0]
    n_channels = batch.shape[1]
    latent_dims = model.encoder.latent_dims

    mu, logvar = (torch.empty((batch_size, 2 * n_channels, latent_dims// n_channels, 0)).to(batch) for _ in range(2))
    REC = torch.empty(batch_size, n_channels, 0).to(batch)
    # print(REC.shape)
    # print(mu.shape)

    for i in range(n):
        rec, _mu, _logvar = model(batch)

        REC = torch.cat((REC, rec.unsqueeze(-1)), dim=-1)
        mu = torch.cat((mu, _mu.unsqueeze(-1)), dim=-1)
        logvar = torch.cat((logvar, _logvar.unsqueeze(-1)), dim=-1)

    # print("shapes after cat: mu, logvar, REC ", mu.shape, logvar.shape, REC.shape)
    mu, logvar = (torch.mean(t, dim=-1) for t in [mu, logvar])
    REC = torch.mean(REC, dim=-1)
    #     print("shapes after mean: mu, logvar, REC ", mu.shape, logvar.shape, REC.shape)

    return REC, mu, logvar
def train_sgvb_loss(qnet, pnet, metrics_dict, prefix='pretrain_', name=None):
    with torch.autograd.profiler.record_function(name if name else 'pre_sgvb_loss'):
        logpx_z = pnet['x'].log_prob()
        logqz_x = qnet['z'].log_prob()
        logpz = pnet['z'].log_prob()

        kl_term = torch.mean(logqz_x - logpz)
        recons_term = torch.mean(logpx_z)
        metrics_dict[prefix + 'recons'] = recons_term.item()
        metrics_dict[prefix + 'kl'] = kl_term.item()
        
        

        return -torch.mean(logpx_z + 0.2 * (logpz - logqz_x))
    
def train_vae(train_loader, encoder, decoder, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        # Set the network to train mode
        encoder.train()
        decoder.train()
        
        # Iterate over the training data
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass through the encoder and compute the latent variables
            qnet = encoder(data)
            
            # Sample from the latent variables and forward pass through the decoder
            z_sample = qnet['z'].rsample()
            pnet = decoder(z_sample)
            x_rec = pnet['x'].rsample()
            
            # Compute the loss using the SGVB estimator
            metrics_dict = {}
            # Forward pass through the decoder and compute the reconstructed data
#             print("data", data.shape)
            logpx_z = pnet['x'].log_prob(x_rec).sum(dim=-1)
            logqz_x = qnet['z'].log_prob(z_sample).sum(dim=-1)
            logpz = torch.distributions.Normal(0, 1).log_prob(z_sample).sum(dim=-1)

            # Compute the loss using the SGVB estimator
            kl_term = (logqz_x - logpz).mean()
            recons_term = logpx_z.mean()
            loss = - (recons_term + 0.2 * kl_term)
            
            metrics_dict['pretrain_' + 'recons'] = recons_term.item()
            metrics_dict['pretrain_' + 'kl'] = kl_term.item()
            
            # Backward pass and update the parameters
            loss.backward()
            optimizer.step()
            
            # Print the loss and other metrics
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item()}, "
                      f"Reconstruction = {metrics_dict['pretrain_recons']}, "
                      f"KL = {metrics_dict['pretrain_kl']}")

### Train function
def train(model, train_loader, criterion, optimizer, device, epoch, VQ = True):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(device)
        optimizer.zero_grad()        

        if VQ:
            x_rec, loss, mu, logvar = model(data)
        else:
#             x_rec, mu, logvar = model(data)
            x_rec, mu, logvar = sample_mean(model, data, 10)
            # x_rec_window_length = x_rec.shape[2]
            loss = criterion(x_rec, data[:,:,0], mu, logvar)
        # print(x_rec.shape)
        # print(data[:, :, 0].shape)
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