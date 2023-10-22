#!/usr/bin/env python
# coding: utf-8

import torch;

torch.manual_seed(0)
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
from utils import *
import utils


class NoWindow(Dataset):
    def __init__(self, data, args, labels=None, norm=True):
        self.data = data
        self.labels = labels
        self.n = data.shape[-1]
        self.norm = norm
        self.min_max = args.min_max
        

    def __getitem__(self, index):
        x, label = self.data[index], None
        if self.norm:
            norm = torch.sum(x, axis=(x.dim() - 1), keepdim=True) / self.n
            norm = 1

        label = self.labels[index]
#         idxs = utils.get_means_indices(label)
#         label[idxs, 1] = label[idxs, 1] / norm
        
        x_norm, v_min, dist = self.min_max_norm(x)
        x_std, mean, std = self.standarization(x)
#         x = x_norm if self.min_max else x_std
#         x = x / norm
        return (x_norm, x_std), label, (dist, v_min, std, mean)

    def __len__(self):
        return self.data.shape[0]
    
    def min_max_norm(self, data):
        v_min, v_max = data.min(), data.max()
        dist = v_max - v_min
        normed_val = (data - v_min) / dist        
        return normed_val, v_min, dist
    
    def standarization(self, data):
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        normed_val = (data - mean) / std        
        return normed_val, mean, std 


class SlidingWindow(Dataset):
    def __init__(self, data, L, stride=0):
        self.data = data
        self.L = L
        self.stride = stride

    def __getitem__(self, index):
        if self.data.shape[-1] - index >= self.L:
            x = self.data[..., index:index + self.L]
            norm = torch.sum(x, axis=(self.data.dim() - 1), keepdim=True) / self.L
            x = x / norm
            return x, "", norm

    def __len__(self):
        return self.data.shape[-1] - self.L


class StridedWindow(Dataset):
    def __init__(self, data, args, stride=0):
        self.data = data
        self.L = args.L
        self.stride = stride
        
        self.min_max = args.min_max
        self.v_min, self.v_max = data.min(), data.max()
        self.dist = self.v_max - self.v_min

    def __getitem__(self, index):
        delay = index * self.stride
        current_window = index * self.L

        if self.data.shape[-1] - current_window >= self.L:
            x = self.data[..., (current_window - delay): ((current_window + self.L) - delay)]
#             norm = torch.sum(x/ self.L, axis=(self.data.dim() - 1), keepdim=True) #/ self.L
#             x = x / norm
#             print(x.shape)
            x_norm = self.min_max_norm(x)
            x_std, mean, std = self.standarization(x)
#             x = x_norm if self.min_max else x_std

#             print(mean, std)
            return (x_norm, x_std), "", (self.dist, self.v_min, std, mean)

    def __len__(self):
        return (self.data.shape[-1] // self.L)
    
    def min_max_norm(self, data):
        normed_val = (data - self.v_min) / self.dist        
        return normed_val
    
    def standarization(self, data):
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        normed_val = (data - mean) / std        
        return normed_val, mean, std    


def create_loader_noWindow(x, args, labels, norm=True):
    x = torch.FloatTensor(x)
    n = x.shape[0]
    train_split, test_split = args.split
    window = args.no_window
    shuffle = args.shuffle
    batch_size = args.bs

    train_, train_labels = x[:int(train_split * n)], labels[:int(train_split * n)]
    val_, val_labels = x[int(train_split * n):int(test_split * n)], labels[int(train_split * n):int(test_split * n)]
    test_, test_labels = x[int(test_split * n):], labels[int(test_split * n):]

    train_data = DataLoader(window(train_, args, train_labels, norm=norm),  # slidingWindow, stridedWindow
                            batch_size=batch_size,  # 59, # 22
                            shuffle=shuffle
                            )
    val_data = DataLoader(window(val_, args, val_labels, norm=norm),
                          batch_size=batch_size,
                          shuffle=shuffle
                          )
    test_data = DataLoader(window(test_, args, test_labels, norm=norm),
                           batch_size=batch_size,
                           shuffle=shuffle
                           )
    return train_data, val_data, test_data


def create_loader_Window(x, args):
    # from train import noWindow
    x = torch.FloatTensor(x)
    n = x.shape[-1]
    train_split, test_split = args.split
    window = args.window
    batch_size = args.bs
    shuffle=False

    train_ = x[..., :int(train_split * n)]
    val_ = x[..., int(train_split * n):int(test_split * n)]
    test_ = x[..., int(test_split * n):]

    train_data = DataLoader(window(train_, args),  # slidingWindow, stridedWindow
                            batch_size=batch_size,  # 59, # 22
                            shuffle=shuffle,
                            drop_last=False
                            )
    val_data = DataLoader(window(val_, args),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          drop_last=False
                          )
    test_data = DataLoader(window(test_, args),
                           batch_size=batch_size,
                           shuffle=shuffle,
                           drop_last=False
                           )
    return train_data, val_data, test_data

def pick_data(data_tup, args):
    if args.min_max:
        data = data_tup[0].to(args.device)
    else:
        data = data_tup[1].to(args.device)
    return data


# In[3]:
def criterion(recon_x, x, mu, logvar):
    ### reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    ### KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    ### total loss
    loss = recon_loss + 1 * kld_loss
    return loss


def sample_mean(model, batch, n):
    batch_size = batch.shape[0]
    n_channels = batch.shape[1]
    latent_dims = model.encoder.latent_dims

    mu, logvar = (torch.empty((batch_size, 2 * n_channels, latent_dims // n_channels, 0)).to(batch) for _ in range(2))
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


def train(model, train_loader, args, optimizer, epoch):
    device = args.device
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    train_loss = 0

    for batch_idx, (data_tup, labels, norm) in enumerate(train_loader):
        
        data = pick_data(data_tup, args)
        norm = [n.to(device) for n in norm]
        optimizer.zero_grad()
        

        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e = model(data)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            #             print("True Loss: ", loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t True Loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item() / len(data), loss.item()))
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return train_loss / len(train_loader.dataset)


def tune(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    for p in model.parameters():
        p.requires_grad = True
    for p in model.quantizer.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True

    num_embed = model.quantizer._num_embed
    latent_dims = model._latent_dims
    n_channels = model._n_channels

    train_loss = 0
    rec = torch.empty(n_channels, 0, device=device)
    x = torch.empty(n_channels, 0, device=device)
    quantizer_output = torch.empty(0, num_embed, latent_dims, device=device)

    for batch_idx, (data_tup, _, norm) in enumerate(train_loader):

        data = pick_data(data_tup, args)
        norm = [n.to(device) for n in norm]
        optimizer.zero_grad()

        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e = model(data)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return loss




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

### Test Function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x_rec, mu, logvar = model(data)

            ### sum up batch loss
            loss = criterion(x_rec, data[:, :, 0], mu, logvar)
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


def train_MCMC(model, train_loader, criterion, optimizer, device, epoch, VQ=True, repetitions=5):
    model.train()
    n_channels = model._n_channels
    x_len = len(train_loader.dataset)
    bs = train_loader.batch_size
    label_data = torch.empty(0, n_channels, device=device)
    train_loss = 0
    n = 5
    L = model._L
    for i in range(repetitions):
        train_loss = 0
        new_x = torch.empty(0, n_channels, device=device)

        #         print("here again")
        for batch_idx, (data, v) in enumerate(train_loader):

            data = data.to(device)
            v = v.to(device)
            optimizer.zero_grad()

            if VQ:
                x_rec, loss, mu, logvar, mu_rec, logvar_rec = model(data)
            else:
                #             x_rec, mu, logvar = model(data)
                x_rec, mu, logvar = sample_mean(model, data, 10)
                if v.dim() == 1:
                    v = v.unsqueeze(-1)
                    v = v.unsqueeze(-1)
                # x_rec_window_length = x_rec.shape[2]
                loss = criterion((x_rec * v)[:, :, 0], data[:, :, 0], mu, logvar)
            # print(x_rec.shape)
            # print(data[:, :, 0].shape)
            #             rec_reshaped = x_rec[:,:,0]
            #             print("norm shape", v.shape)
            #             print("new_x shape", new_x.shape)
            #             print("rec point to add", x_rec[:,:,0].shape)
            new_x = torch.cat((new_x, (x_rec * v)[:, :, 0]), dim=0)
            if i == 0:
                label_data = torch.cat((label_data, (data * v)[:, :, 0]), dim=0)
            #             print("new x filling", new_x.shape)
            if data.shape[0] != bs:
                #                 print("last batch")
                #                 print("shape to take", (x_rec*v)[-1,:,0:].permute(1,0).shape)
                new_x = torch.cat((new_x, (x_rec * v)[-1, :, 0:].permute(1, 0)), dim=0).T
                if i == 0:
                    label_data = torch.cat((label_data, (data * v)[-1, :, 0:].permute(1, 0)), dim=0).T

            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        #         print("final shape of x:", new_x.shape)
        x = torch.FloatTensor(new_x.cpu().detach().numpy())
        new_x, label_data = new_x.cpu(), label_data.cpu()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(new_x.T.detach().numpy(), "r--")
        ax.plot(label_data.T.detach().numpy(), "b", alpha=0.2)
        train_loader = DataLoader(SlidingWindow(x, L),
                                  batch_size=22,  # 59, # 22
                                  shuffle=False
                                  )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(new_x.T.detach().numpy(), "r--")
    ax.plot(label_data.T.detach().numpy(), "b", alpha=0.2)

    return train_loss / len(train_loader.dataset)
