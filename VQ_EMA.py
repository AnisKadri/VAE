#!/usr/bin/env python
# coding: utf-8

# In[11]:
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
from dataGen import Gen
from utils import compare, experiment
from train import slidingWindow, criterion, train, test, objective, train_vae
from Encoders import LongShort_TCVAE_Encoder, RnnEncoder, MST_VAE_Encoder, MST_VAE_Encoder_dist
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder, MST_VAE_Decoder, MST_VAE_Decoder_dist
from vae import VariationalAutoencoder, VQ_MST_VAE, VQ_Quantizer

import torch; torch.manual_seed(955)
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader
# import optuna
# from optuna.samplers import TPESampler
# import torchaudio

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pprint
from VQ_EMA_fn import *
# all parameters for generating the time series should be configured in this cell
periode = 15 #days
step = 5 # mess interval in minutes
val = 500
n_channels = 1
effects = {
    "Pulse": {
        "occurances":0,
        "max_amplitude":1.5,   
        "interval":40
        },
    "Trend": {
        "occurances":2,
        "max_slope":0.005,
        "type":"linear"
        },
    "Seasonality": {
        "occurances":3,
        "frequency_per_week":(7, 14), # min and max occurances per week
        "amplitude_range":(5, 20),
        },
    "std_variation": {
        "occurances":0,
        "max_value":10,
        "interval":1000,
        },
    "channels_coupling":{
        "occurances":0,
        "coupling_strengh":20
        },
    "Noise": {
        "occurances":0,
        "max_slope":0.005,
        "type":"linear"
        }
    }

X = Gen(periode, step, val, n_channels, effects)
x, params, e_params = X.parameters()
pprint.pprint(params)
pprint.pprint(e_params)
X.show()

### Init Model
latent_dims = 6 # 6 # 17
L= 32# 39 #32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# v = vae(n_channels, L, latent_dims)
# v = VariationalAutoencoder(input_size = n_channels,
#                       hidden_size = 30,
#                       num_layers = 3,
#                       latent_dims= latent_dims,
#                       v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder, # RnnEncoder, LongShort_TCVAE_Encoder,
#                       v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder, # RnnDecoder, LongShort_TCVAE_Decoder,
#                       L = L,
#                       slope = 0.2,
#                       first_kernel = 21)
v = VQ_MST_VAE(n_channels = n_channels,
                            num_layers =  3,#4, #3
                            latent_dims= latent_dims,
                            v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
                            v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
                            v_quantizer = VQ_Quantizer,
                            L=L,
                            slope = 0,
                            first_kernel = 20, #11, #20
                            commit_loss = 0.25) #10 5

v = v.to(device)
opt = optim.Adam(v.parameters(), lr = 0.005043529186448577) # 0.005043529186448577 0.006819850049647945
print(v)
x = torch.FloatTensor(x)


n = x.shape[1]


train_ = x[:, :int(0.8*n)]
val_   = x[:, int(0.8*n):int(0.9*n)]
test_  = x[:, int(0.9*n):]

train_data = DataLoader(slidingWindow(train_, L),
                        batch_size= 22,# 59, # 22
                        shuffle = False
                        )
val_data = DataLoader(slidingWindow(val_, L),
                        batch_size=22,
                        shuffle = False
                        )
test_data = DataLoader(slidingWindow(test_, L),
                        batch_size=22,
                        shuffle = False
                        )

for epoch in range(1, 50):
    train(v, train_data, criterion, opt, device, epoch, VQ = True)

torch.save(x, r'modules\data_{}channels_{}latent_{}window.pt'.format(n_channels,latent_dims, L))
torch.save(params, r'modules\params_{}channels_{}latent_{}window.pt'.format(n_channels,latent_dims, L))
torch.save(v, r'modules\vq_ema_{}channels_{}latent_{}window.pt'.format(n_channels,latent_dims, L))
# In[19]:


def compare(dataset, model, VQ=True):
    model.eval()
    rec = []
    x = []
    with torch.no_grad():
        for i, (data, v) in enumerate(dataset):
            if VQ:
                x_rec, loss, mu, logvar = model(data)
            else:
                x_rec, mu, logvar = model(data)
            z = model.reparametrization_trick(mu, logvar)
            if v.dim() == 1:
                v = v.unsqueeze(0)
                v = v.T
                v = v.unsqueeze(-1)
#             print(v.shape)
#             print(x_rec.shape)
#             print((x_rec * v).shape)
#             print(i)

            x.extend((data*v)[:,:,0].detach().numpy())
            rec.extend(((x_rec*v)[:,:,0]).detach().numpy())
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rec, "r--")
    ax.plot(x[:], "b-")
    plt.ylim(50,600)
    plt.grid(True)
    plt.show()


# In[23]:


v.cpu()
compare(test_data, v, VQ=True)
v.to(device)


# In[22]:



# In[39]:


# def objective(trial, model, x, criterion_fcn, train_fcn, n_channels, epochs):
#     # Define the hyperparameters to optimize
#     learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-2)
#     num_layers = trial.suggest_int('num_layers', 3, 7)
#     latent_dims = trial.suggest_int('latent_dims', 2, 20)
#     first_kernel = trial.suggest_int('first_kernel', 10, 30)
#     slope = trial.suggest_int('slope', 0.1, 0.3)
#     commit_loss = trial.suggest_int('commit_loss', 0.1, 10)
#     L = trial.suggest_int('L', 30, 512)
#     batch_size = trial.suggest_int('batch_size', 10, 100)
#     ### Init Model
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     n = x.shape[1]
#     train_ = x[:, :int(0.8*n)]
#     val_   = x[:, int(0.8*n):int(0.9*n)]
#     test_  = x[:, int(0.9*n):]
#
#     train_data = DataLoader(slidingWindow(train_, L),
#                             batch_size=batch_size,
#                             shuffle = False
#                             )
#     val_data = DataLoader(slidingWindow(val_, L),
#                             batch_size=batch_size,
#                             shuffle = False
#                             )
#     test_data = DataLoader(slidingWindow(test_, L),
#                             batch_size=batch_size,
#                             shuffle = False
#                             )
#
#     v = VQ_MST_VAE(n_channels = n_channels,
#                             num_layers = num_layers,
#                             latent_dims= latent_dims,
#                             v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
#                             v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
#                             v_quantizer = VQ_Quantizer,
#                             L=L,
#                             slope = slope,
#                             first_kernel = first_kernel,
#                             commit_loss = 10)
#     v = v.to(device)
#     # Define the loss function and optimizer
#     optimizer = optim.Adam(v.parameters(), lr=learning_rate)
#
#     for epoch in range(1, epochs):
#         loss = train_fcn(v, train_data, criterion_fcn, optimizer, device, epoch, VQ=True)
#
#
#     # Return the validation accuracy as the objective value
#     return loss
#

# In[47]:

#
# import optuna
# from optuna.samplers import TPESampler
# # Define the Optuna study and optimize the hyperparameters
# epochs = 50
# study = optuna.create_study(sampler=TPESampler(), direction='minimize')
# study.optimize(lambda trial: objective(trial,
#                                        VariationalAutoencoder,
#                                        x,
#                                        criterion,
#                                        train,
#                                        n_channels,
#                                        epochs
#                                       ),
#                n_trials=50)
#
#
# # In[48]:
#
#
# study.best_trial


# In[ ]:




