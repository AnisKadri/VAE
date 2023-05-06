#!/usr/bin/env python
# coding: utf-8

# # Pipeline for testing models on MTS
# 
# ## Introduction
# This notebook is meant to be a draft for different **VAE** (Variational Autoencoder) struktures to test them on generated **MTS** (Multivariate Time Series) with varying complexity.    
# It uses the **dataGen** class to simulate some artificial manufakturing Data, where all the parameters and effects (Seasonalities, Trends, Couling and Anomalies) can be controlled.   
# The **Encoders** and **Decoders** classes contain the different encoding, decoding blocks to form the vae, **vae** uses combines them together to form vae.   
# **train** class contains the functions to train and test the models. All other functions for plotting the results and experimenting with the Latent Representation are stored in **utils**.  

# In[1]:
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from dataGen import Gen
from utils import compare, experiment
from train import slidingWindow, criterion, train, test, objective, train_vae
from Encoders import LongShort_TCVAE_Encoder, RnnEncoder, MST_VAE_Encoder, MST_VAE_Encoder_Linear, MST_VAE_Encoder_dist
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder, MST_VAE_Decoder, MST_VAE_Decoder_Linear, MST_VAE_Decoder_dist
from vae import VariationalAutoencoder, VQ_MST_VAE, VQ_Quantizer

import torch; torch.manual_seed(955)
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader
# import optuna
# from optuna.samplers import TPESampler


import numpy as np
import matplotlib.pyplot as plt
import pprint


# ## Generating the fake data
# 
# Here we specify all the parameters for the data generation.   
# It is important to note that all the parameters control the shape of the data but the **MTS** is generated randomly with respect to these inputs. The actual paramters ($\mu$ and $\gamma$ at each point, indexes and values of each effect etc...) are stored in the **params** and **e_params** attributes of the class.
# ### Channels
# - **Periode:** The number of days to simulate.
# - **Step:** How many minutes between each Measurement.
# - **Val:** The maximum Value possible in y-axis.
# - **n_channels:** The number of channels to simulate.   
# 
# ### Effects
# All Effect but the **Noise** are applied on the mean and std level (internal) and not on the final values.
# - **Pulse:** Can be a point or a rectangular pulse over an interval.
# - **Trend:** Is basically a shift in one of the channels, can be linear or quadratic.
# - **Seasonality:** Adds a sinusiudal fluctuation on a channel, the frequency here is 'how many oscillations per week'.
# - **Std_variation:** Changes the std over an interval .
# - **Channel_Coupling** Adds a coupling between two channels over the whole Simulation. If active, the Cov Matrix will have non-zero values beside the diagonal.
# - **Noise:** Adds some noise on the generated values to simulate measurement noise &rarr; $y_{noise} = y +\epsilon $.

# In[2]:


# all parameters for generating the time series should be configured in this cell
periode = 15 #days
step = 5 # mess interval in minutes
val = 100
n_channels = 5
effects = {
    "Pulse": {
        "occurances":5,
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
# pprint.pprint(params)
pprint.pprint(e_params)
X.show()


# ## Model Init
# The **vae** and **optimizer** are initialized in this cell.  
# - **input_size** is the number of channels.
# - **hidden_size** is the number of Neurons in the hidden size of the last MLP layer of the **Encoder** (to generate the $\mu$ and $\sigma$).
# - **num_layer** is the number of layers in the main Structure (TCN or RNN for now).
# - **latent_dims** is the number of variables in the Latent Representation.
# - **v_encoder** which Encoder to use from the **Encoders** class.
# - **v_decoder** which Encoder to use from the **Decoders** class.
# - **L** is the window length.
# - **slope** is the slope value for the LeakyRelu activation (if they are used).
# - **first_kernel** is only relevant for the LongShort_TCVAE and specifies the Kernel length of the first conv layer in the long TCN.

# In[3]:


### Init Model
latent_dims = n_channels * 3
L=30
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# v = vae(n_channels, L, latent_dims)
# v = VariationalAutoencoder(input_size = n_channels,
#                            hidden_size = 30,
#                            num_layers = 3,
#                            latent_dims= latent_dims,
#                            v_encoder = LongShort_TCVAE_Encoder, # RnnEncoder, LongShort_TCVAE_Encoder,
#                            v_decoder = LongShort_TCVAE_Decoder, # RnnDecoder, LongShort_TCVAE_Decoder,
#                            L = L,
#                            slope = 0.2,
#                            first_kernel = 21)
# v = VQ_MST_VAE(n_channels = n_channels,
#                            num_layers = 3,
#                            latent_dims= 10,
#                            v_encoder = LongShort_TCVAE_Encoder, # MST_VAE_Encoder, #LongShort_TCVAE_Encoder,
#                            v_decoder = LongShort_TCVAE_Decoder, # MST_VAE_Decoder, #LongShort_TCVAE_Decoder,
#                            v_quantizer = VQ_Quantizer,
#                            L=30,
#                            slope = 0.2,
#                            first_kernel = 15,
#                            commit_loss = 3.5)
v = VariationalAutoencoder(input_size = n_channels,
                           num_layers = 3,
                           latent_dims= latent_dims,
                           v_encoder = MST_VAE_Encoder_Linear, # MST_VAE_Encoder_Linear, #LongShort_TCVAE_Encoder,
                           v_decoder = MST_VAE_Decoder_Linear, # MST_VAE_Decoder_Linear, #LongShort_TCVAE_Decoder,
                           L=L,
                           slope = 0.2,
                           first_kernel = 15
                           )
enc = MST_VAE_Encoder_dist(     n_channels,
                           num_layers = 3,
                           slope = 0.2,
                           first_kernel = 15)
dec = MST_VAE_Decoder_dist(     n_channels,
                           num_layers = 3,
                           slope = 0.2,
                           first_kernel = 15)


v = v.to(device)
enc = enc.to(device)
dec = dec.to(device)
opt = optim.Adam(v.parameters(), lr = 0.001571)
# torch.save(v, r'modules\mst_vae.pt')


# ## Split and Dataloader
# This cell is for splitting the Data and creating a Dataloader for each set. The dataloaders will return the data using an overlapping sliding window over the time axis.

# In[4]:


# serie = torch.tensor(serie).float()
x = torch.FloatTensor(x)
n = x.shape[1]


train_ = x[:, :int(0.8*n)]
val_   = x[:, int(0.8*n):int(0.9*n)]
test_  = x[:, int(0.9*n):]

train_data = DataLoader(slidingWindow(train_, L),
                        batch_size=10,
                        shuffle = False
                        )
val_data = DataLoader(slidingWindow(val_, L),
                        batch_size=10,
                        shuffle = False
                        )
test_data = DataLoader(slidingWindow(test_, L),
                        batch_size=10,
                        shuffle = False
                        )


# ## Training 
# The data is trained using the train class for 100 epochs. A hyperparameter optimisation can be also run (needs gpu)

# In[5]:


for epoch in range(1, 100):
    train(v, train_data, criterion, opt, device, epoch, VQ=False)
torch.save(v, r'modules\mst_vae_lin2.pt')
# test(v, test_data, criterion, device)


# In[ ]:


# # Define the Optuna study and optimize the hyperparameters
# epochs = 10
# study = optuna.create_study(sampler=TPESampler(), direction='minimize')
# study.optimize(lambda trial: objective(trial,
#                                        VariationalAutoencoder,
#                                        train_data,
#                                        test_data,
#                                        criterion,
#                                        train,
#                                        test,
#                                        n_channels,
#                                        L, epochs
#                                       ),
#                n_trials=10)


# 

# ## Plot and Experiment
# **compare** function plots the <span style="color:blue">Original Data</span> vs <span style="color:red">Reconstruction </span>.
# **experiment** generates 2 interactive plots.
# - **Left** is the same plot generated by compare (<span style="color:blue">Original Data</span> vs <span style="color:red">Reconstruction </span>)
# - **Right** are the Latent Variables **Z** over time.   
# The Sliders on the bottom control the values of each $z$ to see the effect it has on the reconstruction. both plots react to the change of $z$ values.

# In[6]:


compare(train_data, v)


# In[7]:


compare(test_data, v)




# In[9]:


experiment(test_data, v)


# In[ ]:




