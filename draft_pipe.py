#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib inline
# %matplotlib notebook
from dataGen import Gen
from multiscale_vae import vae
from train import slidingWindow, criterion, train, test
from Encoders import LongShort_TCVAE_Encoder, RnnEncoder
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder
from vae import VariationalAutoencoder

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
import pprint


# In[2]:


# all parameters for generating the time series should be configured in this cell
periode = 15 #days
step = 5 # mess interval in minutes
val = 100
n_channels = 3
effects = {
    "Pulse": {
        "occurances":0,
        "max_amplitude":2,   
        "interval":20
        },
    "Trend": {
        "occurances":1,
        "max_slope":0.005,
        "type":"mixed"
        },
    "Seasonality": {
        "occurances":2,
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

### Init Model
latent_dims = 15
L = 60
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# v = vae(n_channels, L, latent_dims)
v = VariationalAutoencoder(input_size = n_channels,
                           hidden_size = 30,
                           num_layers = 3,
                           latent_dims= latent_dims,
                           v_encoder = LongShort_TCVAE_Encoder, # RnnEncoder, LongShort_TCVAE_Encoder,
                           v_decoder = LongShort_TCVAE_Decoder, # RnnDecoder, LongShort_TCVAE_Decoder,
                           L = L,
                           slope = 0.2,
                           first_kernel = 21)
opt = optim.Adam(v.parameters(), lr = 0.001571)


# In[3]:


X = Gen(periode, step, val, n_channels, effects)
x, params, e_params = X.parameters()
# pprint.pprint(params)
pprint.pprint(e_params)
X.show()


# In[4]:


# serie = torch.tensor(serie).float()
x = torch.FloatTensor(x)
n = x.shape[1]

train_ = x[:, :int(0.8*n)]
val_   = x[:, int(0.8*n):int(0.9*n)]
test_  = x[:, int(0.9*n):]

# train_set = slidingWindow(train, 30)
# val_set = slidingWindow(val, 30)
# test_set = slidingWindow(test, 30)

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


# In[5]:


print(v)


# In[6]:


for epoch in range(1, 100):
    train(v, train_data, criterion, opt, device, epoch)


# In[7]:


def compare(dataset, model):
    model.eval()
    rec = []
    x = []
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x_rec, mu, logvar = model(data)
            z = v.reparametrization_trick(mu, logvar)

            x.extend(data[:,:,0].detach().numpy())
            rec.extend(x_rec[:].detach().numpy())
        
    print(mu[-1, :], logvar[-1, :])
    plt.plot(rec, "r--")
    plt.plot(x[:], "b-")
    plt.ylim(0,100)
    plt.grid(True)
    
    return z[-1, :]


# In[8]:


z = compare(train_data, v)


# In[9]:


z = compare(test_data, v)


# In[9]:


test(v, test_data, criterion, device)


# In[ ]:


# 1) Pulse/ Rechteck
# 2) Trends (linear or not)
# 3) Periodicity
# std. effects
# 5) coupling durch cov matrix
# 6) Noise
# 7) Effect of latent dim änderung, welche größe ist am besten  geeignet, soll latent_dim = n_channels sein? couploung durch hiarchie?
# Modeling of interactions effects in die simulation hilft um die interpretation zu validieren, man kann die effecte aus der realen daten 
# besser mathematisch verstehen
# 8) Short and long term effects durch die verschiedenen Convolutions weights"aktivierung von neuronen/ welche neuraonen sind mehr active in welche fälle"
# idee is to know what layers capt long term effects and which layers capt short terms to be able to better interpret the data
# 9) also change the effects to be intern effects -> they happen on the level of mean and std rather than additif to the channels and let only be some random noise additif (y = x +epsilon)
# all the rest (trends, seasonalities, pulses should happen on the level of mean and std)


# Write and document all the steps I wanna do and make check list
# try to make the data scales and dimention real (x axes represents tims -> days, hours, secs / are the seasonalities daily or weekly etc)


# In[40]:


# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ipywidgets import interact



def experiment(data, model):
    x = []
    z = []
    rec = []
    with torch.no_grad():
        for i, data in enumerate(data):
            x_rec, mu, logvar = model(data)
            Z = v.reparametrization_trick(mu, logvar)
            
            x.extend(data[:,:,0].detach().numpy())
            z.extend(Z[:].detach().numpy())
            rec.extend(x_rec[:].detach().numpy())
            
    x = np.array(x)
    z = np.array(z)
    rec = np.array(rec)                 

    # Create a figure and axis object
    fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [1, 2]})   
    
    # Plot the initial data
    ax1 = axs[0]
    ax2 = axs[1]
    line1 = ax1.plot(x, "b")
    line2 = ax1.plot( rec, "r")
    line3 = ax2.plot( z, "g")
    
    
    sliders = []
    slider_axs = []
              
#     # Add a slider widget for variable 1
    for i in range(z.shape[1]):
        slider_axs.append( plt.axes([0.1, (0.00-0.05*i), 0.8, 0.03]) )
        sliders.append( Slider(slider_axs[i], r'$Z_{}$'.format(i), -10, 10, valinit=z[-1,i]) )   


    # Define a function to update the plot
    def update(val):
        
        for i in range(z.shape[1]):
            z[:,i] = sliders[i]. val            
        rec = v.decoder(z)        

        line2.set_ydata(rec)    
        line3.set_ydata(z)

        fig.canvas.draw_idle()

    # Connect the sliders to the update function
    for slider in sliders:
        slider.on_changed(update)

    # Show the plot
    plt.show()


# In[41]:


# experiment(test_data, v)


# In[5]:


import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def objective(trial):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-1)
#     num_hidden_units = trial.suggest_int('num_hidden_units', 16, 256)
    num_layers = trial.suggest_int('num_layers', 3, 5)
#     dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    latent_dims = trial.suggest_int('latent_dims', 3, 15)
    first_kernel = trial.suggest_int('first_kernel', 15, 30)
    slope = trial.suggest_int('slope', 0, 0.4)
    
    ### Init Model
    latent_dims = 15
    L = 60
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # v = vae(n_channels, L, latent_dims)
    v = VariationalAutoencoder( input_size = n_channels,
                                hidden_size = 30,
                                num_layers = num_layers,
                                latent_dims= latent_dims,
                                v_encoder = LongShort_TCVAE_Encoder, # RnnEncoder, LongShort_TCVAE_Encoder,
                                v_decoder = LongShort_TCVAE_Decoder, # RnnDecoder, LongShort_TCVAE_Decoder,
                                L = L,
                                slope = 0.2,
                                first_kernel = first_kernel)
    # Define the loss function and optimizer
    optimizer = optim.Adam(v.parameters(), lr=learning_rate)
    
    for epoch in range(1, 100):
        train(v, train_data, criterion, optimizer, device, epoch)
    
    test_loss = test(v, test_data, criterion, device)
    print(test_loss)
    
    # Return the validation accuracy as the objective value
    return test_loss


# train_data = DataLoader(slidingWindow(train_, L),
#                         batch_size=10,
#                         shuffle = False
#                         )
# val_data = DataLoader(slidingWindow(val_, L),
#                         batch_size=10,
#                         shuffle = False
#                         )
# test_data = DataLoader(slidingWindow(test_, L),
#                         batch_size=10,
#                         shuffle = False
#                         )

# Define the number of epochs to train for
# num_epochs = 100

# # Define the Optuna study and optimize the hyperparameters
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# # Print the best hyperparameters and validation accuracy found
# print('Best hyperparameters: {}'.format(study.best_params))
# print('Best test loss: {:.2f}%'.format(study.best_value * 100))


# # In[6]:


# print('Best hyperparameters: {}'.format(study.best_params))
# print('Best test loss: {:.2f}%'.format(study.best_value * 100))


# In[ ]:




