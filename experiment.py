import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from dataGen import Gen
from utils import compare, experiment, experiment_VQ
from train import slidingWindow, criterion, train, test, objective
from Encoders import LongShort_TCVAE_Encoder, RnnEncoder
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder
from vae import VariationalAutoencoder

import torch; torch.manual_seed(955)
import torch.optim as optim
from torch.utils.data import DataLoader
# import optuna
# from optuna.samplers import TPESampler


import numpy as np
import matplotlib.pyplot as plt
import pprint

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
# n_channels = 1
# effects = {
#     "Pulse": {
#         "occurances":2,
#         "max_amplitude":1.5,
#         "interval":40
#         },
#     "Trend": {
#         "occurances":2,
#         "max_slope":0.005,
#         "type":"linear"
#         },
#     "Seasonality": {
#         "occurances":2,
#         "frequency_per_week":(7, 14), # min and max occurances per week
#         "amplitude_range":(5, 20),
#         },
#     "std_variation": {
#         "occurances":0,
#         "max_value":10,
#         "interval":1000,
#         },
#     "channels_coupling":{
#         "occurances":0,
#         "coupling_strengh":20
#         },
#     "Noise": {
#         "occurances":0,
#         "max_slope":0.005,
#         "type":"linear"
#         }
#     }

X = Gen(periode, step, val, n_channels, effects)
x, params, e_params = X.parameters()
# pprint.pprint(params)
pprint.pprint(e_params)
# X.show()

latent_dims = n_channels * 3
L = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
v = torch.load(r'modules\beta_vae_5channels.pt')

opt = optim.Adam(v.parameters(), lr = 0.001571)
for epoch in range(1, 100):
    train(v, train_data, criterion, opt, device, epoch, VQ=False)

# test(v, test_data, criterion, device)
# v.cpu()
print(v.parameters)

compare(train_data, v, VQ=False)

experiment(train_data, v, latent_dims)
# experiment_VQ(test_data, v, latent_dims)