import torch; torch.manual_seed(955)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataGen import Gen
from Encoders import LongShort_TCVAE_Encoder#, RnnEncoder, MST_VAE_Encoder, MST_VAE_Encoder_dist
from Decoders import LongShort_TCVAE_Decoder#, RnnDecoder, MST_VAE_Decoder, MST_VAE_Decoder_dist
from vae import Variational_Autoencoder, VQ_MST_VAE, VQ_Quantizer
from utils import train_on_effect, generate_data, extract_parameters, suppress_prints, add_mu_std
from train import train, train_MCMC, stridedWindow, slidingWindow, criterion

import numpy as np
import matplotlib.pyplot as plt
import pprint
# all parameters for generating the time series should be configured in this cell
periode = 365 #days
step = 5 # mess interval in minutes
val = 500
n_channels = 5
effects = {
    "Pulse": {
        "occurances":12,
        "max_amplitude":1.5,
        "interval":40
        },
    "Trend": {
        "occurances":6,
        "max_slope":0.0002,
        "type":"linear"
        },
    "Seasonality": {
        "occurances":10,
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

### Init Model
latent_dims = 30 # 6 # 17
L= 2016# 39 #32
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# v = Variational_Autoencoder(n_channels = n_channels,
#                             num_layers =  3,#4, #3
#                             latent_dims= latent_dims,
#                             v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
#                             v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
#                             L=L,
#                             slope = 0,
#                             first_kernel = 20, #11, #20
#                             ß = 1.5,
#                             modified=True,
#                             reduction = True)
v = VQ_MST_VAE(n_channels = n_channels,
                            num_layers =  4,#4, #3
                            latent_dims= latent_dims,
                            v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
                            v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
                            v_quantizer = VQ_Quantizer,
                            L=L,
                            slope = 0,
                            first_kernel = 1008, #11, #20
                            commit_loss = 1.5,
                            modified=True,
                            reduction = True) #10 5


v = v.to(device)

opt = optim.Adam(v.parameters(), lr = 0.005043529186448577) # 0.005043529186448577 0.006819850049647945
print(v)

x = torch.FloatTensor(x)
n = x.shape[1]

train_ = x[:, :int(0.8*n)]
val_   = x[:, int(0.8*n):int(0.9*n)]
test_  = x[:, int(0.9*n):]

train_data = DataLoader(stridedWindow(train_, L),# slidingWindow, stridedWindow
                        batch_size= 50,# 59, # 22
                        shuffle = False
                        )
val_data = DataLoader(slidingWindow(val_, L),
                        batch_size=500,
                        shuffle = False
                        )
test_data = DataLoader(slidingWindow(test_, L),
                        batch_size=500,
                        shuffle = False
                        )

labels = extract_parameters(n_channels, e_params, effects)
labels = add_mu_std(labels, params)
print(labels)