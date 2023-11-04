import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataGen import Gen, FastGen, Gen2
from Encoders import LongShort_TCVAE_Encoder, RnnEncoder, MST_VAE_Encoder, MST_VAE_Encoder_dist
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder, MST_VAE_Decoder, MST_VAE_Decoder_dist
from vae import Variational_Autoencoder, VQ_MST_VAE, VQ_Quantizer
from utils import *  # train_on_effect, generate_data, extract_parameters, suppress_prints, add_mu_std
from train import *
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pprint

torch.set_printoptions(sci_mode=False)
def save(obj, name, effect, n_channels, latent_dims, L, i):
    torch.save(obj, r'modules\vq_vae_{}_{}_{}channels_{}latent_{}window_{}.pt'.format(name, effect, n_channels, latent_dims, L, i))

# all parameters for generating the time series should be configured in this cell

p = 2
i = 2
effect = "Seasonality"
args = GENV(n_channels=1, latent_dims=5, n_samples=100, shuffle=False, periode=p, L=288 * p, min_max=True,
            num_layers=3, robust=False, first_kernel=288, num_embed=512, modified=False)

effects = {
    "Pulse": {
        "occurances": 1,
        "max_amplitude": 5,
        "interval": 40,
        "start": None
    },
    "Trend": {
        "occurances": 10,
        "max_slope": 0.002,
        "type": "linear",
        "start": None
    },
    "Seasonality": {
        "occurances": 10,
        "frequency_per_week": (14, 21),  # min and max occurances per week
        "amplitude_range": (5, 20),
        "start": -5
    },
    "std_variation": {
        "occurances": 0,
        "max_value": 10,
        "interval": 30,
        "start": None
    },
    "channels_coupling": {
        "occurances": 0,
        "coupling_strengh": 20
    },
    "Noise": {
        "occurances": 0,
        "max_slope": 0.005,
        "type": "linear"
    }
}

# args.num_embed = 1 #args.enc_out**2
vae = Variational_Autoencoder(args,
                            v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
                            v_decoder = LongShort_TCVAE_Decoder #MST_VAE_Decoder,
                           )
vq = VQ_MST_VAE(args,
                v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
                v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
                v_quantizer = VQ_Quantizer) #10 5

vae = vae.to(args.device)
vq = vq.to(args.device)

opt_vae = optim.Adam(vae.parameters(), lr = 0.002043529186448577) # 0.005043529186448577 0.006819850049647945
opt_vq = optim.Adam(vq.parameters(), lr = 0.002043529186448577) # 0.005043529186448577 0.006819850049647945

train_data, val_data, test_data, X = generate_labeled_data(args,
                                                           effects,
                                                           effect=effect,
                                                           occurance=4,
                                                           return_gen=True,
                                                           anomalies=False)
x, params, e_params = X.parameters()

np.set_printoptions(suppress=True)
VAE_losses = []
VQ_losses = []

for epoch in range(1, 200):
    loss_vae = train(vae, train_data, args, opt_vae, epoch)
    loss_vq = train(vq, train_data, args, opt_vq, epoch)

    # if epoch % 10 == 1:
        # display.clear_output(wait=True)
        # show_results(vae, train_data, args)
        # show_results(vq, train_data, args, vq=True)

    VAE_losses.append(loss_vae)
    VQ_losses.append(loss_vq)

    print('====> VAE: Sample {} Average loss: {:.4f}'.format(epoch, loss_vae / len(train_data.dataset)))
    print('====> VQ: Sample {} Average loss: {:.4f}'.format(epoch, loss_vq / len(train_data.dataset)))

save(x, "data", effect, args.n_channels, args.latent_dims, args.L, i)
save(params, "params", effect, args.n_channels, args.latent_dims, args.L, i)
save(e_params, "e_params", effect, args.n_channels, args.latent_dims, args.L, i)
save(vae, "vae", effect, args.n_channels, args.latent_dims, args.L, i)
save(vq, "vq", effect, args.n_channels, args.latent_dims, args.L, i)


# train_data_long, val_data_long, test_data_long, X_long = generate_long_data(args,
#                                                                             effects,
#                                                                             effect=effect,
#                                                                             occurance=2,
#                                                                             return_gen=True,
#                                                                             anomalies=False)
# x_long, params_long, e_params_long = X_long.parameters()
#
#
# args.min_max = False
# np.set_printoptions(suppress=True)
# VAE_losses = []
# VQ_losses = []
#
# for epoch in range(1, 1000):
#     loss_vae = train(vae, train_data_long, args, opt_vae, epoch)
#     loss_vq = train(vq, train_data_long, args, opt_vq, epoch)
#
#     # if epoch % 100 == 1:
#     #     display.clear_output(wait=True)
#     #     show_results_long(vae, train_data_long, args)
#     #     show_results_long(vq, train_data_long, args, vq=True)
#
#     VAE_losses.append(loss_vae)
#     VQ_losses.append(loss_vq)
#
#     print('====> VAE: Sample {} Average loss: {:.4f}'.format(epoch, loss_vae / len(train_data_long.dataset)))
#     print('====> VQ: Sample {} Average loss: {:.4f}'.format(epoch, loss_vq / len(train_data_long.dataset)))