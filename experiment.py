import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from VQ_EMA_fn import *
from dataGen import Gen
from utils import  experiment, experiment_VQ, new_experiment_VQ
# from train import slidingWindow, criterion, train, test, objective
# from Encoders import LongShort_TCVAE_Encoder, RnnEncoder
# from Decoders import LongShort_TCVAE_Decoder, RnnDecoder
# from vae import VariationalAutoencoder

# import torch; torch.manual_seed(955)
# import torch.optim as optim
from torch.utils.data import DataLoader
# # import optuna
# # from optuna.samplers import TPESampler


import numpy as np
import matplotlib.pyplot as plt
import pprint

x = torch.load(r'modules\data_18channels.pt')
x = torch.FloatTensor(x)
print(type(x))
v = torch.load(r'modules\vq_ema.pt')
L = v._L
latent_dims = v._latent_dims
batch_size = 22
n = x.shape[1]

train_ = x[:, :int(0.8*n)]
val_   = x[:, int(0.8*n):int(0.9*n)]
test_  = x[:, int(0.9*n):]
train_data = DataLoader(slidingWindow(train_, L),
                        batch_size=batch_size,
                        shuffle = False
                        )
val_data = DataLoader(slidingWindow(val_, L),
                        batch_size=batch_size,
                        shuffle = False
                        )
test_data = DataLoader(slidingWindow(test_, L),
                        batch_size=batch_size,
                        shuffle = False
                        )




# opt = optim.Adam(v.parameters(), lr = 0.001571)
# for epoch in range(1, 100):
#     train(v, train_data, criterion, opt, device, epoch, VQ=True)


# test(v, test_data, criterion, device)
# v.cpu()
print(v.parameters)
print('here')
compare(train_data, v, VQ=True)

# experiment(train_data, v, latent_dims)
# new_experiment_VQ(test_data, v)
# experiment_VQ(train_data, v)
