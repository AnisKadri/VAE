


from Encoders import LongShort_TCVAE_Encoder, RnnEncoder, MST_VAE_Encoder, MST_VAE_Encoder_dist
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder, MST_VAE_Decoder, MST_VAE_Decoder_dist
# from vae import VQ_MST_VAE, VQ_Quantizer

import torch; torch.manual_seed(955)
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from dataGen import Gen
import pprint
import numpy as np
import matplotlib.pyplot as plt

def lin_size(n, num_layers, first_kernel=None):
    for i in range(0, num_layers):

        if i == 0 and first_kernel != None:
            n = 1 + ((n - first_kernel) // 2)
        else:
            n = 1 + ((n - 2) // 2)

    if n <= 0:
        raise ValueError("Window Length is too small in relation to the number of Layers")

    return n * 2 * num_layers


class TCVAE_Encoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(TCVAE_Encoder, self).__init__()

        self.n = lin_size(L, num_layers, first_kernel)
        self.cnn_layers = nn.ModuleList()
        self.n_channels = input_size

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        # CNN Layers that double the channels each time
        for i in range(0, num_layers):
            if i == 0:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(
                    nn.Conv1d(input_size, input_size * 2, kernel_size=first_kernel, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2))
            else:
                self.cnn_layers.append(
                    nn.Conv1d(input_size * 2 * i, input_size * 2 * (i + 1), kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i + 1)))

        # MLP Layers for Mu and logvar output
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.n * input_size, latent_dims * self.n_channels),
            #             nn.Linear(self.n * input_size, self.n * input_size),
            #             nn.ReLU(True)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.n * input_size, latent_dims * self.n_channels),
            #             nn.Linear(self.n * input_size, self.n * input_size),
            #             nn.ReLU(True)
        )

        # Init CNN
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
        ### CNN
        for i, cnn in enumerate(self.cnn_layers):
            # print("Encoder Cnn", x.shape)
            x = cnn(x)
        cnn_shape = x.shape
        x = x.view(x.size(0), -1)
        # print("Encoder reshape after Cnn ", x.shape)
        ### MLP
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        # print("Encoder mu after lin ", mu.shape)
        mu = mu.view(mu.shape[0], self.n_channels, -1)
        logvar = logvar.view(logvar.shape[0], self.n_channels, -1)
        # print("Encoder mu after reshape ", mu.shape)
        # print("Calculated n", self.n)
                # mu.reshape

        return mu, logvar

class TCVAE_Encoder_modified(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(TCVAE_Encoder_modified, self).__init__()

        self.n = lin_size(L, num_layers, first_kernel)
        self.cnn_layers = nn.ModuleList()
        self.n_channels = input_size
        self.L = L
        self.num_layers = num_layers
        self.cnn_output = self.n // (2 * self.num_layers)
        self.latent_dims = latent_dims

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        # CNN Layers that double the channels each time
        for i in range(0, num_layers):
            if i == 0:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(
                    nn.Conv1d(input_size, input_size * 2, kernel_size=first_kernel, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2))
            else:
                self.cnn_layers.append(
                    nn.Conv1d(input_size * 2 * i, input_size * 2 * (i + 1), kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i + 1)))

        # MLP Layers for Mu and logvar output
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.cnn_output, self.latent_dims),
            #             nn.Linear(self.n * input_size, self.n * input_size),
            #             nn.ReLU(True)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.cnn_output, self.latent_dims),
            #             nn.Linear(self.n * input_size, self.n * input_size),
            #             nn.ReLU(True)
        )

        # Init CNN
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
        ### CNN
        for i, cnn in enumerate(self.cnn_layers):
            # print("Encoder Cnn", x.shape)
            x = cnn(x)
        cnn_shape = x.shape
        # x = x.view(x.size(0), -1)
        # print("Encoder reshape after Cnn ", x.shape)
        # ### MLP
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        # print("Encoder mu after lin ", mu.shape)
        # mu = mu.view(mu.shape[0], self.n_channels, -1)
        # logvar = logvar.view(logvar.shape[0], self.n_channels, -1)
        # print("Encoder mu after reshape ", mu.shape)
                # mu.reshape

        return mu, logvar

class TCVAE_Encoder_unified(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None, modified=True):
        super(TCVAE_Encoder_unified, self).__init__()

        self.n = lin_size(L, num_layers, first_kernel)
        self.cnn_layers = nn.ModuleList()
        self.n_channels = input_size
        self.L = L
        self.num_layers = num_layers
        self.latent_dims = latent_dims
        self.modified = modified

        if self.modified:
            self.lin_input = self.n // (2 * self.num_layers)
            self.lin_output = self.latent_dims
        else:
            self.lin_input = self.n * self.n_channels
            self.lin_output = self.latent_dims * self.n_channels

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        # CNN Layers that double the channels each time
        for i in range(0, num_layers):
            if i == 0:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(
                    nn.Conv1d(input_size, input_size * 2, kernel_size=first_kernel, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2))
            else:
                self.cnn_layers.append(
                    nn.Conv1d(input_size * 2 * i, input_size * 2 * (i + 1), kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i + 1)))

        # MLP Layers for Mu and logvar output
        self.encoder_mu = nn.Linear(self.lin_input, self.lin_output)
        self.encoder_logvar = nn.Linear(self.lin_input, self.lin_output)

        # Init CNN
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
        ### CNN
        for i, cnn in enumerate(self.cnn_layers):
            #             print("Encoder Cnn", x.shape)
            x = cnn(x)
        cnn_shape = x.shape
        #         print("Encoder after Cnn ", x.shape)
        if not self.modified:
            x = x.view(x.size(0), -1)
        #             print("Encoder reshape after Cnn ", x.shape)
        # ### MLP
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        #         print("Encoder mu after lin ", mu.shape)
        if not self.modified:
            mu = mu.view(mu.shape[0], self.n_channels, -1)
            logvar = logvar.view(logvar.shape[0], self.n_channels, -1)
        #             print("Encoder mu after reshape ", mu.shape)
        # mu.reshape

        return mu, logvar

class LongShort_TCVAE_Encoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None, modified=True,
                 reduction=False):
        super(LongShort_TCVAE_Encoder, self).__init__()
        self.latent_dims = latent_dims
        self.n_channels = input_size
        self._modified = modified
        self._reduction = reduction
        if modified:
            self.red_input = 2 * 2 * input_size * num_layers
        else:
            self.red_input = self.n_channels * 2

        self.short_encoder = TCVAE_Encoder_unified(input_size, num_layers, latent_dims, L, slope, first_kernel=None,
                                                   modified=self._modified)
        self.long_encoder = TCVAE_Encoder_unified(input_size, num_layers, latent_dims, L, slope, first_kernel,
                                                  modified=self._modified)

        self.reduction_layer = nn.Conv1d(self.red_input, self.red_input // 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short_mu, short_logvar = self.short_encoder(x)
        long_mu, long_logvar = self.long_encoder(x)

        mu = torch.cat((short_mu, long_mu), axis=1)
        logvar = torch.cat((short_logvar, long_logvar), axis=1)

        # print("Short Encoder mu: ", short_mu.shape)
        # print("Long Encoder mu: ", long_mu.shape)
        #
        # print("After Cat: ", mu.shape)
        if self._reduction:
            mu = self.reduction_layer(mu)
            logvar = self.reduction_layer(logvar)

        return mu, logvar


class VQ_Quantizer(nn.Module):
    def __init__(self, num_embed, dim_embed, commit_loss, decay, epsilon=1e-5):
        super(VQ_Quantizer, self).__init__()

        self._num_embed = num_embed
        self._dim_embed = dim_embed
        self._commit_loss = commit_loss

        self._embedding = nn.Embedding(self._num_embed, self._dim_embed)
        self._embedding.weight.data.uniform_(-1 / self._num_embed, 1 / self._num_embed)

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embed))
        self._ema_w = nn.Parameter(torch.Tensor(num_embed, dim_embed))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self._std = 0.8

    def forward(self, x):
        # x : BCL -> BLC
        #         print(x.shape)
        x_shape = x.shape
        x = x.permute(0, 2, 1).contiguous()

        #         print(x.shape)

        # flaten the input to have the Channels as embedding space
        x_flat = x.view(-1, self._dim_embed)
        #         print(x_flat.shape)

        # Calculate the distance to embeddings

        #         print("the non squared x", x_flat.shape )
        #         print("the non squared embed weights", self._embedding.weight.t().shape)
        #         print("the x ", torch.sum(x_flat**2, dim = 1, keepdim = True).shape)
        #         print("the embed ", torch.sum(self._embedding.weight**2, dim = 1).shape)
        #         print("the matmul ", torch.matmul(x_flat, self._embedding.weight.t()).shape)
        dist = (torch.sum(x_flat ** 2, dim=1, keepdim=True)
                + torch.sum(self._embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(x_flat, self._embedding.weight.t()))
        #         print(dist.shape)

        embed_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        #         print("embed indices",embed_indices)
        if self.training:
            noise = torch.randn(embed_indices.shape) * self._std
            noise = torch.round(noise).to(torch.int32).to(embed_indices)
            new_embed_indices = embed_indices + noise
            new_embed_indices = torch.clamp(new_embed_indices, max=self._num_embed - 1, min=0)
            # embed_indices = new_embed_indices
        # print("noise ",noise.shape)
        # print("both together",new_embed_indices)
        embed_Matrix = torch.zeros_like(dist)
        #         embed_Matrix = torch.zeros(embed_indices.shape[0], self._num_embed).to(x)
        #         print(embed_Matrix.shape)
        embed_Matrix.scatter_(1, embed_indices, 1)
        #         print("embed_indices", embed_indices)
        #         print("Embedding ", embed_Matrix.shape, embed_Matrix)
        #         print("codebook", self._embedding.weight.shape, self._embedding.weight)

        # get the corresponding e vectors
        quantizer = torch.matmul(embed_Matrix, self._embedding.weight)
        #         print("the quantizer", quantizer.shape, quantizer)
        quantizer = quantizer.view(x_shape).permute(0, 2, 1).contiguous()
        #         print("the quantizer", quantizer.shape, quantizer)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(embed_Matrix, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embed * self._epsilon) * n)

            dw = torch.matmul(embed_Matrix.t(), x_flat)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        #         print("quantizer ", quantizer.shape)

        # Loss
        #         first_loss = F.mse_loss(quantizer, x.detach())
        #         second_loss = F.mse_loss(quantizer.detach(), x)
        #         loss = first_loss + self._commit_loss * second_loss

        # Loss EMA
        e_loss = F.mse_loss(quantizer.detach(), x)
        loss = self._commit_loss * e_loss
        #         print(loss)

        # straigh-through gradient
        quantizer = x + (quantizer - x).detach()
        quantizer = quantizer.permute(0, 2, 1).contiguous()
        #         print("quantizer ", quantizer.shape)

        return quantizer, loss


class VQ_MST_VAE(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, v_encoder, v_decoder, v_quantizer,
                 L=30,
                 slope=0.2,
                 first_kernel=None,
                 commit_loss=0.25,
                 modified=True,
                 reduction=False,
                 ):
        super(VQ_MST_VAE, self).__init__()

        self._n_channels = n_channels
        self._num_layers = num_layers
        self._latent_dims = latent_dims
        self._v_encoder = v_encoder
        self._v_decoder = v_decoder
        self._v_quantizer = v_quantizer
        self._L = L
        self._slope = slope
        self._first_kernel = first_kernel
        self._commit_loss = commit_loss
        self._reduction = reduction
        self._modified = modified
        if self._modified:
            self._num_embed = self._n_channels * 4 * self._num_layers
        else:
            self._num_embed = self._n_channels * 2
        if self._reduction:
            self._num_embed = self._num_embed // 2

        self.encoder = self._v_encoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope,
                                       self._first_kernel, self._modified, self._reduction)
        self.decoder = self._v_decoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope,
                                       self._first_kernel, self._modified, self._reduction)
        self.quantizer = self._v_quantizer(self._num_embed, self._latent_dims, self._commit_loss, decay=0.99,
                                           epsilon=1e-5)

        self.bn = nn.BatchNorm1d(self._num_embed)

    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparametrization_trick(mu, logvar)
        # print(z.shape)
        # z = self.bn(self.quantizer._embedding.weight[None,:])
        # is_larger = torch.all(torch.gt(z[0], self.quantizer._embedding.weight))
        # print(z.shape)
        # print("Is encoder output larger than the set of vectors?", is_larger)
        e, loss_quantize = self.quantizer(z)

#         print("----------------Encoder Output-------------")
#         print("mu and logvar", mu.shape, logvar.shape)
#         print("----------------Reparametrization-------------")
#         print("Z", z.shape)
#         print("----------------Quantizer-------------")
#         print("quantized shape", e.shape)
#         print("loss shape", loss_quantize)

        #         mu_dec, logvar_dec = self.decoder(e)
        #         x_rec = self.reparametrization_trick(mu_dec, mu_dec)
        x_rec, mu_rec, logvar_rec = self.decoder(e)
        loss_rec = F.mse_loss(x_rec, x, reduction='sum')
        loss = loss_rec + loss_quantize

#         print("----------------Decoding-------------")
#         print("----------------Decoder Output-------------")
#         # print("mu and logvar Decoder", mu_dec.shape, logvar_dec.shape)
#         print("rec shape", x_rec.shape)
        return x_rec, loss, mu, logvar, mu_rec, logvar_rec

    # In[12]:


class TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None, modified=True, reduction= True):
        super(TCVAE_Decoder, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        self.n_channels = input_size
        self.latent_dims = latent_dims
        self.first_kernel = first_kernel
        self.num_layers = num_layers
        self.n = lin_size(L, num_layers, first_kernel)

        self.modified = modified
        self.reduction = reduction
        self.cnn_layers = nn.ModuleList()
        self.decoder_input = self.n_channels
        if not self.reduction:
            self.decoder_input = self.decoder_input*2

        if self.first_kernel == None:
            first_kernel = 2
        else:
            first_kernel = self.first_kernel
        input_lin = self.latent_dims * 2 ** (self.num_layers - 1) + first_kernel - 2

        if self.modified:
            self.cnn_input = self.decoder_input * self.num_layers * 2
            print(self.n_channels)
            print(self.cnn_input)
            # CNN Layers that double the channels each time
            for i in range(0, num_layers):
                if i == 0:
                    if first_kernel == None: first_kernel = 2
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input, self.cnn_input // 2, kernel_size=2, stride=2, padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // 2))
                elif i == num_layers - 1:
                    if first_kernel == None: first_kernel = 2
                    if self.reduction:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input // (2 * i), self.cnn_input // (2 * (i + 1)),
                                               kernel_size=first_kernel, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // (2 * (i + 1))))
                    else:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input // (2 * i), self.cnn_input // (4 * (i + 1)),
                                               kernel_size=first_kernel, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // (4 * (i + 1))))
                else:
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input // (2 * i), self.cnn_input // (2 * (i + 1)), kernel_size=2,
                                           stride=2, padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // (2 * (i + 1))))
        else:
            self.cnn_input = self.n_channels
            for i in range(0, num_layers):
                if i == 0:
                    if first_kernel == None: first_kernel = 2
                    if self.reduction:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input, self.cnn_input, kernel_size=2, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))
                    else:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input * 2, self.cnn_input, kernel_size=2, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))
                elif i == num_layers - 1:
                    if first_kernel == None: first_kernel = 2
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input, self.cnn_input, kernel_size=first_kernel, stride=2,
                                           padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))
                else:
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input, self.cnn_input, kernel_size=2, stride=2, padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))

        self.decoder_lin = nn.Linear(input_lin, L)
        # MLP Layers for Mu and logvar output
        self.decoder_mu = nn.Linear(input_lin, L)
        self.decoder_logvar = nn.Linear(input_lin, L)

        # Init CNN
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
        # print("Decoder Input ", x.shape)
        #         x = x.view(x.shape[0], x.shape[1]* x.shape[2], -1)
        #         print("Decoder Input reshaped ",x.shape)
        #         x = self.decoder_lin(z)
        #         print("Decoder Input after lin ",x.shape)
        #         x = x.view(x.shape[0],x.shape[1],1)
        #         print("Decoder Input reshaped again",x.shape)

        for i, cnn in enumerate(self.cnn_layers):
            # print("Decoder Cnn ", x.shape)
            x = cnn(x)
        ########################################################################## Put back for old decoder
        #         #         print("Decoder shape after Cnn, should be reshaped? ", x.shape)
        #         x = self.decoder_lin(x)
        #         #         print("Decoder after lin ", x.shape)

        #         return x
        ###################################################################################################################
        cnn_shape = x.shape
        #         print("Decoder after Cnn ", x.shape)
        if not self.modified:
            x = x.view(x.size(0), -1)
        #             print("Decoder reshape after Cnn ", x.shape)
        # ### MLP
        mu = self.decoder_mu(x)
        logvar = self.decoder_logvar(x)
        #         print("Encoder mu after lin ", mu.shape)
        if not self.modified:
            mu = mu.view(mu.shape[0], self.n_channels, -1)
            logvar = logvar.view(logvar.shape[0], self.n_channels, -1)
        #             print("Encoder mu after reshape ", mu.shape)
        # mu.reshape

        return mu, logvar


class LongShort_TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None,
                 modified=True, reduction=False):
        super(LongShort_TCVAE_Decoder, self).__init__()

        self._modified = modified
        self._reduction = reduction
        # if not self._reduction:
        #     input_size = input_size * 2
        # print("input to decoder", input_size)

        self.longshort_decoder = TCVAE_Decoder(input_size, num_layers, 2 * latent_dims, L, slope,
                                               first_kernel, modified=self._modified, reduction=self._reduction)

        self.short_decoder = TCVAE_Decoder(input_size, num_layers, 2 * latent_dims, L, slope,
                                           first_kernel=None, modified=self._modified, reduction=self._reduction)

        self.reduction_layer = nn.Conv1d(2 * input_size, input_size, kernel_size=1, stride=1, padding=0)

    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z):
        ################################### old decoder ############################################################
        #         x_long = self.longshort_decoder(z)
        #         x_short = self.short_decoder(z)
        #         # print("x_long ", x_long.shape)
        #         # print("x_short ", x_short.shape)

        #         x = torch.cat((x_short, x_long), dim=1)  # B.2C.L
        #         # print("x_cat ", x.shape)
        #         # if self._reduction:
        #         x = self.reduction_layer(x)
        #         # print("x_red ", x.shape)
        #         return x
        #################################################################################################################
        short_mu, short_logvar = self.short_decoder(z)
        long_mu, long_logvar = self.longshort_decoder(z)

        mu = torch.cat((short_mu, long_mu), axis=1)
        logvar = torch.cat((short_logvar, long_logvar), axis=1)

        #         print("Short Decoder mu: ", short_mu.shape)
        #         print("Long Decoder mu: ", long_mu.shape)

        #         print("After Cat: ", mu.shape)
        #         if self._reduction:
        mu = self.reduction_layer(mu)
        logvar = self.reduction_layer(logvar)
        x = self.reparametrization_trick(mu, logvar)
        #         print("Reconstruction at the end: ", x.shape)
        return x, mu, logvar


class slidingWindow(Dataset):
    def __init__(self, data, L):
        self.data = data
        self.L = L

    def __getitem__(self, index):
        if self.data.shape[-1] - index >= self.L:
            x = self.data[..., index:index + self.L]
            v = torch.sum(x / self.L, axis=(self.data.dim() - 1), keepdim=True)
            x = x / v
            return (x, v)

    def __len__(self):
        return self.data.shape[-1] - self.L
        # if self.data.dim() == 2:
        #     return self.data.shape[1] - self.L
        # else:
        #     return self.data.shape[0] - self.L


### Cost function
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


def train(model, train_loader, criterion, optimizer, device, epoch, VQ=True):
    model.train()
    train_loss = 0

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
            loss = criterion(x_rec * v, data[:, :, 0], mu, logvar)
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

    return train_loss / len(train_loader.dataset)


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

            x.extend((data * v)[:, :, 0].detach().numpy())
            rec.extend(((x_rec * v)[:, :, 0]).detach().numpy())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rec, "r--")
    ax.plot(x[:], "b-")
    plt.ylim(50, 600)
    plt.grid(True)
    plt.show()


class Variational_Autoencoder(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, v_encoder, v_decoder,
                 L=30,
                 slope=0.2,
                 first_kernel=None,
                 ß=0.25,
                 modified=True,
                 reduction=False,
                 ):
        super(Variational_Autoencoder, self).__init__()

        self._n_channels = n_channels
        self._num_layers = num_layers
        self._latent_dims = latent_dims
        self._v_encoder = v_encoder
        self._v_decoder = v_decoder
        #         self._v_quantizer = v_quantizer
        self._L = L
        self._slope = slope
        self._first_kernel = first_kernel
        self._ß = ß
        self._reduction = reduction
        self._modified = modified
        #         if self._modified:
        #             self._num_embed = self._n_channels * 4 * self._num_layers
        #         else:
        #             self._num_embed = self._n_channels * 2
        #         if self._reduction:
        #             self._num_embed = self._num_embed // 2

        self.encoder = self._v_encoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope,
                                       self._first_kernel, self._modified, self._reduction)
        self.decoder = self._v_decoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope,
                                       self._first_kernel, self._modified, self._reduction)

    #         self.quantizer = self._v_quantizer(self._num_embed, self._latent_dims, self._commit_loss, decay=0.99,
    #                                            epsilon=1e-5)

    #         self.bn = nn.BatchNorm1d(self._num_embed)

    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        z = self.reparametrization_trick(mu, logvar)
        # print(z.shape)
        # z = self.bn(self.quantizer._embedding.weight[None,:])
        # is_larger = torch.all(torch.gt(z[0], self.quantizer._embedding.weight))
        # print(z.shape)
        # print("Is encoder output larger than the set of vectors?", is_larger)
        #         e, loss_quantize = self.quantizer(z)

        # print("----------------Encoder Output-------------")
        # print("mu and logvar", mu.shape, logvar.shape)
        # print("----------------Reparametrization-------------")
        # print("Z", z.shape)
        # print("----------------Quantizer-------------")
        # print("quantized shape", e.shape)
        # print("loss shape", loss_quantize)

        #         mu_dec, logvar_dec = self.decoder(e)
        #         x_rec = self.reparametrization_trick(mu_dec, mu_dec)
        x_rec, mu_rec, logvar_rec = self.decoder(z)

        loss_rec = F.mse_loss(x_rec, x, reduction='sum')
        loss = loss_rec + self._ß * loss_kld

        # print("----------------Decoding-------------")
        # print("----------------Decoder Output-------------")
        # # print("mu and logvar Decoder", mu_dec.shape, logvar_dec.shape)
        # print("rec shape", x_rec.shape)
        return x_rec, loss, mu, logvar, mu_rec, logvar_rec


def set_effect(effect):
    if effect == "no_effect":
        effects = {
            "Pulse": {
                "occurances": 0,
                "max_amplitude": 1.5,
                "interval": 40
            },
            "Trend": {
                "occurances": 0,
                "max_slope": 0.005,
                "type": "linear"
            },
            "Seasonality": {
                "occurances": 0,
                "frequency_per_week": (7, 14),  # min and max occurances per week
                "amplitude_range": (5, 20),
            },
            "std_variation": {
                "occurances": 0,
                "max_value": 10,
                "interval": 1000,
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
    elif effect == "trend":
        effects = {
            "Pulse": {
                "occurances": 0,
                "max_amplitude": 1.5,
                "interval": 40
            },
            "Trend": {
                "occurances": 1,
                "max_slope": 0.005,
                "type": "linear"
            },
            "Seasonality": {
                "occurances": 0,
                "frequency_per_week": (7, 14),  # min and max occurances per week
                "amplitude_range": (5, 20),
            },
            "std_variation": {
                "occurances": 0,
                "max_value": 10,
                "interval": 1000,
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
    elif effect == "seasonality":
        effects = {
            "Pulse": {
                "occurances": 0,
                "max_amplitude": 1.5,
                "interval": 40
            },
            "Trend": {
                "occurances": 0,
                "max_slope": 0.005,
                "type": "linear"
            },
            "Seasonality": {
                "occurances": 1,
                "frequency_per_week": (7, 14),  # min and max occurances per week
                "amplitude_range": (5, 20),
            },
            "std_variation": {
                "occurances": 0,
                "max_value": 10,
                "interval": 1000,
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

    return effects
def generate_data(n_channels, effect, L, periode=15, step=5, val=500 ):
    effects = set_effect(effect)
    X = Gen(periode, step, val, n_channels, effects)
    x, params, e_params = X.parameters()
    # pprint.pprint(params)
    # pprint.pprint(e_params)
    # X.show()
    x = torch.FloatTensor(x)

    # x = F.normalize(x, p=2, dim=1)
    n = x.shape[1]
    # L = 30

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
    return X, train_data, test_data


def train_on_effect(model, opt, device, n_channels=1, effect='no_effect', n_samples=10, epochs_per_sample=50):
    L = model._L
    latent_dims = model._latent_dims
    for i in range(n_samples):
        X, train_data, test_data = generate_data(n_channels, effect, L)
        x, params, e_params = X.parameters()

        for epoch in range(1, epochs_per_sample):
            train(model, train_data, criterion, opt, device, epoch, VQ=True)
        save(x, "data", effect, n_channels, latent_dims, L, i)
        save(params, "params", effect, n_channels, latent_dims, L, i)
        save(e_params, "e_params", effect, n_channels, latent_dims, L, i)
        save(model, "model", effect, n_channels, latent_dims, L, i)
    return model, X, train_data

def save(obj, name, effect, n_channels, latent_dims, L, i):
    torch.save(obj, r'modules\vq_vae_{}_{}_{}channels_{}latent_{}window_{}.pt'.format(name, effect, n_channels, latent_dims, L, i))


def get_average_norm_scale(train_data, model):
    n_channels = model._n_channels
    norm = torch.empty(n_channels, 0)

    for i, (data, norm_scale) in enumerate(train_data):
        reshaped_norm = norm_scale.permute(1, 0, 2).flatten(1)
        norm = torch.cat((norm, reshaped_norm), 1)

    avg_norm = torch.mean(norm, dim=1)
    return avg_norm


def get_latent_variables(train_data, model):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_channels = model._n_channels
    latent_dims = model._latent_dims

    latents = torch.empty(n_channels, latent_dims, 0, device=device)

    for i, (data, norm_scale) in enumerate(train_data):
        data = data.to(device)

        mu, logvar = model.encoder(data)
        z = model.reparametrization_trick(mu, logvar)

        reshaped_mu, reshaped_logvar, reshaped_z = (t.permute(1, 2, 0) for t in [mu, logvar, z])

        latents = torch.cat((latents, reshaped_z), 2)

    avg_latents = torch.mean(latents, dim=2)
    return latents, avg_latents