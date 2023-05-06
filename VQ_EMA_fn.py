


from Encoders import LongShort_TCVAE_Encoder, RnnEncoder, MST_VAE_Encoder, MST_VAE_Encoder_dist
from Decoders import LongShort_TCVAE_Decoder, RnnDecoder, MST_VAE_Decoder, MST_VAE_Decoder_dist
# from vae import VQ_MST_VAE, VQ_Quantizer

import torch; torch.manual_seed(955)
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
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
            #             print("Encoder Cnn", x.shape)
            x = cnn(x)
        cnn_shape = x.shape
        x = x.view(x.size(0), -1)
        #         print("Encoder reshape after Cnn ", x.shape)
        ### MLP
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        #         print("Encoder mu after lin ", mu.shape)
        mu = mu.view(mu.shape[0], self.n_channels, -1)
        logvar = logvar.view(logvar.shape[0], self.n_channels, -1)
        #         print("Encoder mu after reshape ", mu.shape)
        #         mu.reshape

        return mu, logvar


class LongShort_TCVAE_Encoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(LongShort_TCVAE_Encoder, self).__init__()
        self.latent_dims = latent_dims

        self.short_encoder = TCVAE_Encoder(input_size, num_layers, latent_dims, L, slope, first_kernel=None)
        self.long_encoder = TCVAE_Encoder(input_size, num_layers, latent_dims, L, slope, first_kernel)

    def forward(self, x):
        short_mu, short_logvar = self.short_encoder(x)
        long_mu, long_logvar = self.long_encoder(x)

        mu = torch.cat((short_mu, long_mu), axis=1)
        logvar = torch.cat((short_logvar, long_logvar), axis=1)

        # print("Short Encoder mu: ", short_mu.shape)
        # print("Long Encoder mu: ", long_mu.shape)
        #
        # print("After Cat: ", mu.shape)

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

    def forward(self, x):
        # x : BCL -> BLC
        #         print(x.shape)
        x = x.permute(0, 2, 1).contiguous()
        x_shape = x.shape

        # flaten the input to have the Channels as embedding space
        x_flat = x.view(-1, self._dim_embed)

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
        #         print(embed_indices)
        embed_Matrix = torch.zeros_like(dist)
        #         print(embed_Matrix.shape)
        embed_Matrix.scatter_(1, embed_indices, 1)
        #         print("Embedding ", embed_Matrix[:10,:])

        # get the corresponding e vectors
        quantizer = torch.matmul(embed_Matrix, self._embedding.weight).view(x_shape)

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
                 commit_loss=0.25
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

        self.encoder = self._v_encoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope,
                                       self._first_kernel)
        self.decoder = self._v_decoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope,
                                       self._first_kernel)
        self.quantizer = self._v_quantizer(self._latent_dims, self._n_channels, self._commit_loss, decay=0.99,
                                           epsilon=1e-5)

    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)


        z = self.reparametrization_trick(mu, logvar)

        e, loss_quantize = self.quantizer(z)

        #         mu_dec, logvar_dec = self.decoder(e)
        #         x_rec = self.reparametrization_trick(mu_dec, mu_dec)
        x_rec = self.decoder(e)
        loss_rec = F.mse_loss(x_rec, x, reduction='sum')
        loss = loss_rec + loss_quantize
        #         print("----------------Encoder Output-------------")
        #         print("mu and logvar", mu.shape, logvar.shape)
        #         print("----------------Reparametrization-------------")
        #         print("Z", z.shape)
        #         print("----------------Quantizer-------------")
        #         print("quantized shape", e.shape)
        #         print("loss shape", loss_quantize)
        #         print("----------------Decoding-------------")
        # print("----------------Decoder Output-------------")
        #         print("mu and logvar Decoder", mu_dec.shape, logvar_dec.shape)
        # print("rec shape", x_rec.shape)
        return x_rec, loss, mu, logvar

    # In[12]:


class TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(TCVAE_Decoder, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        self.input_size = input_size
        self.latent_dims = latent_dims
        self.first_kernel = first_kernel
        self.num_layers = num_layers

        self.n = lin_size(L, num_layers, first_kernel)

        self.cnn_layers = nn.ModuleList()

        if self.first_kernel == None:
            first_kernel = 2
        else:
            first_kernel = self.first_kernel
        input_lin = self.latent_dims * 2 ** (self.num_layers - 1) + first_kernel - 2
        #         print(input_lin)
        self.decoder_lin = nn.Linear(input_lin, L)

        #         # CNN Layers that double the channels each time
        #         for i in range(num_layers, 0, -1):
        #             if i == 1:
        #                 if first_kernel == None: first_kernel = 2
        #                 self.cnn_layers.append(nn.ConvTranspose1d(input_size * 2, input_size, kernel_size=2, stride=2, padding=0))
        #                 self.cnn_layers.append(nn.LeakyReLU(slope, True))
        #             else:
        #                 self.cnn_layers.append(nn.ConvTranspose1d(input_size * 2 * i, input_size * 2 * (i-1), kernel_size=2, stride=2, padding=0))
        #                 self.cnn_layers.append(nn.LeakyReLU(slope, True))
        #                 self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i-1)))

        # CNN Layers that double the channels each time
        for i in range(0, num_layers):
            if i == 0:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(
                    nn.ConvTranspose1d(input_size * 2, input_size, kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size))
            elif i == num_layers - 1:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(
                    nn.ConvTranspose1d(input_size, input_size, kernel_size=first_kernel, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size))
            else:
                self.cnn_layers.append(nn.ConvTranspose1d(input_size, input_size, kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size))

        # Init CNN
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
        #         print("Decoder Input ", z.shape)
        #         x = x.view(x.shape[0], x.shape[1]* x.shape[2], -1)
        #         print("Decoder Input reshaped ",x.shape)
        #         x = self.decoder_lin(z)
        #         print("Decoder Input after lin ",x.shape)
        #         x = x.view(x.shape[0],x.shape[1],1)
        #         print("Decoder Input reshaped again",x.shape)

        for i, cnn in enumerate(self.cnn_layers):
            #             print("Decoder Cnn ", x.shape)
            x = cnn(x)
        #         print("Decoder shape after Cnn, should be reshaped? ", x.shape)
        x = self.decoder_lin(x)
        #         print("Decoder after lin ", x.shape)

        return x  # x[:,:,0]


class LongShort_TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(LongShort_TCVAE_Decoder, self).__init__()

        self.longshort_decoder = TCVAE_Decoder(input_size, num_layers, 2 * latent_dims, L, slope, first_kernel)
        self.short_decoder = TCVAE_Decoder(input_size, num_layers, 2 * latent_dims, L, slope, first_kernel=None)

        self.reduction_layer = nn.Conv1d(2 * input_size, input_size, kernel_size=1, stride=1, padding=0)

    def forward(self, z):
        x_long = self.longshort_decoder(z)
        x_short = self.short_decoder(z)
        #         print("x_long ", x_long.shape)
        #         print("x_short ",x_short.shape)

        x_cat = torch.cat((x_short, x_long), dim=1)  # B.2C.L
        #         print("x_cat ",x_cat.shape)

        x_red = self.reduction_layer(x_cat)
        #         print("x_red ",x_red.shape)
        return x_red


class slidingWindow(Dataset):
    def __init__(self, data, L):
        self.data = data
        self.L = L

    def __getitem__(self, index):
        if self.data.shape[1] - index >= self.L:
            x = self.data[:, index:index + self.L]
            v = torch.sum(x / self.L, axis=1).unsqueeze(1)
            #             print(x.shape)
            #             print(v.unsqueeze(1).shape)
            x = x / v
            return (x, v)

    def __len__(self):
        return self.data.shape[1] - self.L


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
        optimizer.zero_grad()

        if VQ:
            x_rec, loss, mu, logvar = model(data)
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
