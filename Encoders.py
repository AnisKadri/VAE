#!/usr/bin/env python
# coding: utf-8

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def lin_size(n, num_layers, first_kernel = None):
    
    for i in range(0, num_layers):
        
        if i == 0 and first_kernel != None:
            n = 1 + ((n - first_kernel) // 2)
        else:
            n = 1 + ((n - 2) // 2)
            
    if n <= 0:
        raise ValueError("Window Length is too small in relation to the number of Layers")
            
    return n * 2 * num_layers

class TCVAE_Encoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(TCVAE_Encoder, self).__init__()   
        
        self.n =  lin_size(L, num_layers, first_kernel)        
        self.cnn_layers = nn.ModuleList()
        self.n_channels = input_size
        self.latent_dims = latent_dims
        
        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)
                
        # CNN Layers that double the channels each time
        for i in range(0, num_layers):            
            if i == 0:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(nn.Conv1d(input_size, input_size * 2, kernel_size=first_kernel, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2))
            else:                
                self.cnn_layers.append(nn.Conv1d(input_size * 2 * i, input_size * 2 * (i+1), kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i+1)))
                
        # MLP Layers for Mu and logvar output
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.n * input_size, latent_dims),
#             nn.ReLU(True)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.n * input_size, latent_dims),
#             nn.ReLU(True)
        )   
        
        #Init CNN
        self.cnn_layers.apply(init_weights)
         
    def forward(self, x):
        ### CNN
        for i, cnn in enumerate(self.cnn_layers):
            x = cnn(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        
        ### MLP
        mu = torch.clamp(self.encoder_mu(x), min=-200, max=200)
        logvar = torch.clamp(self.encoder_logvar(x), min=-5, max=2)
        # print(mu.shape)
        mu = mu.view(mu.shape[0], self.n_channels, -1)
        logvar = logvar.view(logvar.shape[0], self.n_channels, -1)
        
        return mu, logvar
    


class LongShort_TCVAE_Encoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(LongShort_TCVAE_Encoder, self).__init__()
        self.latent_dims = latent_dims
        self.short_encoder = TCVAE_Encoder(input_size, num_layers, latent_dims, L, slope, first_kernel= None)
        self.long_encoder = TCVAE_Encoder(input_size, num_layers, latent_dims, L, slope, first_kernel)

    def forward(self, x):        
        short_mu, short_logvar = self.short_encoder(x)      # BCZ
        long_mu, long_logvar = self.long_encoder(x)         # BCZ

        mu = torch.cat((short_mu, long_mu), axis=1)         # B.2C.Z
        logvar = torch.cat((short_logvar, long_logvar), axis=1) #B.2C.Z
        return mu, logvar


class MST_Encoder(nn.Module):
    def __init__(self, n_channels, num_layers, slope=0.2, first_kernel=None):
        super(MST_Encoder, self).__init__()

        self._n_channels = n_channels
        self._num_layers = num_layers
        self._slope = slope
        self._first_kernel = first_kernel

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        self.cnn_layers = nn.ModuleList()

        for i in range(0, self._num_layers):
            # Kernel for firs layer is 2 if short or 'first_kernel' for long
            if i == 0:
                if self._first_kernel == None: self._first_kernel = 2
                self.cnn_layers.append(
                    nn.Conv1d(self._n_channels, self._n_channels, kernel_size=self._first_kernel, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(self._slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(self._n_channels))
            # Last layer outputs 2* n_channels
            elif i == self._num_layers - 1:
                self.cnn_layers.append(
                    nn.Conv1d(self._n_channels, 2 * self._n_channels, kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(self._slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(2 * self._n_channels))
            # CNN Layers that don#t change n_channels
            else:
                self.cnn_layers.append(
                    nn.Conv1d(self._n_channels, self._n_channels, kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(self._slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(self._n_channels))

                # Init Layers
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
        ### CNN Shape BLC
        for i, cnn in enumerate(self.cnn_layers):
            # print(x.shape)
            x = cnn(x)
        return x


class MST_VAE_Encoder(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(MST_VAE_Encoder, self).__init__()  
        
        self._n_channels = n_channels
        self._num_layers = num_layers
        self._slope = slope
        self._first_kernel = first_kernel
        
        
        self.short_encoder = MST_Encoder(self._n_channels, self._num_layers, self._slope, first_kernel= None)
        self.long_encoder = MST_Encoder(self._n_channels, self._num_layers, self._slope, self._first_kernel)    
        
        self.reduction_layer = nn.Conv1d(2*self._n_channels,self._n_channels, kernel_size=1, stride=1, padding=0)
        
        self.encoder_mu = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        self.encoder_logvar = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(self._slope)

        
        torch.nn.init.kaiming_uniform_(self.reduction_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.encoder_mu.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.encoder_logvar.weight, mode="fan_in", nonlinearity="leaky_relu")


    def forward(self, x):
        
        x_short = self.short_encoder(x)       # B.2C.Ls
        x_long = self.long_encoder(x)         # B.2C.Ll
        
        x_cat = torch.cat((x_short, x_long), dim = 2)     # B.2C.L_sl
        x_red = self.reduction_layer(x_cat)               # B.C.L_sl
        x_red = self.leakyrelu(x_red)                     # B.C.L_sl
        
        mu = self.encoder_mu(x_red)                       # B.C.L_sl
        logvar = torch.clamp(self.encoder_logvar(x_red), min=-5, max = 2)    # B.C.L_sl
    
        # print("Input Shape", x.shape)
        # print("x_short Shape", x_short.shape)
        # print("x_long Shape", x_long.shape)
        # print("Cat Shape", x_cat.shape)
        # print("After Reduction Shape", x_red.shape)
        # print("mu Shape", mu.shape)
        # print("logvar Shape", logvar.shape)
     
        return mu, logvar


class MST_VAE_Encoder_Linear(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(MST_VAE_Encoder_Linear, self).__init__()

        self.n = lin_size(L, num_layers, first_kernel)// (2*num_layers) + lin_size(L, num_layers, None)// (2*num_layers)
        self._n_channels = n_channels
        self._num_layers = num_layers
        self._latent_dims = latent_dims
        self._slope = slope
        self._first_kernel = first_kernel

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        self.short_encoder = MST_Encoder(self._n_channels, self._num_layers, self._slope, first_kernel=None)
        self.long_encoder = MST_Encoder(self._n_channels, self._num_layers, self._slope, self._first_kernel)

        self.reduction_layer = nn.Conv1d(2 * self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)

        # MLP Layers for Mu and logvar output
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.n * self._n_channels, self._latent_dims),
            #             nn.ReLU(True)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.n * self._n_channels, self._latent_dims),
            #             nn.ReLU(True)
        )
        self.leakyrelu = nn.LeakyReLU(self._slope)

        torch.nn.init.kaiming_uniform_(self.reduction_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.encoder_mu.apply(init_weights)
        self.encoder_logvar.apply(init_weights)

    def forward(self, x):
        x_short = self.short_encoder(x)  # B.2C.Ls
        x_long = self.long_encoder(x)  # B.2C.Ll

        x_cat = torch.cat((x_short, x_long), dim=2)  # B.2C.L_sl
        x_red = self.reduction_layer(x_cat)  # B.C.L_sl
        x_red = self.leakyrelu(x_red)  # B.C.L_sl
        x_red= x_red.view(x_red.shape[0], -1)

        ### MLP
        mu = self.encoder_mu(x_red)
        logvar = torch.clamp(self.encoder_logvar(x_red), min=-5, max=2)
        mu = mu.view(mu.shape[0], self._n_channels, -1)
        logvar = logvar.view(logvar.shape[0], self._n_channels, -1)
        # mu = self.encoder_mu(x_red)  # B.C.L_sl
        # logvar = torch.clamp(self.encoder_logvar(x_red), min=-5, max=2)  # B.C.L_sl

        # print("Input Shape", x.shape)
        # print("x_short Shape", x_short.shape)
        # print("x_long Shape", x_long.shape)
        # print("Cat Shape", x_cat.shape)
        # print("After Reduction Shape", x_red.shape)
        # print("mu Shape", mu.shape)
        # print("logvar Shape", logvar.shape)

        return mu, logvar
    
class MST_VAE_Encoder_dist(nn.Module):
    def __init__(self, n_channels, num_layers, slope = 0.2, first_kernel = None):
        super(MST_VAE_Encoder_dist, self).__init__()  
        
        self._n_channels = n_channels
        self._num_layers = num_layers
        self._slope = slope
        self._first_kernel = first_kernel
        
        
        self.short_encoder = MST_Encoder(self._n_channels, self._num_layers, self._slope, first_kernel= None)
        self.long_encoder = MST_Encoder(self._n_channels, self._num_layers, self._slope, self._first_kernel)    
        
        self.reduction_layer = nn.Conv1d(2*self._n_channels,self._n_channels, kernel_size=1, stride=1, padding=0)
        
        self.encoder_mu = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        self.encoder_logvar = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(self._slope)

        
        torch.nn.init.kaiming_uniform_(self.reduction_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.encoder_mu.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.encoder_logvar.weight, mode="fan_in", nonlinearity="leaky_relu")


    def forward(self, x):
        
        x_short = self.short_encoder(x)       
        x_long = self.long_encoder(x)   
        
        x_cat = torch.cat((x_short, x_long), dim = 2)                
        x_red = self.reduction_layer(x_cat)
        x_red = self.leakyrelu(x_red)  
        
        mu = self.encoder_mu(x_red)
        logvar = torch.clamp(self.encoder_logvar(x_red), min=-5, max = 2)
    
#         print("Input Shape", x.shape)
#         print("x_short Shape", x_short.shape)
#         print("x_long Shape", x_long.shape)
#         print("Cat Shape", x_cat.shape)
#         print("After Reduction Shape", x_red.shape)
#         print("mu Shape", mu.shape)
#         print("logvar Shape", logvar.shape)        
     

        z_dist = D.Normal(loc=mu, scale=torch.exp(0.5*logvar))
        return {'z': z_dist}


class RnnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(RnnEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dims = latent_dims

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=slope)

        self.encoder_mu = nn.Sequential(
            nn.Linear(hidden_size, latent_dims),
            #             nn.ReLU(True)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(hidden_size, latent_dims),
            #             nn.ReLU(True)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Initialize the hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through the LSTM layer
        x, (hn, cn) = self.lstm(x, (h0, c0))

        ### MLP
        mu = self.encoder_mu(x[:, -1, :])
        logvar = self.encoder_logvar(x[:, -1, :])

        # Return the output and final hidden and cell states
        return mu, logvar