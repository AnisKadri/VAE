#!/usr/bin/env python
# coding: utf-8


import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

def lin_size(n, num_layers, first_kernel = None):
    
    for i in range(1, num_layers+1):
        
        if i == 1 and first_kernel != None:
            n = 1 + ((n - first_kernel) // 2)
        else:
            n = 1 + ((n - 2) // 2)
            
    if n <= 0:
        raise ValueError("Window Length is too small in relation to the number of Layers")
            
    return n * 2 * num_layers

class TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(TCVAE_Decoder, self).__init__() 
        
        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)
                
        self.input_size = input_size
        
        self.n =  lin_size(L, num_layers, first_kernel)
        
        self.decoder_lin = nn.Linear(latent_dims, input_size * 2 * num_layers)
        
        self.cnn_layers = nn.ModuleList()
        
        # CNN Layers that double the channels each time
        for i in range(num_layers, 0, -1):            
            if i == 1:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(nn.ConvTranspose1d(input_size * 2, input_size, kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
            else:                
                self.cnn_layers.append(nn.ConvTranspose1d(input_size * 2 * i, input_size * 2 * (i-1), kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i-1)))               
        
        # Init CNN
        self.cnn_layers.apply(init_weights)
                
        
    def forward(self, z):

        z = z.view(z.shape[0], -1)
        x = self.decoder_lin(z)
        x = x.view(x.shape[0],x.shape[1],1)
       
        for i, cnn in enumerate(self.cnn_layers):
            x = cnn(x)

        return x[:,:,0]

class LongShort_TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(LongShort_TCVAE_Decoder, self).__init__()  
        
        self.longshort_decoder = TCVAE_Decoder(input_size, num_layers, 2 * latent_dims, L, slope, first_kernel)
        
    def forward(self, z):
        return self.longshort_decoder(z) # z: B.C.Z -> B.C.L_new (L_new is a new window length that depends on the Conv layers in the decoder, L_new = 8 for latent dims = 5)

class MST_Decoder(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(MST_Decoder, self).__init__()

        self._n_channels = n_channels
        self._num_layers = num_layers
        self._slope = slope
        self._first_kernel = first_kernel

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        self.cnn_layers = nn.ModuleList()

        # CNN Layers that double the channels each time
        # for i in range(num_layers, 0, -1):
        #     if i == 1:
        #         if first_kernel == None: first_kernel = 2
        #         self.cnn_layers.append(
        #             nn.ConvTranspose1d(self._n_channels * 2, self._n_channels * 2, kernel_size=2, stride=2, padding=0))
        #         self.cnn_layers.append(nn.LeakyReLU(slope, True))
        #     else:
        #         self.cnn_layers.append(
        #             nn.ConvTranspose1d(self._n_channels * 2 * i, self._n_channels * 2 * (i - 1), kernel_size=2, stride=2,
        #                                padding=0))
        #         self.cnn_layers.append(nn.LeakyReLU(slope, True))
        #         self.cnn_layers.append(nn.BatchNorm1d(self._n_channels * 2 * (i - 1)))
        for i in range(0, self._num_layers):
            # First kernel is 2 for short and 'first_kernel' for long
            if i == 0:
                if self._first_kernel == None: self._first_kernel = 2
                self.cnn_layers.append(
                    nn.ConvTranspose1d(self._n_channels, self._n_channels, kernel_size=self._first_kernel, stride=1,
                                       padding=0))
                self.cnn_layers.append(nn.LeakyReLU(self._slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(self._n_channels))
            # Last Layer doubles n_channels
            elif i == num_layers - 1:
                self.cnn_layers.append(
                    nn.ConvTranspose1d(self._n_channels, 2 * self._n_channels, kernel_size=2, stride=1, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(self._slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(2 * self._n_channels))
            # Adds Conv Layers
            else:
                self.cnn_layers.append(
                    nn.ConvTranspose1d(self._n_channels, self._n_channels, kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(self._slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(self._n_channels))
                # Init CNN
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
        ### CNN Shape BCL
        for i, cnn in enumerate(self.cnn_layers):
            #             print(x.shape)
            x = cnn(x)
        return x

class MST_VAE_Decoder(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(MST_VAE_Decoder, self).__init__()  
        
        self._n_channels = n_channels
        self._num_layers = num_layers
        self._slope = slope
        self._first_kernel = first_kernel
        
        # Long-Short layers
        self.short_decoder = MST_Decoder(self._n_channels, self._num_layers, self._slope, first_kernel= None)
        self.long_decoder = MST_Decoder(self._n_channels, self._num_layers, self._slope, self._first_kernel)    
        # Reduction Layer
        self.reduction_layer = nn.ConvTranspose1d(2 *self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        # mu and logvar output layers
        self.decoder_mu = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        self.decoder_logvar = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        # Activation layer for the Reduction
        self.leakyrelu = nn.LeakyReLU(self._slope)
        # Init layers
        torch.nn.init.kaiming_uniform_(self.reduction_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.decoder_mu.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.decoder_logvar.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):

        x_short_in = x#[:,:,:2]   # Either take the concatenated L_sl or split it into Ls and Ll
        x_long_in = x#[:,:,3:]    # Either take the concatenated L_sl or split it into Ls and Ll
        
        x_short = self.short_decoder(x_short_in)        # B.2C.L_dec
        x_long = self.long_decoder(x_long_in)           # B.2C.L_dec
        
        x_cat = torch.cat((x_short, x_long), dim = 2)   # B.2C.L_2dec
        x_red = self.reduction_layer(x_cat)             # B.C.L_2dec
        x_red = self.leakyrelu(x_red)                   # B.C.L_2dec

        mu = self.decoder_mu(x_red)                     # B.C.L_2dec
        logvar = torch.clamp(self.decoder_logvar(x_red), min=-5, max = 2)  # B.C.L_2dec
        
        # print("Input Shape", x.shape)
        # print("x_short Shape", x_short.shape)
        # print("x_long Shape", x_long.shape)
        # print("Cat Shape", x_cat.shape)
        # print("After Reduction Shape", x_red.shape)
        # print("mu Shape", mu.shape)
        # print("logvar Shape", logvar.shape)
        return x_red[:,:,0]    # Output x_red directly or output reconstruction mu and logvar
        # return mu, logvar


class MST_VAE_Decoder_Linear(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(MST_VAE_Decoder_Linear, self).__init__()

        self.n = lin_size(L, num_layers, first_kernel)// (2*num_layers) + lin_size(L, num_layers, None)// (2*num_layers)


        self._n_channels = n_channels
        self._num_layers = num_layers
        self._slope = slope
        self._first_kernel = first_kernel

        # Linear layer
        self.decoder_lin = nn.Linear(latent_dims, self._n_channels * 2 * self._num_layers)
        # Long-Short layers
        self.short_decoder = MST_Decoder(self._n_channels, self._num_layers, self._slope, first_kernel=None)
        self.long_decoder = MST_Decoder(self._n_channels, self._num_layers, self._slope, self._first_kernel)
        # Reduction Layer
        self.reduction_layer = nn.ConvTranspose1d(2 * self._n_channels, self._n_channels, kernel_size=1, stride=1,
                                                  padding=0)
        # mu and logvar output layers
        self.decoder_mu = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        self.decoder_logvar = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        # Activation layer for the Reduction
        self.leakyrelu = nn.LeakyReLU(self._slope)
        # Init layers

        torch.nn.init.kaiming_uniform_(self.decoder_lin.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.reduction_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.decoder_mu.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.decoder_logvar.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x_short_in = x  # [:,:,:2]   # Either take the concatenated L_sl or split it into Ls and Ll
        x_long_in = x  # [:,:,3:]    # Either take the concatenated L_sl or split it into Ls and Ll
        x = x.view(x.shape[0], -1)
        x = self.decoder_lin(x)
        # x = x.view(x.shape[0], self._n_channels, -1)
        x = x.view(x.shape[0],x.shape[1],1)

        x_short = self.short_decoder(x)  # B.2C.L_dec
        x_long = self.long_decoder(x)  # B.2C.L_dec

        x_cat = torch.cat((x_short, x_long), dim=2)  # B.2C.L_2dec
        x_red = self.reduction_layer(x_cat)  # B.C.L_2dec
        x_red = self.leakyrelu(x_red)  # B.C.L_2dec

        mu = self.decoder_mu(x_red)  # B.C.L_2dec
        logvar = torch.clamp(self.decoder_logvar(x_red), min=-5, max=2)  # B.C.L_2dec

        # print("Input Shape", x.shape)
        # print("x_short Shape", x_short.shape)
        # print("x_long Shape", x_long.shape)
        # print("Cat Shape", x_cat.shape)
        # print("After Reduction Shape", x_red.shape)
        # print("mu Shape", mu.shape)
        # print("logvar Shape", logvar.shape)
        return x_red[:, :, 0]  # Output x_red directly or output reconstruction mu and logvar
        # return mu, logvar

class MST_VAE_Decoder_dist(nn.Module):
    def __init__(self, n_channels, num_layers, slope = 0.2, first_kernel = None):
        super(MST_VAE_Decoder_dist, self).__init__()  
        
        self._n_channels = n_channels
        self._num_layers = num_layers
        self._slope = slope
        self._first_kernel = first_kernel
        
        # Long-Short layers
        self.short_decoder = MST_Decoder(self._n_channels, self._num_layers, self._slope, first_kernel= None)
        self.long_decoder = MST_Decoder(self._n_channels, self._num_layers, self._slope, self._first_kernel)    
        # Reduction Layer
        self.reduction_layer = nn.ConvTranspose1d(2 *self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        # mu and logvar output layers
        self.decoder_mu = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        self.decoder_logvar = nn.Conv1d(self._n_channels, self._n_channels, kernel_size=1, stride=1, padding=0)
        # Activation layer for the Reduction
        self.leakyrelu = nn.LeakyReLU(self._slope)
        # Init layers
        torch.nn.init.kaiming_uniform_(self.reduction_layer.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.decoder_mu.weight, mode="fan_in", nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.decoder_logvar.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):

        x_short_in = x#[:,:,:2]
        x_long_in = x#[:,:,3:]
        
        x_short = self.short_decoder(x_short_in)        
        x_long = self.long_decoder(x_long_in)
        
        x_cat = torch.cat((x_short, x_long), dim = 2)      
        x_red = self.reduction_layer(x_cat)
        x_red = self.leakyrelu(x_red)
        
        mu = self.decoder_mu(x_red)
        logvar = torch.clamp(self.decoder_logvar(x_red), min=-5, max = 2)
        
#         print("Input Shape", x.shape)
#         print("x_short Shape", x_short.shape)
#         print("x_long Shape", x_long.shape)
#         print("Cat Shape", x_cat.shape)
#         print("After Reduction Shape", x_red.shape)
#         print("mu Shape", mu.shape)
#         print("logvar Shape", logvar.shape)        
     
        x_dist = D.Normal(loc=mu, scale=torch.exp(0.5*logvar))
        return {'x': x_dist}


class RnnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, L=30, slope=0.2, first_kernel=None):
        super(RnnDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.L = L

        # Define the linear layer for the latent space to hidden state
        self.latent_to_hidden = nn.Linear(latent_dims, hidden_size * num_layers)

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=slope)

        #         # Define the output layer
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        # Map the latent space vector to the initial hidden and cell states

        hidden = self.latent_to_hidden(z)

        hidden = hidden.view(-1, self.num_layers, self.hidden_size)

        # Generate sequence of output tensors
        outputs = []
        #         x = torch.zeros((hidden.size(0), 1, self.input_size), device=hidden.device)
        x = torch.zeros(hidden.shape[0], self.L, self.input_size)
        hidden = hidden.permute(1, 0, 2)
        cell = torch.zeros_like(hidden)

        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        # Pass the output through the output layer
        output = self.fc(output)
        #         print(output.shape)
        output = output.permute(0, 2, 1)

        return output[:, :, 0]


