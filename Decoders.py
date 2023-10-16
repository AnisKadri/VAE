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
    def __init__(self, args, first_kernel=None):
        super(TCVAE_Decoder, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                m.bias.data.fill_(0.01)

        self.n_channels = args.n_channels
        self.latent_dims = args.latent_dims
        self.first_kernel = first_kernel
        self.num_layers = args.num_layers
        self.slope = args.slope
        self.L = args.L
        self.n = lin_size(self.L, self.num_layers, self.first_kernel)

        self.modified = args.modified
        self.reduction = args.reduction
        self.cnn_layers = nn.ModuleList()
        self.decoder_input = self.n_channels
        if not self.reduction:
            self.decoder_input = self.decoder_input*2

        if self.first_kernel == None:
            self.first_kernel = 2
        else:
            self.first_kernel = self.first_kernel
        
        input_lin = self.latent_dims * 2 ** (self.num_layers ) + self.first_kernel - 2

        if self.modified:
            self.cnn_input = self.decoder_input * self.num_layers * 2
            # CNN Layers that double the channels each time
            for i in range(0, self.num_layers):
                if i == 0:
                    if first_kernel == None: first_kernel = 2
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input, self.cnn_input // 2, kernel_size=2, stride=2, padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // 2))
                elif i == self.num_layers - 1:
                    if first_kernel == None: first_kernel = 2
                    if self.reduction:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input // (2 * i), self.cnn_input // (2 * (i + 1)),
                                               kernel_size=first_kernel, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // (2 * (i + 1))))
                    else:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input // (2 * i), self.cnn_input // (4 * (i + 1)),
                                               kernel_size=first_kernel, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // (4 * (i + 1))))
                else:
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input // (2 * i), self.cnn_input // (2 * (i + 1)), kernel_size=2,
                                           stride=2, padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input // (2 * (i + 1))))
        else:
            self.cnn_input = self.n_channels
            for i in range(0, self.num_layers):
                if i == 0:
                    if first_kernel == None: first_kernel = 2
                    if self.reduction:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input, self.cnn_input, kernel_size=2, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))
                    else:
                        self.cnn_layers.append(
                            nn.ConvTranspose1d(self.cnn_input * 2, self.cnn_input, kernel_size=2, stride=2, padding=0))
                        self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                        self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))
                elif i == self.num_layers - 1:
                    if first_kernel == None: first_kernel = 2
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input, self.cnn_input, kernel_size=first_kernel, stride=2,
                                           padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))
                else:
                    self.cnn_layers.append(
                        nn.ConvTranspose1d(self.cnn_input, self.cnn_input, kernel_size=2, stride=2, padding=0))
                    self.cnn_layers.append(nn.LeakyReLU(self.slope, True))
                    self.cnn_layers.append(nn.BatchNorm1d(self.cnn_input))

        self.decoder_lin = nn.Linear(input_lin, self.L)
        # MLP Layers for Mu and logvar output
        self.decoder_mu = nn.Linear(input_lin, self.L)
        self.decoder_logvar = nn.Linear(input_lin, self.L)

        # Init CNN
        self.cnn_layers.apply(init_weights)

    def forward(self, x):
#         print("Decoder Input ", x.shape)
        #         x = x.view(x.shape[0], x.shape[1]* x.shape[2], -1)
        #         print("Decoder Input reshaped ",x.shape)
        #         x = self.decoder_lin(z)
        #         print("Decoder Input after lin ",x.shape)
        #         x = x.view(x.shape[0],x.shape[1],1)
        #         print("Decoder Input reshaped again",x.shape)

        for i, cnn in enumerate(self.cnn_layers):
#             print("Decoder Cnn ", x.shape)
            x = cnn(x)
        ########################################################################## Put back for old decoder
        #         #         print("Decoder shape after Cnn, should be reshaped? ", x.shape)
        #         x = self.decoder_lin(x)
        #         #         print("Decoder after lin ", x.shape)

        #         return x
        ###################################################################################################################
        cnn_shape = x.shape
#         print("Decoder after Cnn ", x.shape)
#         if not self.modified:
#             x = x.view(x.size(0), -1)
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
    def __init__(self, args):
        super(LongShort_TCVAE_Decoder, self).__init__()

        self._modified = args.modified
        self._reduction = args.reduction
        self.n_channels = args.n_channels
        self.first_kernel= args.first_kernel


        self.longshort_decoder = TCVAE_Decoder(args, first_kernel=self.first_kernel)

        self.short_decoder = TCVAE_Decoder(args, first_kernel=None)

        self.reduction_layer = nn.Conv1d(2 * self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)

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


class MST_Decoder(nn.Module):
    def __init__(self, args, first_kernel=None):
        super(MST_Decoder, self).__init__()

        self._n_channels = args.n_channels
        self._num_layers = args.num_layers
        self._slope = args.slope
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
            elif i == self.num_layers - 1:
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
            # print(x.shape)
            x = cnn(x)
        return x

class MST_VAE_Decoder(nn.Module):
    def __init__(self, args):
        super(MST_VAE_Decoder, self).__init__()  
        
        self._n_channels = args.n_channels
        self._num_layers = args.num_layers
        self._slope = args.slope
        self._first_kernel = args.first_kernel
        
        # Long-Short layers
        self.short_decoder = MST_Decoder(args, first_kernel=None)
        self.long_decoder = MST_Decoder(args, first_kernel=self._first_kernel)
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
    def __init__(self, args):
        super(MST_VAE_Decoder_Linear, self).__init__()

        self.n = lin_size(args.L, args.num_layers, args.first_kernel) // \
                 (2*args.num_layers) + lin_size(args.L, args.num_layers, None) // (2*args.num_layers)


        self._n_channels = args.n_channels
        self._num_layers = args.num_layers
        self._slope = args.slope
        self._first_kernel = args.first_kernel

        # Linear layer
        self.decoder_lin = nn.Linear(self.latent_dims, self._n_channels * 2 * self._num_layers)
        # Long-Short layers
        self.short_decoder = MST_Decoder(args, first_kernel=None)
        self.long_decoder = MST_Decoder(args, first_kernel=self._first_kernel)
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
        # print("decoder input ", x.shape)
        x = x.view(x.shape[0], -1)
        # print("decoder input reshaped ", x.shape)
        x = self.decoder_lin(x)
        # print("decoder input after lin ", x.shape)
        # x = x.view(x.shape[0], self._n_channels, -1)
        x = x.view(x.shape[0],self._n_channels, -1)
        # print("decoder input reshaped after lin ", x.shape)

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
    def __init__(self, args):
        super(MST_VAE_Decoder_dist, self).__init__()  
        
        self._n_channels = args.n_channels
        self._num_layers = args.num_layers
        self._slope = args.slope
        self._first_kernel = args.first_kernel
        
        # Long-Short layers
        self.short_decoder = MST_Decoder(args, first_kernel=None)
        self.long_decoder = MST_Decoder(args, first_kernel=self._first_kernel)
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
    def __init__(self, args, modified=False, reduction = False):
        super(RnnDecoder, self).__init__()

        self.n_channels = args.n_channels
#         self.hidden_size = hidden_size
        self.latent_dims = args.latent_dims
        self.num_layers = args.num_layers
        self.L = args.L
        self.first_kernel = args.first_kernel
        self.modified = modified
        self.reduction = reduction
        self.n = lin_size(args.L, args.num_layers, args.first_kernel)
        
        self.decoder_input = self.n_channels
        factor = 2
        
        if self.modified:
            factor = 2**self.num_layers
            
        if self.reduction:
            factor = factor//2
            
        self.hidden_size = self.n_channels * factor * self.latent_dims


        # Define the linear layer for the latent space to hidden state
        self.latent_to_hidden = nn.Linear(self.latent_dims, self.hidden_size * self.num_layers)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_channels, self.hidden_size, self.num_layers, batch_first=True, dropout=args.slope)

        #         # Define the output layer
        self.fc = nn.Linear(self.hidden_size * self.latent_dims, self.n_channels)

    def forward(self, z):
        # Map the latent space vector to the initial hidden and cell states

#         hidden = self.latent_to_hidden(z)
        print("input to decoder", z.shape)
#         print(self.latent_dims, self.hidden_size)

        z = z.view(z.shape[0], -1).unsqueeze(-1)
        print("reshape z", z.shape)

        # Generate sequence of output tensors
        outputs = []
        #         x = torch.zeros((hidden.size(0), 1, self.input_size), device=hidden.device)
#         x = torch.zeros(z.shape[0], self.L, self.n_channels)
        x = torch.zeros(z.shape[0], self.hidden_size, self.n_channels)
        print("the new x", x.shape)
        hidden = z.permute(1, 0, 2)
        hidden = torch.zeros(self.num_layers, z.shape[0], self.n_channels)
        print("the hidden", hidden.shape)
        cell = torch.zeros_like(hidden)
        output, (hidden, cell) = self.lstm(z, (hidden,cell))

#         output, (hidden, cell) = self.lstm(x, (hidden, cell))

        # Pass the output through the output layer
        output = self.fc(output)
        #         print(output.shape)
        output = output.permute(0, 2, 1)

        return output[:, :, 0]


