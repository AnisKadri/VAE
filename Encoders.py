#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions


# In[2]:


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
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, L = 30, first_kernel = None):
        super(TCVAE_Encoder, self).__init__()   
        
        self.n =  lin_size(L, num_layers, first_kernel)
        
        self.cnn_layers = nn.ModuleList()
        
        for i in range(0, num_layers):
            
            if i == 0:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(nn.Conv1d(input_size, input_size * 2, kernel_size=first_kernel, stride=2, padding=0))
                self.cnn_layers.append(nn.ReLU(True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2))
            else:                
                self.cnn_layers.append(nn.Conv1d(input_size * 2 * i, input_size * 2 * (i+1), kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.ReLU(True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i+1)))
                
        
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.n * input_size, latent_dims),
#             nn.ReLU(True)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.n * input_size, latent_dims),
#             nn.ReLU(True)
        )
        
#         ### Linear section: mean
#         self.encoder_mu = nn.Sequential(
#             nn.Linear(self.n * n_channels*8, latent_dims)
# #             nn.Linear(self.n * n_channels*8, 32*latent_dims),
# #             nn.ReLU(True),
# #             nn.Linear(32*latent_dims, latent_dims)
#         )
        
#         self.encoder_logvar = nn.Sequential(
#             nn.Linear(self.n * n_channels*8, latent_dims)
# #             nn.Linear(self.n * n_channels*8, 32*latent_dims),
# #             nn.ReLU(True),
# #             nn.Linear(32*latent_dims, latent_dims)
#         )
         

    def forward(self, x):
        ### CNN
        for i, cnn in enumerate(self.cnn_layers):
            x = cnn(x)
#         x = self.cnn_layers(x) 
        x = x.view(x.size(0), -1)
        
        ### MLP
        mu = self.encoder_mu(x)  
        logvar = self.encoder_logvar(x)
        
        return mu, logvar


# In[ ]:


class LongShort_TCVAE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, L = 30, first_kernel = None):
        super(LongShort_TCVAE_Encoder, self).__init__()   
        
        self.short_encoder = TCVAE_Encoder(input_size, hidden_size, num_layers, latent_dims, L, first_kernel= None)
        self.long_encoder = TCVAE_Encoder(input_size, hidden_size, num_layers, latent_dims, L, first_kernel)
        
    def forward(self, x):        
        short_mu, short_logvar = self.short_encoder(x)        
        long_mu, long_logvar = self.long_encoder(x)

        mu = torch.cat((short_mu, long_mu), axis=1)
        logvar = torch.cat((short_logvar, long_logvar), axis=1)
        
        return mu, logvar

