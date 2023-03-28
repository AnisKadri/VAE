#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions


# In[ ]:


def lin_size(n, num_layers, first_kernel = None):
    
    for i in range(1, num_layers+1):
        
        if i == 1 and first_kernel != None:
            n = 1 + ((n - first_kernel) // 2)
        else:
            n = 1 + ((n - 2) // 2)
            
    if n <= 0:
        raise ValueError("Window Length is too small in relation to the number of Layers")
            
    return n * 2 * num_layers


# In[ ]:


class TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(TCVAE_Decoder, self).__init__()  
        self.input_size = input_size
        
        self.n =  lin_size(L, num_layers, first_kernel)
        
        self.decoder_lin = nn.Linear(latent_dims, input_size * 2 * num_layers)
        
        self.cnn_layers = nn.ModuleList()
        
        for i in range(num_layers, 0, -1):
            
            if i == 1:
                if first_kernel == None: first_kernel = 2
                self.cnn_layers.append(nn.ConvTranspose1d(input_size * 2, input_size, kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
            else:                
                self.cnn_layers.append(nn.ConvTranspose1d(input_size * 2 * i, input_size * 2 * (i-1), kernel_size=2, stride=2, padding=0))
                self.cnn_layers.append(nn.LeakyReLU(slope, True))
                self.cnn_layers.append(nn.BatchNorm1d(input_size * 2 * (i-1)))
                
        
    def forward(self, z):

        x = self.decoder_lin(z) 
        x = x.view(x.shape[0],x.shape[1],1)
#             x = self.cnn_layers()
        for i, cnn in enumerate(self.cnn_layers):
            x = cnn(x)

        return x[:,:,0]  


# In[3]:


class LongShort_TCVAE_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(LongShort_TCVAE_Decoder, self).__init__()  
        
        self.longshort_decoder = TCVAE_Decoder(input_size, num_layers, 2 * latent_dims, L, slope, first_kernel)
        
    def forward(self, z):
        return self.longshort_decoder(z)


# In[ ]:

class RnnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, L = 30, slope = 0.2, first_kernel = None):
        super(RnnDecoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.L = L
        
        # Define the linear layer for the latent space to hidden state
        self.latent_to_hidden = nn.Linear(latent_dims, hidden_size*num_layers)
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = slope)
        
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
        
        return output[:,:,0]

#         for i in range(self.input_size):
#             # Forward pass through the LSTM layer
# #             print(hidden.shape)
#             output, (hidden, cell) = self.lstm(x, (hidden, cell))
            
#             # Pass the output through the output layer
#             output = self.fc(output)
# #             print(output.shape)
            
#             # Append the output tensor to the list of outputs
#             outputs.append(output)
            
#             # Update the input tensor to use the predicted output as the next input
#             x = outputs[-1]
# #             print(x.shape)
        
#         print(output_tensor[:,:,0, -1].shape)
        # Stack the list of output tensors into a single tensor
#         output_tensor = torch.stack(outputs, dim=1)
#         print(output_tensor[:,:,0, -1].shape)

        # Return the output tensor
#         return output_tensor[:,:,0, -1]






