import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, v_encoder, v_decoder, L = 30, first_kernel = None):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = v_encoder(input_size, hidden_size, num_layers, latent_dims, L, first_kernel)
        self.decoder = v_decoder(input_size, hidden_size, num_layers, latent_dims, L, first_kernel)
        
    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrization_trick(mu, logvar)
        return self.decoder(z), mu, logvar   