import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.distributions


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()        
        
         ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )        
        
        ### Linear section: mean
        self.encoder_mean = nn.Sequential(
            nn.Linear(3 * 3 * 32, 100),
            nn.ReLU(True),
            nn.Linear(100, latent_dims)
        )
        
        ### Linear section: std
        self.encoder_std = nn.Sequential(
            nn.Linear(3 * 3 * 32, 100),            
            nn.ReLU(True),
            nn.Linear(100, latent_dims)
        )   

    def forward(self, x):
        ### CNN
        x = self.encoder_cnn(x)     
        x = x.view(x.size(0), -1)  
        
        ### MLP
        mu = self.encoder_mean(x)  
        sigma = self.encoder_std(x)
        return mu, sigma 

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        
        ### linear decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 100),
            nn.ReLU(True),
            nn.Linear(100, 3 * 3 * 32),
        )        

        ### deconcv layer
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, z):
        z = self.decoder_lin(z)
        z = z.view(z.size(0), 32, 3, 3)
        z = self.decoder_conv(z)
        z = torch.sigmoid(z)
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
        
    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrization_trick(mu, logvar)
        return self.decoder(z), mu, logvar   