import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dims, v_encoder, v_decoder, L = 30, slope = 0.2, first_kernel = None):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = v_encoder(input_size, hidden_size, num_layers, latent_dims, L, slope, first_kernel)
        self.decoder = v_decoder(input_size, hidden_size, num_layers, latent_dims, L, slope, first_kernel)
        
    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrization_trick(mu, logvar)
        return self.decoder(z), mu, logvar   
    
# CodeBook Dim: K x D
# D is the number of Channels
# K is the number of vectors which is the number of features for each channels
# Example: I have 3 Channels MTS and want for each channel 5 features then K = 5, D = 3
class VQ_Quantizer(nn.Module):
    def __init__(self, num_embed, dim_embed, commit_loss):
        super(VQ_Quantizer, self).__init__()  
        
        self._num_embed = num_embed
        self._dim_embed = dim_embed
        self._commit_loss = commit_loss
        
        self._embedding = nn.Embedding(self._num_embed, self._dim_embed)
        self._embedding.weight.data.uniform_(-1/self._num_embed, 1/self._num_embed)        

    def forward(self, x):
        # x : BCL -> BLC
#         print(x.shape)
        x = x.permute(0,2,1).contiguous()
        x_shape = x.shape
        
        # flaten the input to have the Channels as embedding space
        x_flat = x.view(-1, self._dim_embed)
        
        # Calculate the distance to embeddings
        
#         print("the non squared x", x_flat.shape )
#         print("the non squared embed weights", self._embedding.weight.t().shape)
#         print("the x ", torch.sum(x_flat**2, dim = 1, keepdim = True).shape)
#         print("the embed ", torch.sum(self._embedding.weight**2, dim = 1).shape)
#         print("the matmul ", torch.matmul(x_flat, self._embedding.weight.t()).shape)
        dist = (torch.sum(x_flat**2, dim = 1, keepdim = True) 
               + torch.sum(self._embedding.weight**2, dim = 1)
               - 2 * torch.matmul(x_flat, self._embedding.weight.t()))
#         print(dist.shape)
        
        embed_indices = torch.argmin(dist, dim = 1).unsqueeze(1)
#         print(embed_indices)
        embed_Matrix = torch.zeros_like(dist)
#         print(embed_Matrix.shape)
        embed_Matrix.scatter_(1, embed_indices, 1)
#         print("Embedding ", embed_Matrix[:10,:])

        # get the corresponding e vectors
        quantizer = torch.matmul(embed_Matrix, self._embedding.weight).view(x_shape)    
#         print("quantizer ", quantizer.shape)
        
        # Loss
        first_loss = F.mse_loss(quantizer, x.detach())
        second_loss = F.mse_loss(quantizer.detach(), x)
        loss = first_loss + self._commit_loss * second_loss
#         print(loss)
        
        # straigh-through gradient
        quantizer = x + (quantizer -x).detach()
        quantizer = quantizer.permute(0,2,1).contiguous()
#         print("quantizer ", quantizer.shape)       
     
        return quantizer, loss
    
class VQ_MST_VAE(nn.Module):
    def __init__(self, n_channels, num_layers, latent_dims, v_encoder, v_decoder, v_quantizer,
                 L = 30,
                 slope = 0.2,
                 first_kernel = None,
                 commit_loss = 0.25
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
        
        
        self.encoder = self._v_encoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope, self._first_kernel)
        self.decoder = self._v_decoder(self._n_channels, self._num_layers, self._latent_dims, self._L, self._slope, self._first_kernel)
        self.quantizer = self._v_quantizer(self._latent_dims, self._n_channels, self._commit_loss)
        
    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)                
        
        z = self.reparametrization_trick(mu, logvar)
        
        e, loss_quantize = self.quantizer(z)

#         mu_dec, logvar_dec = self.decoder(e)
#         x_rec = self.reparametrization_trick(mu_dec, mu_dec)
        x_rec = self.decoder(e)
        loss_rec = F.mse_loss(x_rec[:,:,0], x[:,:,0], reduction='sum')
        loss = loss_rec + loss_quantize
#         print("----------------Encoder Output-------------")
#         print("mu and logvar", mu.shape, logvar.shape)
#         print("----------------Reparametrization-------------")
#         print("Z", z.shape)
#         print("----------------Quantizer-------------")
#         print("quantized shape", e.shape)
#         print("loss shape", loss_quantize)
#         print("----------------Decoding-------------")
#         print("----------------Decoder Output-------------")
#         print("mu and logvar Decoder", mu_dec.shape, logvar_dec.shape)
#         print("rec shape", x_rec.shape)
        return x_rec[:,:,0], loss, mu, logvar 