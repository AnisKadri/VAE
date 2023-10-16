import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions


class Variational_Autoencoder(nn.Module):
    def __init__(self, args, v_encoder, v_decoder):
        super(Variational_Autoencoder, self).__init__()

        self._n_channels = args.n_channels
        self._num_layers = args.num_layers
        self._latent_dims = args.latent_dims
        self._v_encoder = v_encoder
        self._v_decoder = v_decoder
        #         self._v_quantizer = v_quantizer
        self._L = args.L
        self._slope = args.slope
        self._first_kernel = args.first_kernel
        self._ß = args.ß
        self._reduction = args.reduction
        self._modified = args.modified
        #         if self._modified:
        #             self._num_embed = self._n_channels * 4 * self._num_layers
        #         else:
        #             self._num_embed = self._n_channels * 2
        #         if self._reduction:
        #             self._num_embed = self._num_embed // 2

        self.encoder = self._v_encoder(args)
        self.decoder = self._v_decoder(args)

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

#         print("----------------Encoder Output-------------")
#         print("mu and logvar", mu.shape, logvar.shape)
#         print("----------------Reparametrization-------------")
#         print("Z", z.shape)
        # print("----------------Quantizer-------------")
#         print("loss shape", loss_quantize)

        #         mu_dec, logvar_dec = self.decoder(e)
        #         x_rec = self.reparametrization_trick(mu_dec, mu_dec)
        x_rec, mu_rec, logvar_rec = self.decoder(z)

        loss_rec = F.mse_loss(x_rec, x, reduction='sum')
        loss = loss_rec + self._ß * loss_kld

        # print("----------------Decoding-------------")
#         print("----------------Decoder Output-------------")
        # # print("mu and logvar Decoder", mu_dec.shape, logvar_dec.shape)
#         print("x origin", x.shape)
#         print("rec shape", x_rec.shape)
#         print("loss_rec", loss_rec.shape)
#         print("loss_kld", loss_kld.shape)
#         print("loss", loss.shape)
        return x_rec, loss, mu, logvar, mu_rec, logvar_rec, z
    
# CodeBook Dim: K x D
# D is the number of Channels
# K is the number of vectors which is the number of features for each channels
# Example: I have 3 Channels MTS and want for each channel 5 features then K = 5, D = 3
class VQ_Quantizer(nn.Module):
    def __init__(self, args, decay, epsilon=1e-5):
        super(VQ_Quantizer, self).__init__()

        self._num_embed = args.num_embed
        self._dim_embed = args.latent_dims
        self._commit_loss = args.commit_loss

        self._embedding = nn.Embedding(self._num_embed, self._dim_embed)
        self._embedding.weight.data.uniform_(-1 / self._num_embed, 1 / self._num_embed)

        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embed))
        self._ema_w = nn.Parameter(torch.Tensor(self._num_embed, self._dim_embed))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self._std = 0.8

    def forward(self, x):
        # x : BCL -> BLC
#         print("Entering the Quantizer", x.shape)
        x_shape = x.shape
        x = x.permute(0, 2, 1).contiguous()

#         print("Permutation in the Quantizer", x.shape)

        # flaten the input to have the Channels as embedding space
        x_flat = x.view(-1, self._dim_embed)
#         print("Flatten in the Quantizer", x_flat.shape)

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
            embed_indices = new_embed_indices
        # print("noise ",noise.shape)
#         print("The Embeding indices",embed_indices.flatten(), embed_indices.shape)
#         print("Embedding indices reshaped", embed_indices.view(50, -1))
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

        return quantizer, loss, embed_indices

class VQ_Quantizer__spread(nn.Module):
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
    def __init__(self, args, v_encoder, v_decoder, v_quantizer):
        super(VQ_MST_VAE, self).__init__()

        self._n_channels = args.n_channels
        self._num_layers = args.num_layers
        self._num_embed = args.num_embed
        self._latent_dims = args.latent_dims
        self._v_encoder = v_encoder
        self._v_decoder = v_decoder
        self._v_quantizer = v_quantizer
        self._L = args.L
        self._slope = args.slope
        self._first_kernel = args.first_kernel
        self._commit_loss = args.commit_loss
        self._reduction = args.reduction
        self._modified = args.modified
        # if self._modified:
        #     self._num_embed = self._n_channels * 4 * self._num_layers
        # else:
        #     self._num_embed = self._n_channels * 2
        # if self._reduction:
        #     self._num_embed = self._num_embed // 2

        self.encoder = self._v_encoder(args)
        self.decoder = self._v_decoder(args)
        self.quantizer = self._v_quantizer(args, decay=0.99,
                                           epsilon=1e-5)

        self.bn = nn.BatchNorm1d(self._num_embed)

    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        *_, last_layer = self.decoder.children()
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e2).detach()
        return 0.8 * λ

    def forward(self, x, split_loss=False, ouput_indices=False):
        mu, logvar = self.encoder(x)

        z = self.reparametrization_trick(mu, logvar)
#         print(x.shape)
#         z = self.bn(self.quantizer._embedding.weight[None,:])
#         is_larger = torch.all(torch.gt(z[0], self.quantizer._embedding.weight))
#         print(z.shape)
#         print("Is encoder output larger than the set of vectors?", is_larger)
        e, loss_quantize, indices = self.quantizer(z)

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
        loss_rec = F.mse_loss(x_rec-1, x-1, reduction='sum')
        loss = loss_rec + loss_quantize

#         print("----------------Decoding-------------")
#         print("----------------Decoder Output-------------")
#         print("mu and logvar Decoder", mu_dec.shape, logvar_dec.shape)
#         print("rec shape", x_rec.shape)
        if split_loss == True:
            return x_rec, loss_rec, loss_quantize, mu, logvar, mu_rec, logvar_rec, e
        if ouput_indices == True:
            return x_rec, loss, mu, logvar, mu_rec, logvar_rec, e, indices        
        return x_rec, loss, mu, logvar, mu_rec, logvar_rec, e

    # In[12]: