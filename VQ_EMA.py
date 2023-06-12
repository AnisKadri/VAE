#!/usr/bin/env python
# coding: utf-8

# In[11]:
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch; torch.manual_seed(955)
import torch.optim as optim
from VQ_EMA_fn import *

# import optuna
# from optuna.samplers import TPESampler



### Init Model
n_channels = 1
latent_dims = 7 # 6 # 17
L= 600# 39 #32
device = 'cuda' if torch.cuda.is_available() else 'cpu'


v = Variational_Autoencoder(n_channels = n_channels,
                            num_layers =  3,#4, #3
                            latent_dims= latent_dims,
                            v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
                            v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
                            L=L,
                            slope = 0,
                            first_kernel = 60, #11, #20
                            ß = 1.5,
                            modified=False,
                            reduction = True)
# v = VQ_MST_VAE(n_channels = n_channels,
#                             num_layers =  2,#4, #3
#                             latent_dims= latent_dims,
#                             v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
#                             v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
#                             v_quantizer = VQ_Quantizer,
#                             L=L,
#                             slope = 0,
#                             first_kernel = 20, #11, #20
#                             commit_loss = 0.25,
#                             modified= False,
#                             reduction=True
#                ) #10 5

v = v.to(device)
opt = optim.Adam(v.parameters(), lr = 0.005043529186448577) # 0.005043529186448577 0.006819850049647945

v, X, train_data = train_on_effect(v, opt, device, effect='no_effect', n_samples=10)
v, X, train_data = train_on_effect(v, opt, device, effect='trend', n_samples=10)
v, X, train_data = train_on_effect(v, opt, device, effect='seasonality', n_samples=10)


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

            x.extend((data*v)[:,:,0].detach().numpy())
            rec.extend(((x_rec*v)[:,:,0]).detach().numpy())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rec, "r--")
    ax.plot(x[:], "b-")
    # plt.ylim(50,600)
    plt.grid(True)
    plt.show()


# In[23]:


# v.cpu()
# compare(test_data, v, VQ=True)
# v.to(device)


# In[22]:



# In[39]:


# def objective(trial, model, x, criterion_fcn, train_fcn, n_channels, epochs):
#     # Define the hyperparameters to optimize
#     learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-2)
#     num_layers = trial.suggest_int('num_layers', 3, 7)
#     latent_dims = trial.suggest_int('latent_dims', 2, 20)
#     first_kernel = trial.suggest_int('first_kernel', 10, 30)
#     slope = trial.suggest_int('slope', 0.1, 0.3)
#     commit_loss = trial.suggest_int('commit_loss', 0.1, 10)
#     L = trial.suggest_int('L', 30, 512)
#     batch_size = trial.suggest_int('batch_size', 10, 100)
#     ### Init Model
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     n = x.shape[1]
#     train_ = x[:, :int(0.8*n)]
#     val_   = x[:, int(0.8*n):int(0.9*n)]
#     test_  = x[:, int(0.9*n):]
#
#     train_data = DataLoader(slidingWindow(train_, L),
#                             batch_size=batch_size,
#                             shuffle = False
#                             )
#     val_data = DataLoader(slidingWindow(val_, L),
#                             batch_size=batch_size,
#                             shuffle = False
#                             )
#     test_data = DataLoader(slidingWindow(test_, L),
#                             batch_size=batch_size,
#                             shuffle = False
#                             )
#
#     v = VQ_MST_VAE(n_channels = n_channels,
#                             num_layers = num_layers,
#                             latent_dims= latent_dims,
#                             v_encoder = LongShort_TCVAE_Encoder, #MST_VAE_Encoder,
#                             v_decoder = LongShort_TCVAE_Decoder, #MST_VAE_Decoder,
#                             v_quantizer = VQ_Quantizer,
#                             L=L,
#                             slope = slope,
#                             first_kernel = first_kernel,
#                             commit_loss = 10)
#     v = v.to(device)
#     # Define the loss function and optimizer
#     optimizer = optim.Adam(v.parameters(), lr=learning_rate)
#
#     for epoch in range(1, epochs):
#         loss = train_fcn(v, train_data, criterion_fcn, optimizer, device, epoch, VQ=True)
#
#
#     # Return the validation accuracy as the objective value
#     return loss
#

# In[47]:

#
# import optuna
# from optuna.samplers import TPESampler
# # Define the Optuna study and optimize the hyperparameters
# epochs = 50
# study = optuna.create_study(sampler=TPESampler(), direction='minimize')
# study.optimize(lambda trial: objective(trial,
#                                        VariationalAutoencoder,
#                                        x,
#                                        criterion,
#                                        train,
#                                        n_channels,
#                                        epochs
#                                       ),
#                n_trials=50)
#
#
# # In[48]:
#
#
# study.best_trial


# In[ ]:




