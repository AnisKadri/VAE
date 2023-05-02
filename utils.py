#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
import matplotlib.gridspec as gridspec




# In[ ]:


def compare(dataset, model, VQ=True):
    model.eval()
    rec = []
    x = []
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if VQ:
                x_rec, loss, mu, logvar = model(data)
            else:
                x_rec, mu, logvar = model(data)
            z = model.reparametrization_trick(mu, logvar)
            x.extend(data[:,:,0].detach().numpy())
            rec.extend(x_rec.detach().numpy())

    plt.plot(rec, "r--")
    plt.plot(x[:], "b-")
    # plt.ylim(0,100)
    plt.grid(True)
    plt.show()
    
def compare_dist(dataset, encoder, decoder):
    encoder.eval()
    decoder.eval()
    rec = []
    x = []
    with torch.no_grad():
        for i, data in enumerate(dataset):
             # Forward pass through the encoder and compute the latent variables
            qnet = encoder(data)
            
            # Sample from the latent variables and forward pass through the decoder
            z_sample = qnet['z'].rsample()
            pnet = decoder(z_sample)
            x_rec = pnet['x'].rsample()
#             z = model.reparametrization_trick(mu, logvar)

            x.extend(data[:,:,0].detach().numpy())
            rec.extend(x_rec[:,:,0].detach().numpy())
        
    plt.plot(rec, "r--")
    plt.plot(x[:], "b-")
    plt.ylim(0,100)
    plt.grid(True)


# In[ ]:
@torch.no_grad()
def sample_from_data_TC(model, data, n, latent_dims, VQ=True):
    # Get necessary variables for the init
    latent_dims = 2*latent_dims

    n_channels = data.dataset.data.shape[0]
    T = data.dataset.data.shape[1] - data.dataset.L
    batch_size = data.batch_size

    # Init tensors to store results
    x = torch.empty((T, n_channels))
    mu, logvar, z = (torch.empty((n, T, latent_dims)) for _ in range(3))
    x_rec = torch.empty(n, T, n_channels)

    # Loop through data n times
    for i, data in enumerate(data):
        for j in range(n):
            # generate reconstruction and latent space over the x axis
            rec, _mu, _logvar = model(data)
            # rec, _mu, _logvar = model(data)
            # print(rec.shape, _mu.shape)
            Z = model.reparametrization_trick(_mu, _logvar)
            _mu = torch.clamp(_mu, min = -10, max = 10)
            _logvar = torch.clamp(_logvar, min=-10, max=10)
            Z = torch.clamp(Z, min=-10, max=30)
            rec = model.decoder(Z)
            # print(_mu.mean(-1).shape)
            # Fill the Tensors with data Shape (mu, logvar,z): n*T*Latent_dim      x_rec = n*T*C
            mu[j, i * batch_size: (i + 1) * batch_size, :]      = _mu.view(_mu.shape[0], -1)
            logvar[j, i * batch_size: (i + 1) * batch_size, :]  = _logvar.view(_logvar.shape[0], -1)
            z[j, i * batch_size: (i + 1) * batch_size, :]       = Z.view(Z.shape[0], -1)
            x_rec[j, i * batch_size: (i + 1) * batch_size, :]   = rec

        x[i * batch_size: (i + 1) * batch_size, :]              = data[:, :, 0] # Shape T*C

    # Calculate the mean for mu, logvar, z and x_rec
    print(mu.shape)
    mu, logvar, z, x_rec_mean = (torch.mean(t, dim=0) for t in [mu, logvar, z, x_rec])

    # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    x_rec = torch.permute(x_rec, (1, 0, 2))
    x_rec = x_rec.reshape(T, -1)

    # convert to numpy, print shapes and output
    x, mu, logvar, z, x_rec = (t.detach().numpy() for t in [x, mu, logvar, z, x_rec])
    print("Tensors x, mu, logvar, z, x_rec".format(x.shape, mu.shape, logvar.shape, z.shape, x_rec.shape))
    return x, mu, logvar, z, x_rec, x_rec_mean

@torch.no_grad()
def sample_from_z(model, mu, logvar, n, n_channels, slider_idx, slider_val):  #

    # Input to tensor and init the Reconstructions list
    mu, logvar = torch.from_numpy(mu), torch.from_numpy(logvar) # Shape: TC
    slider_val = torch.tensor(slider_val)
    print("mu and logvar",mu.shape, logvar.shape)
    REC = torch.empty((0, mu.shape[0], n_channels))

    # Generate and cat
    for i in range(n):
        z = model.reparametrization_trick(mu, logvar)                 # Shape: TC

        z[:, slider_idx] = slider_val
        # print(z.shape)# Replace the Z by the value from the slider
        rec = model.decoder(z).unsqueeze(dim=0)                       # Shape: TC
        REC = torch.cat((REC, rec), dim=0)                            # Shape: NTC

    # calculate mean
    REC_mean = torch.mean(REC, dim=0)

    # reshape REC   NTC -> TNC and merge TC
    REC = torch.permute(REC, (1, 0, 2))
    print("REC, REC_mean", REC.shape, REC_mean.shape)
    print(REC_mean.shape[0])
    REC = REC.reshape(REC_mean.shape[0], -1)
    return REC.numpy(), REC_mean.numpy()

def experiment(data, model, latent_dims):
    # Init Slider and slider_axs lists
    sliders = []
    slider_axs = []

    # Sample x, x_rec and the latent space
    x, mu, logvar, z, x_rec, x_rec_mean = sample_from_data_TC(model, data, 100, latent_dims)
    print("x {}, mu {}, logvar {}, z {}, x_rec {} ".format(x.shape, mu.shape, logvar.shape, z.shape, x_rec.shape))
    # Create a figure and axis object
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    axs = [ax1, ax2]

    # Plot the initial data
    data_lines = ax1.plot(x, "b")
    rec_lines = ax1.plot(x_rec, "orange", alpha = 0.05)
    rec_lines_mean = ax1.plot(x_rec_mean, "r")
    z_lines = ax2.plot(z, "g")

    # Add a slider widget for each z
    for i in range(z.shape[1]):
        # If statement to place the sliders on the left and right, for the layout
        if i < z.shape[1] / 2:
            slider_axs.append(plt.axes([0.1, (0.3 - 0.02 * i), 0.35, 0.03]))
        else:
            slider_axs.append(plt.axes([0.5, (0.3 - 0.02 * (i - z.shape[1] / 2)), 0.35, 0.03]))

        # populate the sliders list
        sliders.append(Slider(slider_axs[i], r'$Z_{{{}}}$'.format(i), -100, 100, valinit=z[-1, i]))

    # define initial values
    init_vals = [slider.valinit for slider in sliders]

    def update(val, idx, mu, logvar, z, sliders):
        # save z and sample new rec and rec_mean
        temp = np.copy(z)
        temp[:, idx] = sliders[idx].val
        n_channels = x.shape[1]
        rec, rec_mean = sample_from_z(model, mu, logvar, 100, n_channels, idx, val)

        # redraw rec, rec_mean and z
        for channel, line in enumerate(rec_lines):
            line.set_ydata(rec[:, channel])
        for channel, line in enumerate(rec_lines_mean):
            line.set_ydata(rec_mean[:, channel])
        for channel, line in enumerate(z_lines):
            line.set_ydata(temp[:, channel])

        # Reset Sliders to initial values
        for i, slider in enumerate(sliders):
            if slider.val != init_vals[i] and idx != i:
                slider.reset()
                # sliders[idx].val = init_vals[idx]

        fig.canvas.draw_idle()
        # plt.grid(True)


    def save(text):

        print(text)
        torch.save(model, r'modules\{}.pt'.format(text))

    text_ax = plt.axes([0.5, 0.01, 0.35, 0.05])
    text_box = TextBox(text_ax,'Save as: ', initial="beta_vae3")


    # axsave = fig.add_axes([0.4, 0.05, 0.1, 0.05])
    # bsave = Button(axsave, 'Next')
    # bsave.on_clicked(save)


    # Connect the sliders to the update function
    for i, slider in enumerate(sliders):
        slider.on_changed(lambda val, idx=i: update(val, idx, mu, logvar, z, sliders))
    text_box.on_submit(save)
    # text_box.on_submit(lambda model: save(model))
    # Show the plot
    ax1.grid()
    ax1.set_ylim(0,150)
    ax2.grid()
    plt.show()
    # plt.grid(True)

@torch.no_grad()
def sample_from_data_VQ(model, data, n):
    # Get necessary variables for the init
    # latent_dims = latent_dims

    n_channels = data.dataset.data.shape[0]
    latent_dims = n_channels* 2
    T = data.dataset.data.shape[1] - data.dataset.L
    batch_size = data.batch_size

    # Init tensors to store results
    x = torch.empty((T, n_channels))
    mu, logvar, z = (torch.empty((n, T, latent_dims)) for _ in range(3))
    x_rec = torch.empty(n, T, n_channels)

    # Loop through data n times
    for i, data in enumerate(data):
        for j in range(n):
            # generate reconstruction and latent space over the x axis

            rec, loss, _mu, _logvar = model(data)

            # print(_mu.mean(-1).shape)
            Z = model.reparametrization_trick(_mu, _logvar)
            Z, _ = model.quantizer(Z)
            # Fill the Tensors with data Shape (mu, logvar,z): n*T*Latent_dim      x_rec = n*T*C
            mu[j, i * batch_size: (i + 1) * batch_size, :]      = _mu.mean(-1)#.view(_mu.shape[0], -1)
            logvar[j, i * batch_size: (i + 1) * batch_size, :]  = _logvar.mean(-1)#.view(_logvar.shape[0], -1)
            z[j, i * batch_size: (i + 1) * batch_size, :]       = Z.mean(-1)#.view(Z.shape[0], -1)
            x_rec[j, i * batch_size: (i + 1) * batch_size, :]   = rec

        x[i * batch_size: (i + 1) * batch_size, :]              = data[:, :, 0] # Shape T*C

    # Calculate the mean for mu, logvar, z and x_rec
    mu, logvar, z, x_rec_mean = (torch.mean(t, dim=0) for t in [mu, logvar, z, x_rec])

    # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    x_rec = torch.permute(x_rec, (1, 0, 2))
    x_rec = x_rec.reshape(T, -1)

    # convert to numpy, print shapes and output
    x, mu, logvar, z, x_rec = (t.detach().numpy() for t in [x, mu, logvar, z, x_rec])
    print("Tensors x: {}, mu: {}, logvar: {}, z: {}, x_rec: {}".format(x.shape, mu.shape, logvar.shape, z.shape, x_rec.shape))
    return x, mu, logvar, z, x_rec, x_rec_mean

# @torch.no_grad()
# def sample_from_z_VQ(model, mu, logvar, n, slider_idx, slider_val, VQ=True):  #
#     # get the indices of the changed value in quantizers
#     quantizers = model.quantizer._embedding.weight
#     latent_dims, n_channels = quantizers.shape[0], quantizers.shape[1]
#     latent = slider_idx % latent_dims
#     channel = slider_idx // latent_dims
#
#     # Apply the change
#     quantizers[latent, channel] = slider_val
#
#     # Input to tensor and init the Reconstructions list
#     mu, logvar = torch.from_numpy(mu), torch.from_numpy(logvar) # Shape: TC
#     slider_val = torch.tensor(slider_val)
#     print("mu and logvar",mu.shape, logvar.shape)
#     REC = torch.empty((0, mu.shape[0], n_channels))
#     Z = torch.empty((0, mu.shape[0], n_channels))
#
#     # model.quantizer._embedding.weight[:, slider_idx] = slider_val  # Replace the Z by the value from the slider
#
#     # Generate and cat
#     for i in range(n):
#         z = model.reparametrization_trick(mu, logvar).t().unsqueeze(0)                 # Shape: TC ->CT ->BCT to be conform to the input of quantizer
#         print(z.shape)
#         z, loss = model.quantizer(z)
#         print("e", z.shape)
#         rec = model.decoder(z)                     # Shape: TC
#         print("current rec", rec.shape)
#         print("global rec", REC.shape)
#         REC = torch.cat((REC, rec), dim=0)                            # Shape: NTC
#         Z = torch.cat((Z, z), dim=0)
#
#     # calculate mean
#     REC_mean = torch.mean(REC, dim=0)
#     Z = torch.mean(Z, dim=0)
#
#     # reshape REC   NTC -> TNC and merge TC
#     REC = torch.permute(REC, (1, 0, 2))
#     print("REC, REC_mean, Z", REC.shape, REC_mean.shape, Z.shape)
#     print(REC_mean.shape[0])
#     REC = REC.reshape(REC_mean.shape[0], -1)
#     return REC.numpy(), REC_mean.numpy(), Z.numpy()

@torch.no_grad()
def sample_from_quantizer_VQ(model, data, n):
    # Get necessary variables for the init
    # latent_dims = latent_dims

    n_channels = data.dataset.data.shape[0]
    latent_dims = n_channels *2
    T = data.dataset.data.shape[1] - data.dataset.L
    batch_size = data.batch_size

    # Init tensors to store results
    x = torch.empty((T, n_channels))
    z = torch.empty((n, T, latent_dims))
    x_rec = torch.empty(n, T, n_channels)

    # Loop through data n times
    for i, data in enumerate(data):
        for j in range(n):
            # generate reconstruction and latent space over the x axis

            rec, loss, _mu, _logvar = model(data)

            # print(_mu.mean(-1).shape)
            Z = model.reparametrization_trick(_mu, _logvar)
            Z, _ = model.quantizer(Z)
            # Fill the Tensors with data Shape (mu, logvar,z): n*T*Latent_dim      x_rec = n*T*C
            # mu[j, i * batch_size: (i + 1) * batch_size, :]      = _mu.mean(-1)#.view(_mu.shape[0], -1)
            # logvar[j, i * batch_size: (i + 1) * batch_size, :]  = _logvar.mean(-1)#.view(_logvar.shape[0], -1)
            z[j, i * batch_size: (i + 1) * batch_size, :]       = Z.mean(-1)#.view(Z.shape[0], -1)
            x_rec[j, i * batch_size: (i + 1) * batch_size, :]   = rec

        # x[i * batch_size: (i + 1) * batch_size, :]              = data[:, :, 0] # Shape T*C

    # Calculate the mean for mu, logvar, z and x_rec
    z, x_rec_mean = (torch.mean(t, dim=0) for t in [z, x_rec])

    # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    x_rec = torch.permute(x_rec, (1, 0, 2))
    x_rec = x_rec.reshape(T, -1)

    # convert to numpy, print shapes and output
    z, x_rec = (t.detach().numpy() for t in [z, x_rec])
    print("Tensors z: {}, x_rec: {}".format(z.shape, x_rec.shape))
    return z, x_rec, x_rec_mean
@torch.no_grad()
def experiment_VQ(data, model, latent_dims):
    # Init Slider and slider_axs lists
    sliders = []
    slider_axs = []
    n_channels = data.dataset.data.shape[0]
    # Sample x, x_rec and the latent space
    x, mu, logvar, z, x_rec, x_rec_mean = sample_from_data_VQ(model, data, 100)
    print("x {}, x_rec {} ".format(x.shape, x_rec.shape))

    quantizers = model.quantizer._embedding.weight.detach().numpy()
    print(quantizers.shape)
    # Create a figure and axis object
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    axs = [ax1, ax2]

    # Plot the initial data
    data_lines = ax1.plot(x, "b")
    rec_lines = ax1.plot(x_rec, "orange", alpha = 0.2)
    rec_lines_mean = ax1.plot(x_rec_mean, "r")
    z_lines = ax2.plot(z, "g")
    # quantizers = ax2.plot()

    # Add a slider widget for each z
    for i in range(latent_dims):
        for j in range(n_channels):
            # If statement to place the sliders on the left and right, for the layout
            if i < latent_dims*n_channels / 2:
                slider_axs.append(plt.axes([0.1, (0.3 - 0.02 * i), 0.35, 0.03]))
            else:
                slider_axs.append(plt.axes([0.5, (0.3 - 0.02 * (i - latent_dims / 2)), 0.35, 0.03]))

        # populate the sliders list
        sliders.append(Slider(slider_axs[i], r'$Ch_{{{}}} Z_{{{}}}$'.format(j, i), -100, 100, valinit=quantizers[i, j]))

    # define initial values
    init_vals = [slider.valinit for slider in sliders]

    def update(val, idx, mu, logvar, z, sliders):
        # save z and sample new rec and rec_mean
        temp = np.copy(z)
        # n_channels = x.shape[1]
        # get the indices of the changed value in quantizers
        quantizers = model.quantizer._embedding.weight
        latent_dims, n_channels = quantizers.shape[0], quantizers.shape[1]
        latent = idx % latent_dims
        channel = idx // latent_dims

        # Apply the change
        quantizers[latent, channel] = torch.tensor(val, dtype=torch.float)
        # rec, rec_mean, Z = sample_from_z_VQ(model, mu, logvar, 100, idx, val)
        z, rec, rec_mean = sample_from_quantizer_VQ(model, data, 100)
        # redraw rec, rec_mean and z
        for channel, line in enumerate(rec_lines):
            line.set_ydata(rec[:, channel])
        for channel, line in enumerate(rec_lines_mean):
            line.set_ydata(rec_mean[:, channel])
        for channel, line in enumerate(z_lines):
            line.set_ydata(z[:, channel])

        # Reset Sliders to initial values
        for i, slider in enumerate(sliders):
            if slider.val != init_vals[i] and idx != i:
                slider.reset()

        fig.canvas.draw_idle()
        plt.grid(True)
    def save(text):

        print(text)
        torch.save(model, r'modules\{}.pt'.format(text))

    text_ax = plt.axes([0.5, 0.01, 0.35, 0.05])
    text_box = TextBox(text_ax,'Save as: ', initial="beta_vae3")
    # Connect the sliders to the update function
    for i, slider in enumerate(sliders):
        slider.on_changed(lambda val, idx=i: update(val, idx, mu, logvar, z, sliders))
    text_box.on_submit(save)
    # Show the plot
    plt.show()
    plt.grid(True)