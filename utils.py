#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# import matplotlib.gridspec as gridspec


# In[ ]:

@torch.no_grad()
def compare(dataset, model):
    model.eval()
    rec = []
    x = []

    for i, data in enumerate(dataset):
        x_rec, mu, logvar = model(data)
        z = model.reparametrization_trick(mu, logvar)

        x.extend(data[:, :, 0].detach().numpy())
        rec.extend(x_rec[:].detach().numpy())

    plt.plot(rec, "r--")
    plt.plot(x[:], "b-")
    plt.ylim(0, 100)
    plt.grid(True)


@torch.no_grad()
def sample_from_data(model, data, n, latent_dims):
    # Get necessary variables for the init
    latent_dims = latent_dims*2
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
            Z = model.reparametrization_trick(_mu, _logvar)

            # Fill the Tensors with data Shape (mu, logvar,z): n*T*Latent_dim      x_rec = n*T*C
            mu[j, i * batch_size: (i + 1) * batch_size, :]      = _mu
            logvar[j, i * batch_size: (i + 1) * batch_size, :]  = _logvar
            z[j, i * batch_size: (i + 1) * batch_size, :]       = Z
            x_rec[j, i * batch_size: (i + 1) * batch_size, :]   = rec

        x[i * batch_size: (i + 1) * batch_size, :]              = data[:, :, 0] # Shape T*C

    # Calculate the mean for mu, logvar, z and x_rec
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
        z[:, slider_idx] = slider_val                                 # Replace the Z by the value from the slider
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
    x, mu, logvar, z, x_rec, x_rec_mean = sample_from_data(model, data, 100, latent_dims)
    print("x {}, mu {}, logvar {}, z {}, x_rec {} ".format(x.shape, mu.shape, logvar.shape, z.shape, x_rec.shape))
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

        fig.canvas.draw_idle()
        plt.grid(True)

    # Connect the sliders to the update function
    for i, slider in enumerate(sliders):
        slider.on_changed(lambda val, idx=i: update(val, idx, mu, logvar, z, sliders))

    # Show the plot
    plt.show()
    plt.grid(True)
