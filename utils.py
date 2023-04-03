#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec


# In[ ]:


def compare(dataset, model):
    model.eval()
    rec = []
    x = []
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x_rec, mu, logvar = model(data)
            z = model.reparametrization_trick(mu, logvar)

            x.extend(data[:, :, 0].detach().numpy())
            rec.extend(x_rec[:].detach().numpy())

    plt.plot(rec, "r--")
    plt.plot(x[:], "b-")
    plt.ylim(0, 100)
    plt.grid(True)


# In[ ]:
@torch.no_grad()
def sample_from_data(model, data):
    x, mu, logvar, z, x_rec = [], [], [], [], []

    for i, data in enumerate(data):
        # generate reconstruction and latent space over the x axis
        rec, _mu, _logvar = model(data)
        Z = model.reparametrization_trick(_mu, _logvar)

        # Store the values to lists
        x.extend(data[:, :, 0].detach().numpy())
        mu.extend(_mu[:].detach().numpy())
        logvar.extend(_logvar[:].detach().numpy())
        z.extend(Z[:].detach().numpy())
        x_rec.extend(rec[:].detach().numpy())

    x, mu, logvar, z, x_rec = map(np.array, (x, mu, logvar, z, x_rec))
    return x, mu, logvar, z, x_rec


@torch.no_grad()
def sample_from_z(model, mu, logvar, n, idx, val):  #

    # Input to tensor and init the Reconstructions list
    mu, logvar = torch.from_numpy(mu), torch.from_numpy(logvar)
    mu[:, idx], logvar[:, idx] = val
    REC = torch.empty((0, mu.shape[0], mu.shape[1]))

    # Generate and cat
    for i in range(n):
        z = model.reparametrization_trick(mu, logvar)
        rec = model.decoder(z).unsqueeze(dim=0)
        REC = torch.cat((REC, rec), dim=0)

    # calculate mean
    REC_mean = torch.mean(REC, dim=0)

    return REC, REC_mean


def experiment(data, model):
    # x, z, rec = [], [], []
    # with torch.no_grad():
    #     for i, data in enumerate(data):
    #         x_rec, mu, logvar = model(data)
    #         Z = model.reparametrization_trick(mu, logvar)
    #
    #         x.extend(data[:, :, 0].detach().numpy())
    #         z.extend(Z[:].detach().numpy())
    #         rec.extend(x_rec[:].detach().numpy())
    #
    # x = np.array(x)
    # z = np.array(z)
    # rec = np.array(rec)

    x, mu, logvar, z, x_rec = sample_from_data(model, data)
    # Create a figure and axis object
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    axs = [ax1, ax2]

    # Plot the initial data

    line1 = ax1.plot(x, "b")
    line2 = ax1.plot(x_rec, "r")
    line3 = ax2.plot(z, "g")

    # ax1 = plt.subplot2grid((2, 2), (0, 0))
    # ax1 = plt.subplot2grid((2, 2), (0, 1))

    sliders = []
    slider_axs = []

    # Add a slider widget for each z
    for i in range(z.shape[1]):
        # If statement to place the sliders on the left and right, for the layout
        if i < z.shape[1] / 2:
            slider_axs.append(plt.axes([0.1, (0.3 - 0.02 * i), 0.35, 0.03]))
        else:
            slider_axs.append(plt.axes([0.5, (0.3 - 0.02 * (i - z.shape[1] / 2)), 0.35, 0.03]))

        # populate the sliders list
        sliders.append(Slider(slider_axs[i], r'$Z_{{}}$'.format(i), -100, 100, valinit=z[-1, i]))

    # define initial values
    init_vals = [slider.valinit for slider in sliders]

    def update(val, idx, mu, logvar, z):
        # for i in range(z.shape[1]):
        temp = np.copy(z)
        temp[:, idx] = sliders[idx].val
        rec = model.decoder(torch.Tensor(temp)).detach().numpy()
        # z[:, idx] = sliders[idx].val
        # rec = model.decoder(torch.Tensor(z)).detach().numpy()
        rec, rec_mean = sample_from_z(model, mu, logvar, 100, idx, val)
        for channel, line in enumerate(line2):
            line.set_ydata(rec_mean[:, channel])

        for channel, line in enumerate(line3):
            line.set_ydata(temp[:, channel])
        # slider.set_val(sliders[.valinit)
        # line3[idx].set_ydata(temp[:, idx])
        for i, slider in enumerate(sliders):
            if slider.val != init_vals[i] and idx != i:
                slider.set_val(slider.valinit)

        fig.canvas.draw_idle()
        plt.grid(True)

    # Connect the sliders to the update function
    for i, slider in enumerate(sliders):
        slider.on_changed(lambda val, idx=i: update(val, idx, z))

    # Show the plot
    plt.show()
    plt.grid(True)
