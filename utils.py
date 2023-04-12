#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
            rec.extend(x_rec[:].detach().numpy())
        
    plt.plot(rec, "r--")
    plt.plot(x[:], "b-")
    plt.ylim(0,100)
    plt.grid(True)
    
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


def experiment(data, model):
    x, z, rec = [], [], []
    with torch.no_grad():
        for i, data in enumerate(data):
            x_rec, mu, logvar = model(data)
            Z = model.reparametrization_trick(mu, logvar)

            x.extend(data[:, :, 0].detach().numpy())
            z.extend(Z[:].detach().numpy())
            rec.extend(x_rec[:].detach().numpy())

    x = np.array(x)
    z = np.array(z)
    rec = np.array(rec)

    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    axs = [ax1, ax2]
    # Create a figure and axis object
    # fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})

    # Plot the initial data

    # ax1 = axs[0]
    # ax2 = axs[1]
    line1 = ax1.plot(x, "b")
    line2 = ax1.plot(rec, "r")
    line3 = ax2.plot(z, "g")

    # ax1 = plt.subplot2grid((2, 2), (0, 0))
    # ax1 = plt.subplot2grid((2, 2), (0, 1))

    sliders = []
    slider_axs = []

    # Add a slider widget for variable 1
    for i in range(z.shape[1]):
        if i < z.shape[1]/2:    slider_axs.append(plt.axes([0.1, (0.3 - 0.02 * i), 0.35, 0.03]))
        else: slider_axs.append(plt.axes([0.5, (0.3 - 0.02 * (i- z.shape[1]/2)), 0.35, 0.03]))
        sliders.append(Slider(slider_axs[i], r'$Z_{}$'.format(i), -100, 100, valinit=z[-1, i]))

        # Define a function to update the plot
    # define initial values

    init_vals = [slider.valinit for slider in sliders]

    def update(val, idx, z):



        # for i in range(z.shape[1]):
        temp = np.copy(z)
        temp[:, idx] = sliders[idx].val
        rec = model.decoder(torch.Tensor(temp)).detach().numpy()
        # z[:, idx] = sliders[idx].val
        # rec = model.decoder(torch.Tensor(z)).detach().numpy()

        for channel, line in enumerate(line2):
            line.set_ydata(rec[:, channel])

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

    # Adjust the layout
    # fig.tight_layout()
    # plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, top=0.95, hspace=0.4)
    # Show the plot
    plt.show()
    plt.grid(True)

