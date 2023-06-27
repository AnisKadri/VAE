#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
# import ipywidgets as widgets
from matplotlib.gridspec import GridSpec
# from ipywidgets import interact
# from IPython.display import display, clear_output
import matplotlib.gridspec as gridspec
from dataGen import Gen
from train import stridedWindow, slidingWindow, train, criterion
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import sys
from functools import wraps



def suppress_prints(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Store the original value of sys.stdout
        original_stdout = sys.stdout

        try:
            # Replace sys.stdout with a dummy stream that discards output
            dummy_stream = open("/dev/null", "w")  # Use "nul" on Windows
            sys.stdout = dummy_stream
            return func(*args, **kwargs)
        finally:
            # Restore the original sys.stdout
            sys.stdout = original_stdout

    return wrapper

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
    latent_dims = latent_dims

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

    # Get the model parameters
    n_channels = model._n_channels
    latent_dims = model._latent_dims
    num_embed = model.quantizer._num_embed
    dim_embed = model.quantizer._dim_embed
    code_book = model.quantizer._embedding
    L = model._L

    T = data.dataset.data.shape[1] - L
    batch_size = data.batch_size

    # Init tensors to store results
    x = torch.empty((T, n_channels))
    mu, logvar, z, embed = (torch.empty((n, T, n_channels*2, latent_dims)) for _ in range(4))
    x_rec = torch.empty(n, T, n_channels)

    # Loop through data n times
    for i, (data, v) in enumerate(data):
        for j in range(n):
            # generate reconstruction and latent space over the x axis
            rec, loss, _mu, _logvar = model(data)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
                v = v.unsqueeze(-1)

            _z = model.reparametrization_trick(_mu, _logvar)
            _embed, _ = model.quantizer(_z)

            # Fill the Tensors with data Shape (mu, logvar,z): n*T*Latent_dim      x_rec = n*T*C
            mu[j, i * batch_size: (i + 1) * batch_size, :]      = _mu#.view(_mu.shape[0], -1)
            logvar[j, i * batch_size: (i + 1) * batch_size, :]  = _logvar#.view(_logvar.shape[0], -1)
            z[j, i * batch_size: (i + 1) * batch_size, :]       = _z#.view(Z.shape[0], -1)
            embed[j, i * batch_size: (i + 1) * batch_size, :] = _embed
            x_rec[j, i * batch_size: (i + 1) * batch_size, :]   = (rec * v)[:, :, 0]

        x[i * batch_size: (i + 1) * batch_size, :]              = (data*v)[:, :, 0] # Shape T*C

    # Calculate the mean for mu, logvar, z and x_rec
    mu, logvar, z, embed, x_rec_mean = (torch.mean(t, dim=0) for t in [mu, logvar, z, embed, x_rec])

    # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    x_rec = torch.permute(x_rec, (1, 0, 2))
    x_rec = x_rec.reshape(T, -1)

    # convert to numpy, print shapes and output
    x, z, x_rec = (t.detach().numpy() for t in [x, z, x_rec])
    print("Tensors x: {}, mu: {}, logvar: {}, z: {}, x_rec: {}".format(x.shape, mu.shape, logvar.shape, z.shape, x_rec.shape))
    return x, mu, logvar, z, embed, x_rec, x_rec_mean

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
def sample_from_quantizer_VQ(model, data, mu, logvar, n, codebook):
    # Get necessary variables for the init
    # latent_dims = latent_dims

    # Get the model parameters
    n_channels = model._n_channels
    latent_dims = model._latent_dims
    num_embed = model.quantizer._num_embed
    dim_embed = model.quantizer._dim_embed
    code_book = model.quantizer._embedding
    L = model._L
    # mu = torch.FloatTensor(mu)
    # logvar = torch.FloatTensor(logvar)

    T = data.dataset.data.shape[1] - L
    batch_size = data.batch_size

    # Init tensors to store results
    # mu, logvar, z, embed = (torch.empty((n, T, n_channels * 2, latent_dims)) for _ in range(4))
    # x_rec = torch.empty(n, T, n_channels)
    # x = torch.empty((T, n_channels))
    # z = torch.empty((n, T, latent_dims))
    x_rec = torch.empty(n, T, n_channels)

    model.quantizer._embedding = codebook
    # Loop through data n times
    for i,(_mu, _logvar) in enumerate(zip(mu, logvar)):
        for j in range(n):
            _z = model.reparametrization_trick(_mu, _logvar)
            _embed, _ = model.quantizer(_z)
            rec = model.decoder(_embed)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
                v = v.unsqueeze(-1)

            x_rec[j, i * batch_size: (i + 1) * batch_size, :] = (rec * v)[:, :, 0]

    # Calculate the mean for mu, logvar, z and x_rec
    x_rec_mean = torch.mean(x_rec, dim=0)

    return x_rec, x_rec_mean

    # for channel, line in enumerate(rec_lines[0]):
    #     line.set_ydata(x_rec[:, channel])
    # for channel, line in enumerate(rec_lines[1]):
    #     line.set_ydata(x_rec_mean[:, channel])



    # for i, (data, v) in enumerate(data):
    #     for j in range(n):
    #         # generate reconstruction and latent space over the x axis
    #
    #         rec, loss, _mu, _logvar = model(data)
    #
    #         # print(_mu.mean(-1).shape)
    #         Z = model.reparametrization_trick(_mu, _logvar)
    #         Z, _ = model.quantizer(Z)
    #         # Fill the Tensors with data Shape (mu, logvar,z): n*T*Latent_dim      x_rec = n*T*C
    #         # mu[j, i * batch_size: (i + 1) * batch_size, :]      = _mu.mean(-1)#.view(_mu.shape[0], -1)
    #         # logvar[j, i * batch_size: (i + 1) * batch_size, :]  = _logvar.mean(-1)#.view(_logvar.shape[0], -1)
    #         z[j, i * batch_size: (i + 1) * batch_size, :]       = Z.mean(-1)#.view(Z.shape[0], -1)
    #         x_rec[j, i * batch_size: (i + 1) * batch_size, :]   = rec*v
    #
    #     # x[i * batch_size: (i + 1) * batch_size, :]              = data[:, :, 0] # Shape T*C
    #
    # # Calculate the mean for mu, logvar, z and x_rec
    # z, x_rec_mean = (torch.mean(t, dim=0) for t in [z, x_rec])
    #
    # # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    # x_rec = torch.permute(x_rec, (1, 0, 2))
    # x_rec = x_rec.reshape(T, -1)
    #
    # # convert to numpy, print shapes and output
    # z, x_rec = (t.detach().numpy() for t in [z, x_rec])
    # print("Tensors z: {}, x_rec: {}".format(z.shape, x_rec.shape))
    # return z, x_rec, x_rec_mean
@torch.no_grad()
def experiment_VQ(data, model):
    # Init Slider and slider_axs lists
    sliders = []
    slider_axs = []
    text_boxes = []

    # Get the model parameters
    n_channels = model._n_channels
    latent_dims = model._latent_dims
    code_book = model.quantizer._embedding.weight

    # Sample x, x_rec and the latent space
    x, mu, logvar, z, embed, x_rec, x_rec_mean = sample_from_data_VQ(model, data, 3)
    print("x {}, x_rec {} ".format(x.shape, x_rec.shape))

    quantizers = model.quantizer._embedding.weight.detach().numpy()
    print(quantizers.shape)
    # Create a figure and axis object
    gs = GridSpec(2, 2)
    fig = plt.figure()
    # ax1 = plt.subplot2grid((latent_dims + 1, n_channels + 1), (0, 0), colspan=n_channels + 1)
    # # ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    # ax3 = plt.subplot2grid((2, 2), (2, 1), rowspan= latent_dims, colspan=1)
    # ax4 = plt.subplot2grid((2, 2), (2, 0), colspan=1)
    # axs = [ax1, ax3, ax4]

    ax_plot = fig.add_subplot(gs[0, :])
    data_lines = ax_plot.plot(x, "b")
    rec_lines = ax_plot.plot(x_rec, "orange", alpha = 0.2)
    rec_lines_mean = ax_plot.plot(x_rec_mean, "r")

    ax_plot.set_title('Reconstruction')

    # Plot the initial data
    # data_lines = ax1.plot(x, "b")
    # rec_lines = ax1.plot(x_rec, "orange", alpha = 0.2)
    # rec_lines_mean = ax1.plot(x_rec_mean, "r")
    # rec_lines_ = [rec_lines, rec_lines_mean]
    # z_lines = ax2.plot(z, "g")
    # quantizers = ax2.plot()
    # create the heat map on the bottom right
    ax_heatmap = fig.add_subplot(gs[1, 1])
    code_book_widget = ax_heatmap.imshow(code_book)
    ax_heatmap.set_title('Codebook Heatmap')
    # code_book_widget = ax3.imshow(code_book)

    # create the grid of text inputs on the bottom left
    ax_inputs = fig.add_subplot(gs[1, 0])
    gs_inputs = gs[1, 0].subgridspec(2, 2)

    for i in range(2):
        for j in range(2):
            ax_input = fig.add_subplot(gs_inputs[i, j])
            text_boxes.append(TextBox(ax_input, f'', initial=r'{:.3f}'.format(code_book[i, j]), textalignment="center"))
    ax_inputs.set_title('CodeBook Values')

    # # create the grid of text inputs on the bottom left
    # ax_inputs = fig.add_subplot(gs[1, 0])
    # input_grid = np.zeros((latent_dims, n_channels), dtype=object)
    # gs_inputs = gs[1, 0].subgridspec(latent_dims, n_channels)
    # # ax4.
    # # ax4.plot(gs[:,:])
    #
    # for i in range(latent_dims):
    #     for j in range(n_channels):
    #         ax_input = fig.add_subplot(gs_inputs[i, j])
    #
    #         input_grid[i, j] = TextBox(ax_inputs, f'Input ({i + 1},{j + 1})', initial=r'{}'.format(code_book[i, j]))
    #         # input_grid[i, j].on_submit(lambda val: print(f'({i + 1},{j + 1}): {val}'))
    # ax_inputs.set_title('CodeBook Values')

            # text_box = TextBox(gs[i, j], 'df', initial= r'{}'.format(code_book[i, j]))
            # text_boxes.append(text_box)

    # set the layout parameters for the TextBox widgets
    # for text_box in text_boxes:
    #     text_box.label.set_size(5)
    #     text_box.text_disp.set_size(5)

    # # Add a slider widget for each entry in the codebook
    # row_step = 0.4/n_channels + 0.1
    # col_step = 0.4/latent_dims +0.1
    # width = 0.2 /n_channels
    # height = 0.5/latent_dims
    # for i in range(latent_dims):
    #     for j in range(n_channels):
    #         # place the sliders on the left and right, for the layout  [left, bottom, width, height]
    #         slider_axs.append(plt.axes([col_step * j, row_step * i, width, height]))
    #         k = i*n_channels + j
    #
    #         # populate the sliders list
    #         sliders.append(TextBox(slider_axs[k], label=r'', initial= r'{}'.format(code_book[i, j]) ))
    #         # sliders.append(Slider(slider_axs[k], r'$Ch_{{{}}} Z_{{{}}}$'.format(j, i), -100, 100, valinit=code_book[i, j]))

    # define initial values
    # init_vals = [slider.valinit for slider in sliders]

    # def update(val, idx, mu, logvar, z, sliders):
    #     # # save z and sample new rec and rec_mean
    #     # temp = np.copy(z)
    #     # # n_channels = x.shape[1]
    #     # get the indices of the changed value in quantizers
    #     quantizers = model.quantizer._embedding.weight
    #     latent_dims, n_channels = quantizers.shape[0], quantizers.shape[1]
    #     latent = idx % latent_dims
    #     channel = idx // latent_dims
    #
    #     # Apply the change
    #     quantizers[latent, channel] = torch.tensor(float(val), dtype=torch.float)
    #     # rec, rec_mean, Z = sample_from_z_VQ(model, mu, logvar, 100, idx, val)
    #     rec, rec_mean = sample_from_quantizer_VQ(model, data, mu, logvar, 3, quantizers)
    #     # redraw rec, rec_mean and z
    #     for channel, line in enumerate(rec_lines):
    #         line.set_ydata(rec[:, channel])
    #     for channel, line in enumerate(rec_lines_mean):
    #         line.set_ydata(rec_mean[:, channel])
    #
    #     # Reset Sliders to initial values
    #     # for i, slider in enumerate(sliders):
    #     #     if slider.val != init_vals[i] and idx != i:
    #     #         slider.reset()
    #
    #     fig.canvas.draw_idle()
    #     plt.grid(True)


    def update(val, idx):
        # # save z and sample new rec and rec_mean
        # temp = np.copy(z)
        # # n_channels = x.shape[1]
        # get the indices of the changed value in quantizers
        quantizers = model.quantizer._embedding.weight
        latent_dims, n_channels = quantizers.shape[0], quantizers.shape[1]
        latent = idx % latent_dims
        channel = idx // latent_dims

        # Apply the change
        quantizers[latent, channel] = torch.tensor(float(val), dtype=torch.float)
        # rec, rec_mean, Z = sample_from_z_VQ(model, mu, logvar, 100, idx, val)
        rec, rec_mean = sample_from_quantizer_VQ(model, data, mu, logvar, 3, quantizers)
        # redraw rec, rec_mean and z
        for channel, line in enumerate(rec_lines):
            line.set_ydata(rec[:, channel])
        for channel, line in enumerate(rec_lines_mean):
            line.set_ydata(rec_mean[:, channel])

        # fig.canvas.draw_idle()
        # plt.grid(True)

    # create the grid of text inputs on the bottom left
    ax_inputs = fig.add_subplot(gs[1, 0])
    gs_inputs = gs[1, 0].subgridspec(2, 2)


    for i in range(2):
        for j in range(2):
            ax_input = fig.add_subplot(gs_inputs[i, j])

            text_boxes.append(TextBox(ax_input, f'', initial=r'{:.3f}'.format(code_book[i, j]), textalignment="center"))
            # gs_inputs[i, j].on_submit(lambda  val, idx = i : update(val, idx))
    for i, t in enumerate(text_boxes):
        t.on_submit(lambda text, idx=i: update(text, idx))
    # def save(text):
    #     print(text)
    #     torch.save(model, r'modules\{}.pt'.format(text))
    #
    # text_ax = plt.axes([0.5, 0.01, 0.35, 0.05])
    # text_box = TextBox(text_ax,'Save as: ', initial="beta_vae3")
    # Connect the sliders to the update function
    # for i, slider in enumerate(sliders):
    #     slider.on_submit(lambda  text, idx = i : update(text, idx))
        # slider.on_submit(lambda text, idx=i: update(text, idx, mu, logvar, z, sliders))
    # text_box.on_submit(save)
    # for text_box in text_boxes:
    #     text_box.ax = fig.add_subplot(gs[text_box.ax.rowNum, text_box.ax.colNum])
    #     text_box.ax._frameon = False
    #     text_box.draw_idle()
    print('here again')
    # Show the plot
    fig.tight_layout()
    plt.show()
    # plt.grid(True)



def new_experiment_VQ(data, model):

    # Init Slider and slider_axs lists
    sliders = []
    slider_axs = []

    # Get the model parameters
    n_channels = model._n_channels
    latent_dims = model._latent_dims
    num_embed = model.quantizer._num_embed
    dim_embed = model.quantizer._dim_embed
    code_book = model.quantizer._embedding.weight

    # Sample x, x_rec and the latent space
    x, mu, logvar, z, embed, x_rec, x_rec_mean = sample_from_data_VQ(model, data, 10)
    print("x {}, x_rec {} ".format(x.shape, x_rec.shape))

    def sample_from_quantizer_VQ(codebook):
        # Get necessary variables for the init
        # latent_dims = latent_dims
        model.quantizer._embedding = codebook

        # Get the model parameters
        n_channels = model._n_channels
        latent_dims = model._latent_dims
        num_embed = model.quantizer._num_embed
        dim_embed = model.quantizer._dim_embed
        code_book = model.quantizer._embedding.weight
        print(code_book.shape)

        L = model._L

        T = data.dataset.data.shape[1] - L
        batch_size = data.batch_size

        # Init tensors to store results
        # mu, logvar, z, embed = (torch.empty((n, T, n_channels * 2, latent_dims)) for _ in range(4))
        # x_rec = torch.empty(n, T, n_channels)
        # x = torch.empty((T, n_channels))
        # z = torch.empty((n, T, latent_dims))
        x_rec = torch.empty(n, T, n_channels)


        # Loop through data n times
        for i, (_mu, _logvar) in enumerate(zip(mu, logvar)):
            for j in range(n):
                _z = model.reparametrization_trick(_mu, _logvar)
                _embed, _ = model.quantizer(_z)
                rec = model.decoder(_embed)
                if v.dim() == 1:
                    v = v.unsqueeze(-1)
                    v = v.unsqueeze(-1)

                x_rec[j, i * batch_size: (i + 1) * batch_size, :] = (rec * v)[:, :, 0]

        # Calculate the mean for mu, logvar, z and x_rec
        x_rec_mean = torch.mean(x_rec, dim=0)

        for channel, line in enumerate(rec_lines[0]):
            line.set_ydata(x_rec[:, channel])
        for channel, line in enumerate(rec_lines[1]):
            line.set_ydata(x_rec_mean[:, channel])

        # for i, (data, v) in enumerate(data):
        #     for j in range(n):
        #         # generate reconstruction and latent space over the x axis
        #
        #         rec, loss, _mu, _logvar = model(data)
        #
        #         # print(_mu.mean(-1).shape)
        #         Z = model.reparametrization_trick(_mu, _logvar)
        #         Z, _ = model.quantizer(Z)
        #         # Fill the Tensors with data Shape (mu, logvar,z): n*T*Latent_dim      x_rec = n*T*C
        #         # mu[j, i * batch_size: (i + 1) * batch_size, :]      = _mu.mean(-1)#.view(_mu.shape[0], -1)
        #         # logvar[j, i * batch_size: (i + 1) * batch_size, :]  = _logvar.mean(-1)#.view(_logvar.shape[0], -1)
        #         z[j, i * batch_size: (i + 1) * batch_size, :]       = Z.mean(-1)#.view(Z.shape[0], -1)
        #         x_rec[j, i * batch_size: (i + 1) * batch_size, :]   = rec*v
        #
        #     # x[i * batch_size: (i + 1) * batch_size, :]              = data[:, :, 0] # Shape T*C
        #
        # # Calculate the mean for mu, logvar, z and x_rec
        # z, x_rec_mean = (torch.mean(t, dim=0) for t in [z, x_rec])
        #
        # # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
        # x_rec = torch.permute(x_rec, (1, 0, 2))
        # x_rec = x_rec.reshape(T, -1)
        #
        # # convert to numpy, print shapes and output
        # z, x_rec = (t.detach().numpy() for t in [z, x_rec])
        # print("Tensors z: {}, x_rec: {}".format(z.shape, x_rec.shape))
        # return z, x_rec, x_rec_mean

    quantizers = model.quantizer._embedding.weight.detach().numpy()
    print(quantizers.shape)
    # Create a figure and axis object
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 2), (2, 1), colspan=1)
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
    axs = [ax1, ax2, ax3, ax4]

    # Plot the initial data
    data_lines = ax1.plot(x, "b")
    rec_lines = ax1.plot(x_rec, "orange", alpha=0.2)
    rec_lines_mean = ax1.plot(x_rec_mean, "r")
    rec_lines_ = [rec_lines, rec_lines_mean]


    # Create a list of TextInput widgets
    text_inputs = [widgets.FloatText(value=code_book[j, i]) for i in range(n_channels) for j in range(latent_dims)]

    for text_input in text_inputs:
        display(text_input)
    # Create a GridBox widget to arrange the text inputs in a grid
    grid_box = widgets.GridBox(text_inputs,
                               layout=widgets.Layout(grid_template_columns="repeat({}, 100px)".format(latent_dims),
                                                     grid_template_rows="repeat({}, 30px)".format(n_channels)))
    n = 10
    out = widgets.interactive_output(sample_from_quantizer_VQ,
                                     {'codebook': grid_box.children})

    widgets.VBox([widgets.HBox([grid_box]), out])

def set_effect(effect):
    if effect == "no_effect":
        effects = {
            "Pulse": {
                "occurances": 0,
                "max_amplitude": 1.5,
                "interval": 40
            },
            "Trend": {
                "occurances": 0,
                "max_slope": 0.005,
                "type": "linear"
            },
            "Seasonality": {
                "occurances": 0,
                "frequency_per_week": (7, 14),  # min and max occurances per week
                "amplitude_range": (5, 20),
            },
            "std_variation": {
                "occurances": 0,
                "max_value": 10,
                "interval": 1000,
            },
            "channels_coupling": {
                "occurances": 0,
                "coupling_strengh": 20
            },
            "Noise": {
                "occurances": 0,
                "max_slope": 0.005,
                "type": "linear"
            }
        }
    elif effect == "trend":
        effects = {
            "Pulse": {
                "occurances": 0,
                "max_amplitude": 1.5,
                "interval": 40
            },
            "Trend": {
                "occurances": 1,
                "max_slope": 0.005,
                "type": "linear"
            },
            "Seasonality": {
                "occurances": 0,
                "frequency_per_week": (7, 14),  # min and max occurances per week
                "amplitude_range": (5, 20),
            },
            "std_variation": {
                "occurances": 0,
                "max_value": 10,
                "interval": 1000,
            },
            "channels_coupling": {
                "occurances": 0,
                "coupling_strengh": 20
            },
            "Noise": {
                "occurances": 0,
                "max_slope": 0.005,
                "type": "linear"
            }
        }
    elif effect == "seasonality":
        effects = {
            "Pulse": {
                "occurances": 0,
                "max_amplitude": 1.5,
                "interval": 40
            },
            "Trend": {
                "occurances": 0,
                "max_slope": 0.005,
                "type": "linear"
            },
            "Seasonality": {
                "occurances": 1,
                "frequency_per_week": (7, 14),  # min and max occurances per week
                "amplitude_range": (5, 20),
            },
            "std_variation": {
                "occurances": 0,
                "max_value": 10,
                "interval": 1000,
            },
            "channels_coupling": {
                "occurances": 0,
                "coupling_strengh": 20
            },
            "Noise": {
                "occurances": 0,
                "max_slope": 0.005,
                "type": "linear"
            }
        }

    return effects
def generate_data(n_channels, effect, L, periode=365, step=5, val=500 ):
    effects = set_effect(effect)
    X = Gen(periode, step, val, n_channels, effects)
    x, params, e_params = X.parameters()
    # pprint.pprint(params)
    # pprint.pprint(e_params)
    # X.show()
    x = torch.FloatTensor(x)

    # x = F.normalize(x, p=2, dim=1)
    n = x.shape[1]
    # L = 30

    train_ = x[:, :int(0.8*n)]
    val_   = x[:, int(0.8*n):int(0.9*n)]
    test_  = x[:, int(0.9*n):]

    train_data = DataLoader(stridedWindow(train_, L),
                            batch_size= 22,# 59, # 22
                            shuffle = False
                            )
    val_data = DataLoader(stridedWindow(val_, L),
                            batch_size=22,
                            shuffle = False
                            )
    test_data = DataLoader(stridedWindow(test_, L),
                            batch_size=22,
                            shuffle = False
                            )
    return X, train_data, test_data, effects


def train_on_effect(model, opt, device, n_channels=1, effect='no_effect', n_samples=10, epochs_per_sample=50):
    L = model._L
    latent_dims = model._latent_dims
    for i in range(n_samples):
        X, train_data, test_data = generate_data(n_channels, effect, L)
        x, params, e_params = X.parameters()

        for epoch in range(1, epochs_per_sample):
            train(model, train_data, criterion, opt, device, epoch, VQ=True)
        save(x, "data", effect, n_channels, latent_dims, L, i)
        save(params, "params", effect, n_channels, latent_dims, L, i)
        save(e_params, "e_params", effect, n_channels, latent_dims, L, i)
        save(model, "model", effect, n_channels, latent_dims, L, i)
    return model, X, train_data

def train_on_effect_and_parameters(model, id_model, opt, id_opt, device, n_channels=1, effect='no_effect', n_samples=10, epochs_per_sample=50):
    L = model._L
    latent_dims = model._latent_dims
    for i in range(n_samples):
        for p in model.parameters():
            p.requires_grad = True
        X, train_data, test_data = generate_data(n_channels, effect, L)
        x, params, e_params = X.parameters()
        labels = extract_labels(params, e_params)

        for epoch in range(1, epochs_per_sample):
            train(model, train_data, criterion, opt, device, epoch, VQ=True)            
            
        for epoch in range(1, 500):
            train_identifier(model, id_model, labels, train_data, criterion, id_opt, device, epoch, VQ=True)
            
        test_identifier(model, id_model, labels, train_data, criterion, id_opt, device, epoch, VQ=True)
        print(labels)
#         save(x, "data", effect, n_channels, latent_dims, L, i)
#         save(params, "params", effect, n_channels, latent_dims, L, i)
#         save(e_params, "e_params", effect, n_channels, latent_dims, L, i)
#         save(model, "model", effect, n_channels, latent_dims, L, i)
    return model, X, train_data

def save(obj, name, effect, n_channels, latent_dims, L, i):
    torch.save(obj, r'modules\vq_vae_{}_{}_{}channels_{}latent_{}window_{}.pt'.format(name, effect, n_channels, latent_dims, L, i))


def get_average_norm_scale(train_data, model):
    n_channels = model._n_channels
    norm = torch.empty(n_channels, 0)

    for i, (data, norm_scale) in enumerate(train_data):
        reshaped_norm = norm_scale.permute(1, 0, 2).flatten(1)
        norm = torch.cat((norm, reshaped_norm), 1)

    avg_norm = torch.mean(norm, dim=1)
    return avg_norm


def get_latent_variables(train_data, model):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_channels = model._n_channels
    latent_dims = model._latent_dims

    latents = torch.empty(n_channels, latent_dims, 0, device=device)

    for i, (data, norm_scale) in enumerate(train_data):
        data = data.to(device)

        mu, logvar = model.encoder(data)
        z = model.reparametrization_trick(mu, logvar)

        reshaped_mu, reshaped_logvar, reshaped_z = (t.permute(1, 2, 0) for t in [mu, logvar, z])

        latents = torch.cat((latents, reshaped_z), 2)

    avg_latents = torch.mean(latents, dim=2)
    return latents, avg_latents


def get_index_from_date(date, ref_time = '2023-03-01T00:00:00', step = 5):  
    
    # Convert the Ref time and the given date to a datetime variable
    reference_time = np.datetime64(ref_time)
    reference_time = datetime.strptime(reference_time.astype(str), "%Y-%m-%dT%H:%M:%S")
    date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
    
    # Calculate the diff between the input date and the Ref Time, convert to minutes and get the index using the step
    # step is the number of minutes between each sample
    index = int((date - reference_time).total_seconds() // (60 * step)) 
    
    return index

# look for the index of the first zero element in the given tensor
def get_next_empty_index(param_tensor):
    return (param_tensor == 0).nonzero()[0].item()

@suppress_prints
def extract_param_per_effect(labels, e_params, effect_n, effect):
    print("###################")
    print("Effect: ", effect)
    
    # for the givven effect, first loop through the channel numbers where this effect occures
    for (i, channel) in enumerate(e_params[effect]["channel"]):
        print("//////////////////")
        print("Channel: ", channel)
        # for each occurance of this effect loop through all the parameters of this effect 
        for param_type in e_params[effect]:
            # skip the channel list
            if param_type != "channel":
                print("----------------------")
                
                print("Parameter: ", param_type)
                
                # look for the next empty slot in the labels tensor
                next_idx = get_next_empty_index(labels[channel][effect_n])
                print("Correspending tensor: ", labels[channel][effect_n])
                print("Next Index: ", next_idx)
                
                #get the value of the parameter
                val = e_params[effect][param_type][i]
                
                # if the parameter is the index where it happens (date string) transform it to a an int
                if param_type == "index": val = get_index_from_date(val)
                print("Value:", val)
                
                #fill the labels tensor
                labels[channel][effect_n][next_idx] = val
                
def get_max_occ(effects):
    max_occ = 0
    for effect in effects:
        if effects[effect]["occurances"] > max_occ:
            max_occ = effects[effect]["occurances"]

    return max_occ

def squeeze_labels(labels):
    new_labels = []
    idxs = labels.nonzero()
    
    for index in idxs:
        ch = index[0]    
        new_val = [ch, labels[index[0], index[1], index[2]]]
        new_labels.append(new_val)
        
    new_labels = torch.tensor(new_labels)
    return new_labels

def extract_parameters(n_channels, e_params, effects):
    # create the labels tensor
    n_effects = len(effects)
    max_occ = get_max_occ(effects)
    labels = torch.zeros((n_channels, n_effects, 4 * max_occ))
    
    # loop through the effects and extract their paramaters
    for (effect_n, effect) in enumerate(e_params):
        if effect != "Channels_Coupling":
            extract_param_per_effect(labels, e_params, effect_n, effect)
    labels = squeeze_labels(labels)    
        
    return labels

def add_mu_std(labels, params):
    mu = torch.FloatTensor(params["mu"]).mean(dim=1)
    std = torch.diagonal(torch.FloatTensor(params["cov"])).mean(dim=0)
    mu = torch.stack([torch.arange(len(mu)), mu], dim=1)
    std = torch.stack([torch.arange(len(std)), std], dim=1)
    mu_std = torch.cat((mu, std), dim=0).view(-1,2)
    labels = torch.cat((mu_std, labels), dim=0)
    labels = labels[labels[:, 0].sort(stable=True)[1]]
    
    return labels