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
from dataGen import Gen, FastGen, Gen2
from train import *
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import sys
from functools import wraps
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
# from torchaudio.functional import filtfilt
import random
import pprint
from IPython import display
import scipy.fft as sf
from scipy.signal import find_peaks
# import umap
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
import dill




class GENV:
    def __init__(self, periode=2,
                 step=5,
                 val=500,
                 n_channels=5,
                 n_samples=2,
                 samples_factor=10,
                 bs=256,
                 shuffle=True,
                 L=288*2,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 latent_dims=20,
                 num_layers=3,
                 num_embed=100,
                 slope=0.1,
                 first_kernel=288,
                 ß=0.25,
                 commit_loss=1.5,
                 epochs=100,
                 learning_rate=1e4,
                 beta1=0.5,
                 beta2=0.9,
                 split=(0.8, 0.9), 
                 window=StridedWindow,
                 no_window=NoWindow,
                 modified=True,
                 reduction=True,
                 robust=False,
                 an_percentage=0.15,
                 an_max_amp=10,
                 exp_factor=4,
                 lin_layers=4,
                 n_heads=8,
                 trs_layers=2,
                 min_max=True
                ):
        
        self.periode = periode                     # Length of time series in days.
        self.step = step                           # Messing tackt in mins.
        self.val = val                             # Max value when generating the time series.
        self.n_channels = n_channels               # Number of channels in each time series.
        self.n_samples = n_samples                 # Number of samples per generated time series.
        self.samples_factor = samples_factor       # How many time series to generate.
        self.bs = bs                               # Batch size.
        self.shuffle = shuffle                     # Shuffle option in DataLoader.
        self.L = L                                 # Window length (288 is one day of 5 mins tackt).
        self.device = device                       # Which device the training is on.
        self.latent_dims = latent_dims             # Size of Latent space.
        self.num_layers = num_layers               # Num of hidden Layers in the encoder/decoder.
        self.num_embed = num_embed                 # Num of codebook vectors in VQ.
        self.slope = slope                         # Slope to use in LeakyRELU.
        self.first_kernel = first_kernel           # Size of the first Kernel in the Long TC.
        self.ß=ß                                   # Disentangelement factor in vae.
        self.commit_loss = commit_loss             # Commitement Loss factor of VQ.
        self.epochs = epochs                       # Number of epochs to train.
        self.learning_rate = learning_rate         # Learning rate.
        self.beta1 = beta1                         # Adam beta1 param.
        self.beta2 = beta2                         # Adam beta2 param. 
        self.split = split                         # How the data should be split in train/val/test.
        self.window = window                       # Which Window to use.
        self.no_window = no_window                 # To use for no window.
        self.modified = modified                   # double the number of channels in each TC layer if True
        self.reduction = reduction                 # merge the long and short enc outputs if True   
        self.min_max = min_max                     # choose which normalization to use
        self.robust = robust                       # If True Geman-McClure loss with lambda = 0.1 will be used (from RESIST paper)
        self.an_percentage = an_percentage         # Percentage of random Pulse anomalies, between 0 and 1
        self.an_max_amp = an_max_amp               # Max amplitude Pulse anomalies can take
        self.exp_factor = exp_factor               # Expansio actor in the Feed Forward block of transformer
        self.lin_layers = lin_layers               # Number of lin layer in MLP encoder/decoder
        self.n_heads = n_heads                     # Number of heads in attention layer
        self.trs_layers = trs_layers               # Number of transformer layers
        self.enc_out = self.enc_output(            # number of channels at enc output
            self.modified,   
            self.reduction)
        
    def enc_output(self, mod, red):
        if mod:
            out = self.n_channels * 4 *self.num_layers
        else:
            out = self.n_channels * 2
        if red:
            out = out // 2
        return out
    
def count_files_in_directory(directory_path, target_character):
    try:
        # Filter files based on the target character
        matching_files = [file for file in os.listdir(directory_path) if target_character in file]
        print(len(matching_files))

        return len(matching_files)

    except FileNotFoundError:
        print(f'The specified directory "{directory_path}" does not exist.')
        
def save(file_name, obj, obj_type):
    path = f'modules/{file_name}_{obj_type}.pth'
    torch.save(obj, path, pickle_module=dill)

def save_all(vae, vq, X, args, train_loader, val_loader, test_loader, labels=None, train_labels=None, num= None, name=None):
    if num==None:
        num = count_files_in_directory("modules", "vae")
    if name==None:    
        file_name= f"cp_{num}"
    else: file_name = name
            
    save(file_name, vae, "vae")
    save(file_name, vq, "vq")
    save(file_name, X, "X")
    save(file_name, args, "config")
    save(file_name, train_loader, "train_loader")
    save(file_name, val_loader, "val_loader")
    save(file_name, test_loader, "test_loader")
    
    save(file_name, labels, "labels")
    save(file_name, train_labels, "train_labels")
    
def load(obj_type, num=0):
    dir_path = "modules"
    path = f'modules/cp_{num}_{obj_type}.pth'
    try:
        cp = torch.load(path, pickle_module=dill)            
        return cp

    except FileNotFoundError:
        print(f'The specified file "{path}" does not exist.')
        
def load_all(num=0):
    vae  = load("vae", num)
    vq   = load("vq", num)
    X = load("X", num)
    args = load("config", num)
    train_loader = load("train_loader", num)
    val_loader = load("val_loader", num)
    test_loader = load("test_loader", num)
    labels = load("labels", num)
    train_labels = load("train_labels", num)
    
    if labels == None:   
        print("long")
    else:
        print("label")
    
    return vae, vq, X, args, labels, train_loader, val_loader, test_loader, train_labels


def get_statistics_global(data):
    stats_names_global = ["Mean global: ", "Std global: ", "Max global: ", "Min global: ", "Skew global: ", "Kurt global: "]

    mean_global = data.mean(dim=-1, keepdim=True).squeeze()
    std_global = data.std(dim=-1, keepdim=True).squeeze()
    max_global, _ = data.max(dim=-1, keepdim=True)
    min_global, _ = data.min(dim=-1, keepdim=True)
    skew_global =  torch.tensor(st.skew(data, axis=-1))
    kurt_global =  torch.tensor(st.kurtosis(data, axis=-1))
    
    stats_global = torch.stack([mean_global, std_global, max_global.squeeze(), min_global.squeeze(), skew_global, kurt_global], axis=0)
    
    return stats_names_global, stats_global

def get_statistics_window(model, train_data, args, Origin=True):
    
    stats_names_window = ["Mean window: ", "Std window: ", "Max Window: ", "Min window:", "Skew window: ", "Kurt window: "]
    
    for i, (data_tup, label, norm) in enumerate(train_data):
        data = pick_data(data_tup, args)
        norm = [n.to(args.device) for n in norm]
        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e, indices = model(data, ouput_indices=True)
        
        if Origin:
            denorm_data = revert_standarization(data, norm).cpu()
        else: 
            denorm_data = revert_standarization(x_rec, norm).cpu()
            
        mean_window = denorm_data.mean(dim=-1)
        std_window = denorm_data.std(dim=-1)
        max_window, _ = denorm_data.max(dim=-1, keepdim=True)
        min_window, _ = denorm_data.min(dim=-1, keepdim=True)
        skew_window =  torch.tensor(st.skew(denorm_data, axis=-1))
        kurt_window =  torch.tensor(st.kurtosis(denorm_data, axis=-1))
        
    stats_window = torch.stack([mean_window, std_window, max_window.squeeze(), min_window.squeeze(), skew_window, kurt_window], axis=0)
    
    return stats_names_window, stats_window

def get_stats_loss(model, train_data, args, long=True):
    if long:
        Origin, REC, _= rebuild_TS_non_overlapping(model, train_data, args, keep_norm=False)
        Origin, REC = Origin.T, REC.T
    else:
        Origin, REC, _, _ = rebuild_TS(model, train_data, args, keep_norm=False)

    names_global, stats_global = get_statistics_global(Origin.cpu())
    names_window, stats_window = get_statistics_window(model, train_data, args)
    
    _, stats_global_rec = get_statistics_global(REC.cpu())
    _, stats_window_rec = get_statistics_window(model, train_data, args, Origin=False)

    
    global_loss = F.mse_loss(stats_global_rec, stats_global, reduction='none')
    global_loss = global_loss.mean(dim=(-1)) if long else global_loss.mean(dim=(-1,-2))
    window_loss = F.mse_loss(stats_window_rec, stats_window, reduction='none').mean(dim=(-1,-2))
    

    
    return names_global, global_loss, names_window, window_loss

def plot_stats_results(models, train_data, args, long=True, n_frequencies=0):
    colors = ['blue', 'red', 'purple', 'orange']
    fig, ax = plt.subplots(figsize=(16,8))
    for i, model in enumerate(models):
        names_global, global_loss, names_window, window_loss = get_stats_loss(model, train_data, args, long=long)
        combined_tensors = torch.cat((global_loss, window_loss), dim=0)
        combined_names = names_global + names_window
        
        # Set the width of the bars
        bar_width = 0.8

        # Set the position of each label on the x-axis
        x = np.arange(len(combined_names))

        # Create the grouped bar chart
        bar = ax.bar(x - bar_width/2, combined_tensors, bar_width, label=model.model_type, color=colors[i], alpha=1-0.4*i)

        # Add labels, title, and legend
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Comparison of Metrics between VAE and VQ')
        ax.set_xticks(x)
        ax.set_xticklabels(combined_names)
        ax.legend()
        # Rotate and resize x labels
        plt.xticks(rotation=45, fontsize=18)
        plt.yticks(fontsize=18)


    # Display the plot
    plt.tight_layout()
    plt.show()
    
def print_stats_results(model, train_data, args, long=True, n_frequencies=0):
    names_global, global_loss, names_window, window_loss = get_stats_loss(model, train_data, args, long=long)
    
    for i, el in enumerate(names_global): 
        print(el, global_loss[i].item())
#     if long:
#         freqs_true = get_frequencies_per_week_long(model, train_data, args, n=n_frequencies, Origin=True)
#         freqs_rec = get_frequencies_per_week_long(model, train_data, args, n=n_frequencies, Origin=False)
#         print("Frequencies per Week (Original): ", freqs_true) 
#         print("Frequencies per Week (Original): ", freqs_true)
        
    print("\n")
    for i, el in enumerate(names_window): 
        print(el, window_loss[i].item())
    print("\n")
        
def autocov(x, k):
    mean = np.mean(x, axis=0)
    n = x.shape[-1]

    autocovariance = sum((x[...,i] - mean) * (x[...,i + k] - mean) for i in range(n - k)) / n
    return np.array(autocovariance)

def calc_acf(data):
    autocorr = np.empty_like(data)
    autocv_0 = autocov(data, 0)
    for i in range(data.shape[-1]):
        autocv = autocov(data, i)
        autocorr[i] = autocv/autocv_0
    return autocorr

def generate_acf(data):
    acf = np.empty_like(data)
    for i, sample in enumerate(data):
        for j, channel in enumerate(sample):
            acf[i,j] = calc_acf(channel.numpy())
    return acf

def identify_frequencies(autocorr, args, n_frequencies=1, plot=True):
    # Find significant peaks in the ACF
    min_to_week = 7*24*60
    max_lag = len(autocorr) // 2
    peaks, _ = find_peaks(autocorr[10:max_lag], distance=1, height=[0,None], plateau_size=1)

    periodicities = peaks[:n_frequencies]+10
    freqs = [(( 1/(periodicity*args.step) ) * min_to_week) for periodicity in periodicities]
    if plot:
        # Plot the autocorrelation function (ACF)
        plt.figure(figsize=(12, 6))
        plt.plot(autocorr, marker='o', linestyle='-')
        plt.title('Autocorrelation Function (ACF)')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)        

        # Plot the ACF with identified periodicities
        plt.scatter(periodicities, [autocorr[lag] for lag in periodicities], color='red', marker='x', s=100, label='Periodicity Peaks')
        plt.legend()
        plt.show()
        
        # Output the identified periodicities
        print("Identified periodicities:",freqs)
    return np.array(freqs)
    
def get_means_indices(label):
    unique, idx, counts= torch.unique(label[:,0], return_inverse=True, return_counts=True)    
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

def filter_close_values(input_list, threshold):
    if len(input_list) <= 1:
        return input_list  # Nothing to filter if the list has 0 or 1 element

    input_list.sort()  # Sort the list in ascending order
    filtered_list = [input_list[0]]

    for value in input_list[1:]:
        if abs(value - filtered_list[-1]) > threshold and abs(value % filtered_list[-1]) > threshold:
            filtered_list.append(value)
    filtered_list = np.array(filtered_list)
    filtered_array = filtered_list[filtered_list != 0]

    return filtered_array

def suppress_prints(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        # Store the original value of sys.stdout
        original_stdout = sys.stdout

        try:
            # Replace sys.stdout with a dummy stream that discards output
            dummy_stream = open("NUL", "w")  # Use "nul" on Windows
            sys.stdout = dummy_stream
            return func(*arg, **kwargs)
        finally:
            # Restore the original sys.stdout
            sys.stdout = original_stdout

    return wrapper

def revert_min_max(data, norm):
    dist, v_min = norm[0], norm[1]
    x = data.permute(*torch.arange(data.ndim - 1, -1, -1))
    original_val = (dist * x ) + v_min
    original_val = original_val.permute(*torch.arange(original_val.ndim - 1, -1, -1))
    return original_val

def revert_min_max_s(data, norm):
    dist, v_min = norm[0], norm[1]
    dist = dist[:, np.newaxis, np.newaxis]
    v_min = v_min[:, np.newaxis, np.newaxis]
    
    original_val = (dist * data ) + v_min
    return original_val

def revert_standarization(data, norm):    
    std, mean = norm[2], norm[3]    
    original_val = (std * data) + mean
    return original_val

def mute_norm(norm):
    dist = torch.ones_like(norm[0])
    v_min = torch.zeros_like(norm[1])
    
    std = torch.ones_like(norm[2])
    mean = torch.zeros_like(norm[3])
    return (dist, v_min, std, mean)



def rebuild_TS(model, train_loader, args, keep_norm= False, attention=True):    
    device = args.device    
    min_max = args.min_max
    model.to(device)
    model.eval()
    single_head = int(args.latent_dims/args.n_heads)
    
    for p in model.parameters():
        p.requires_grad = False

    data_shape = train_loader.dataset.data.shape 
    e_indices = torch.empty(data_shape[0], args.enc_out, args.latent_dims) #torch.empty(data_shape[0], args.n_channels, single_head)  # torch.empty(data_shape[0], args.enc_out, args.latent_dims) 
    Origin = torch.empty(data_shape)
    REC = torch.empty(data_shape)
    Att = torch.empty(data_shape[0], args.n_heads, args.n_channels, args.n_channels)

    idx = 0
    for sample_idx, (data_tup, label, norm) in enumerate(train_loader):
        if keep_norm:
            norm = mute_norm(norm)
        
        data = pick_data(data_tup, args)
        norm = [n.to(device) for n in norm]
        bs   = data.shape[0]  
        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e, att = model(data, ouput_indices=True)
        
        denorm_data = revert_min_max_s(data, norm) if min_max else revert_standarization(data, norm)
        denorm_rec =  revert_min_max_s(x_rec, norm) if min_max else revert_standarization(x_rec, norm)
#         print(att.shape)
#         print(e.view(bs, args.enc_out, -1).shape)
#         print(e_indices[idx: (idx+bs)].shape)
        e_indices[idx: (idx+bs)] = e.view(bs, args.enc_out, -1)# e.view(bs, args.n_heads, -1) # e.view(bs, args.enc_out, -1)
        Origin[idx: idx+bs] = denorm_data
        REC[idx: idx+bs] = denorm_rec
        if attention:
            Att[idx: idx+bs] = att
        idx += bs
        
    return Origin, REC, e_indices, Att

# @suppress_prints
def rebuild_TS_non_overlapping(model, train_loader, args, keep_norm= False):
    device = args.device
    L = args.L
    n_channels = args.n_channels
    data_shape = train_loader.dataset.data.shape 
    min_max = args.min_max
    
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    e_indices = torch.empty(args.enc_out, 0).to(device)
#     att_scores = torch.empty(args.enc_out, args.latent_dims).to(device)

    Origin = torch.empty(n_channels, 0).to(device)
    REC = torch.empty(n_channels, 0).to(device)

    idx = 0
    for sample_idx, (data_tup, label, norm) in enumerate(train_loader):       
        if keep_norm:
            norm = mute_norm(norm)

        data = pick_data(data_tup, args)        
        norm = [n.to(device) for n in norm]
        bs   = data.shape[0]
        
        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e, att = model(data, ouput_indices=True)
        print("att", att[-1].shape)
        
        denorm_data = revert_min_max(data, norm) if min_max else revert_standarization(data, norm)
        denorm_rec =  revert_min_max(x_rec, norm) if min_max else revert_standarization(x_rec, norm)
        
        Origin    = torch.cat(( Origin, denorm_data.permute(1,0,2).reshape(args.n_channels, -1) ), axis=1)
        REC       = torch.cat(( REC, denorm_rec.permute(1,0,2).reshape(args.n_channels, -1)   ), axis=1)
#         e_indices = torch.cat(( e_indices, indices.view(args.enc_out, -1)     ), axis=1)
#         att_scores = att
#         print(e_indices.shape)
        idx += bs

    return Origin.T, REC.T, att[-1].view(args.n_channels, -1) #e_indices

def step_by_step_plot(model, data_loader, args, x_lim=None):
     
    dataset = data_loader.dataset.data
    dataset.shape
    if x_lim == None:
        x_lim = dataset.shape[-1]
    
    Origin_no_norm, REC_no_norm, indices = rebuild_TS_non_overlapping(model, data_loader, args, keep_norm=True)
    Origin, REC, indices = rebuild_TS_non_overlapping(model, data_loader, args, keep_norm=False)
    
    plot_rec(dataset.T[:x_lim], dataset.T[:x_lim], title = "Generated Time Serie")
    plot_rec(Origin_no_norm[:x_lim].cpu(), Origin_no_norm[:x_lim].cpu(), title="Normalized Time serie")
    plot_rec(Origin[:x_lim].cpu(), dataset.T[:x_lim], title="Reverted Time serie to original stat vs Generated ")
    
def step_by_step_plot_NoWindow(model, data_loader, args, n=0):
     
    dataset = data_loader.dataset.data
    
    Origin_norm, REC_norm, indices = rebuild_TS(model, data_loader, args, keep_norm=True)
    Origin, REC, indices = rebuild_TS(model, data_loader, args, keep_norm=False)
    
    plot_rec(dataset[n].T, dataset[n].T, title = "Generated Time Serie")
    plot_rec(Origin_norm[n].T.cpu(), Origin_norm[n].T.cpu(), title="Normalized Time serie")
    plot_rec(Origin[n].T.cpu(), dataset[n].T, title="Reverted Time serie to original stat vs Generated ")
    
def show_results_long(model, data, args, vq=False, xlim=800):
    
    model_type = "VQ" if vq else "VAE" 
    title = model_type +": Original data vs Reconstruction"
    Origin_norm, REC_norm, _ = rebuild_TS_non_overlapping(model, data, args, keep_norm=True)
    Origin, REC, indices = rebuild_TS_non_overlapping(model, data, args)
    
    plot_rec(Origin_norm[:xlim].cpu(), REC_norm[:xlim].cpu(), title=title+" (normalized)")
    plot_rec(Origin[:xlim].cpu(), REC[:xlim].cpu(), title=title)
    create_heatmap(indices.cpu(), x_label="Z", title="Latent Variables")
    
    if vq:
        codebook = model.quantizer._embedding.weight
        heatmap = create_heatmap(codebook.cpu().detach().numpy() )
        plot_indices(indices.cpu())
        
def show_results(model, data, args, vq=False, sample=1, plot_latent=True):
    
    model_type = "VQ" if vq else "VAE" 
    sample= args.samples_factor * args.n_samples if sample > args.samples_factor * args.n_samples else sample
    title = model_type +": Original data vs Reconstruction"
    Origin_norm, REC_norm, _, _ = rebuild_TS(model, data, args, keep_norm=True)
    Origin, REC, latents, att = rebuild_TS(model, data, args)
    
    plot_rec(Origin_norm[sample].T.cpu(), REC_norm[sample].T.cpu(), title=title+" (normalized)")
    plot_rec(Origin[sample].T.cpu(), REC[sample].T.cpu(), title=title)
    create_heatmap(latents[sample].T.cpu(), x_label="Z", title="Latent Variables")
    if plot_latent:
        plot_latent_per_channel(latents.cpu(), args)
        plot_latent_per_dim(latents.cpu(), args)
        plot_att_per_head(att.cpu(), args)
#     plot_indices(latents[sample].cpu())
    
    if vq:
        codebook = model.quantizer._embedding.weight
        heatmap = create_heatmap(codebook.cpu().detach().numpy() )
        

def plot_heatmap(ax_heatmap, codebook):
    ax_heatmap.clear()
    heatmap = ax_heatmap.imshow(codebook)
    ax_heatmap.set_title('Codebook Heatmap')
    ax_heatmap


    return heatmap
def create_heatmap(codebook, x_label="Num Embeddings in Codebook", title="Codebook"):
    fig, ax_heatmap = plt.subplots(figsize=(12, 4), dpi=100)
    heatmap = plot_heatmap(ax_heatmap, codebook.T)

    cbar = fig.colorbar(heatmap)
    ax_heatmap.set_xlabel(x_label)
    ax_heatmap.set_ylabel('Decoder Input dim (channels)')
    ax_heatmap.set_title(title)
    plt.show()
    
    return ax_heatmap

def plot_rec(origin, rec, lines=None, title="Original vs Reconstruction"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rec, "r--", alpha = 1, label="Reconstructed TS")
    ax.plot(origin, "b", alpha=0.4, label="Original TS")
    ax.set_title(title)
#     ax.legend(loc="upper right")
    ax.grid()
    
    plt.show()
    
def plot_latent_per_channel(latent, args, title="2 dim per each channel"):
    umap_result = []
    pca_results = []
    tsne_results = []
    for i in range(args.enc_out):
        latent_grp = latent[:,i,:]
        pca = PCA(n_components=2, random_state=0)
        reducer = umap.UMAP(n_components=2, random_state=0)
        tsne = TSNE(n_components=2, random_state=0)
        
        pca_embed = pca.fit_transform(latent_grp)
        tsne_embed =  tsne.fit_transform(latent_grp)
        embedding = reducer.fit_transform(latent_grp)
        
        pca_results.append(pca_embed)
        tsne_results.append(tsne_embed)
        umap_result.append(embedding)
     
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(12, 4), dpi=100)
    
    for i, embed in enumerate(pca_results):
        ax[0].scatter(embed[:,0], embed[:,1], label= "Channel {}".format(i))
    for i, embed in enumerate(tsne_results):
        ax[1].scatter(embed[:,0], embed[:,1], label= "Channel {}".format(i))
    for i, embed in enumerate(umap_result):
        ax[2].scatter(embed[:,0], embed[:,1], label= "Channel {}".format(i))
    
      
    ax[0].set_title(title+ " (PCA)")
    ax[1].set_title(title+ " (t-SNE)")
    ax[2].set_title(title+ " (UMAP)")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.show()
    
    
def plot_latent_per_dim(latent, args, title="2 dim per latent dimension"):
    umap_result = []
    pca_results = []
    tsne_results = []
    print(latent.shape)
    for i in range(args.latent_dims):
        latent_grp = latent[...,i]
        
        pca = PCA(n_components=2, random_state=0)
        reducer = umap.UMAP(n_components=2, random_state=0)
        tsne = TSNE(n_components=2, random_state=0)
        
        pca_embed = pca.fit_transform(latent_grp)
        tsne_embed =  tsne.fit_transform(latent_grp)
        embedding = reducer.fit_transform(latent_grp)
        
        pca_results.append(pca_embed)
        tsne_results.append(tsne_embed)
        umap_result.append(embedding)
     
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(12, 4), dpi=100)
    
    for i, embed in enumerate(pca_results):
        ax[0].scatter(embed[:,0], embed[:,1], label= "Z {}".format(i))
    for i, embed in enumerate(tsne_results):
        ax[1].scatter(embed[:,0], embed[:,1], label= "Z {}".format(i))
    
    for i, embed in enumerate(umap_result):
        ax[2].scatter(embed[:,0], embed[:,1], label= "Z {}".format(i))
        
    ax[0].set_title(title+ " (PCA)")
    ax[1].set_title(title+ " (t-SNE)")
    ax[2].set_title(title+ " (UMAP)")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.show()
    
    
    
def plot_att_per_head(att, args, title="2 dim per attention head"):
    umap_result = []
    pca_results = []
    tsne_results = []
    bs = att.shape[0]
    for i in range(args.n_heads):
        head = att.view(bs, args.n_heads, -1).permute(1,0,2)[i]
        
        pca = PCA(n_components=2, random_state=0)
        reducer = umap.UMAP(n_components=2, random_state=0)
        tsne = TSNE(n_components=2, random_state=0)
        
        pca_embed = pca.fit_transform(head)
        tsne_embed =  tsne.fit_transform(head)
        embedding = reducer.fit_transform(head)
        
        pca_results.append(pca_embed)
        tsne_results.append(tsne_embed)
        umap_result.append(embedding)
     
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(12, 4), dpi=100)
    
    for i, embed in enumerate(pca_results):
        ax[0].scatter(embed[:,0], embed[:,1], label= "Z {}".format(i))
    for i, embed in enumerate(tsne_results):
        ax[1].scatter(embed[:,0], embed[:,1], label= "Z {}".format(i))
    
    for i, embed in enumerate(umap_result):
        ax[2].scatter(embed[:,0], embed[:,1], label= "Z {}".format(i))
        
        
    ax[0].set_title(title+ " (PCA)")
    ax[1].set_title(title+ " (t-SNE)")
    ax[2].set_title(title+ " (UMAP)")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.show()
    
    heads = att.mean(axis = (0,1))
    heatmap = create_heatmap(heads.cpu().detach().numpy() )
    
def plot_indices(indices):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(indices.T[:60, :6], alpha = 0.5)    
    ax.grid()
    
    
    plt.show()
    

    
def generate_long_data(args, effects, periode_factor=182, effect="Seasonality", occurance=1, return_gen=False, anomalies=False):
    args.periode *= periode_factor
    n_samples= args.n_samples
    args.n_samples = 1
    effects = set_effect(effect, effects, occurance)
    
    X_long = Gen2(args=args, effects=effects, fast=False)
    if anomalies:
        X_long.add_random_pulse(0.001)
        X_long.sample()
    x_long, params_long, e_params_long = X_long.parameters()
    X_long.show()
    args.periode = args.periode // periode_factor
    args.n_samples = n_samples

    train_data_long, val_data_long, test_data_long = create_loader_Window(x_long.squeeze(0), args=args)
    
    if return_gen:
        return train_data_long, val_data_long, test_data_long, X_long
    else:
        return train_data_long, val_data_long, test_data_long

# def generate_labeled_data(effects, n_samples=500, periode=7, step=5, val=500, n_channels=1, effect="Seasonality", occurance=1, batch_size=10, split=(0.8, 0.9), norm=True):
#     effects = set_effect(effect, effects, occurance)
    
#     X = FastGen(n_samples, periode, step, val, n_channels, effects)
#     for i in range(9):
#         print("generating: ", i)
#         Y = FastGen(n_samples, periode, step, val, n_channels, effects)    
#         X.merge(Y)

#     x, params, e_params = X.parameters()
#     X.show(10)

#     labels = extract_parameters(n_channels, e_params, effects, n_samples*10)
#     labels = add_mu_std(labels, params)

#     train_data, val_data, test_data = create_loader_noWindow(x, labels, batch_size=batch_size, split=split, norm=norm)
#     return train_data, val_data, test_data

def generate_labeled_data(args, effects, effect="Seasonality", occurance=1, norm=True, return_gen=False, anomalies=False, show=5):
    effects = set_effect(effect, effects, occurance)
    
    X = Gen2(args=args, effects=effects)
    for i in range(1, args.samples_factor):
        print("generating: ", i)
        Y = Gen2(args=args, effects=effects)    
        X.merge(Y)
    if anomalies:
        X.add_random_pulse(0.001)

    x, params, e_params = X.parameters()
    X.show(show)

    labels = extract_parameters(args, e_params=e_params, effects=effects)
#     labels = add_mu_std(labels, params)

    train_data, val_data, test_data = create_loader_noWindow(x, args, labels, norm=norm)
    if return_gen:
        return train_data, val_data, test_data, X
    else:
        return train_data, val_data, test_data

def normalize_signal(signal):
    # Find the minimum and maximum values of the signal
    min_value = np.min(signal)
    max_value = np.max(signal)
    
    # Apply min-max normalization to scale the signal between 0 and 1
    normalized_signal = (signal - min_value) / (max_value - min_value)
    
    return normalized_signal

def perform_fft(normalized_signal, sampling_interval):
    time_scaling = 1 / (24 * 60 * 60)

    # Perform the FFT
    fft_signal = fft(normalized_signal)
    freqs = fftfreq(len(normalized_signal), d=sampling_interval * time_scaling) 
    
    return fft_signal, freqs

def get_n_dominant(fft_signal, freqs, n_best):
    # Find the dominant frequency and corresponding amplitude
    idx = np.argsort(np.abs(fft_signal))[::-1][:n_best]
    dominant_frequency = freqs[idx]
    dominant_amplitude = np.abs(fft_signal[idx])
    
    return dominant_frequency, dominant_amplitude

def fft_freq(signal, sampling_interval, n_best):
    normalized_signal = normalize_signal(signal)
#     normalized_signal = signal
    fft_signal, freqs = perform_fft(normalized_signal, sampling_interval)
    return fft_signal, freqs

def plot_fft(normalized_signal, fft_signal, freqs):
    # Plotting the signal and its spectrum
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(normalized_signal)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time-domain Signal')

    ax2.plot(np.abs(fft_signal))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Frequency-domain Spectrum')

    plt.tight_layout()
    plt.show()
    dominant_frequency, dominant_amplitude = get_n_dominant(fft_signal, freqs, 20)
    print("Dominant Frequency:", dominant_frequency, "Hz")
    print("Dominant Amplitude:", dominant_amplitude)

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
#     print(normal_cutoff)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     b = torch.FloatTensor(b)
#     a = torch.FloatTensor(a)
    y = filtfilt(b, a, data)
    return y

def denoise(ts):
#     x_len = len(ts[0])
#     noisy_signal = np.array(ts)
    noisy_signal = ts.cpu()
    
    # Filter requirements.
    T = 365*24*60*60          # Sample Period
    fs = 1/(5)       # sample rate, Hz
    cutoff = 0.003      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples
    
    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(noisy_signal, cutoff, fs, order)

#     plt.figure(figsize=(10, 6))
#     plt.plot(noisy_signal.T, label='Noisy Signal')
#     plt.plot(y.T, label='Filtered Signal')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.xlim(0,1000)
#     plt.grid(True)
#     plt.show()
    
    return y

def denoise_data(data):
    denoised = np.empty_like(data)
    for i, d in enumerate(data):
        denoised[i] = denoise(d.unsqueeze(0).cpu())
    return denoised

def rescale_x_axis(ts, args):
    n = len(ts)
    t = np.arange(n)
    scale_factor = 182 *args.split[0]
    new_n =41761*2# int(scale_factor * n)
    
    # Generate new time vector and zero-padded time series
    new_t = np.linspace(0, n - 1, new_n)
    new_ts = np.interp(new_t, t, ts)
    return new_ts

def find_freqs_p(denoised, n, args):
    new_ts = rescale_x_axis(denoised, args)
    freqs_w = []
    i, sample_rate, N = 0, 12*24*7*2*365, len(denoised)

    X = sf.rfft(new_ts) #/ N
    freqs = sf.rfftfreq(n=N, d=1/sample_rate)
    
    while len(freqs_w) < n:
        i += 1
        max_freq_ind = np.argpartition(np.abs(X), -i)[-i:]
        freqs_w = filter_close_values(freqs[max_freq_ind][:n], 0.005)
        
#     freqs_w = [fs % 63 for fs in freqs_w]    
    return np.array(freqs_w, dtype=np.float64)

def get_frequencies_per_week(v, train_data, args, n):

    Origin_norm, REC_norm, _ = rebuild_TS(v, train_data, args, keep_norm=True)
    denoised = denoise_data(Origin_norm.cpu())

    freqs = np.empty((denoised.shape[0], args.n_channels, n))
    for i, sample in enumerate(freqs):
        for j, fs in enumerate(sample):
            fs = find_freqs_p(denoised[i, j], n, args)
            freqs[i,j] = fs
            
    return freqs

def get_frequencies_per_week_acf(v, data, args, n):
    
    freqs = np.empty((data.shape[0], args.n_channels, n))    
    acf = generate_acf(data)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            autocorr = acf[i, j]
            freqs[i,j] = identify_frequencies(autocorr, args, n, plot=False)
    return freqs


def find_freqs(denoised, n):
    freqs_w = []
    i, sample_rate, N = 0, 24*12*7, len(denoised)

    X = sf.rfft(denoised) #/ N
    freqs = sf.rfftfreq(n=N, d=1/sample_rate)
    
    while len(freqs_w) < n:
        i += 1
        max_freq_ind = np.argpartition(np.abs(X), -i)[-i:]
#         max_freq_ind, _ = find_peaks(np.abs(X), distance=20)
#         freqs_w = freqs[max_freq_ind]
        freqs_w = filter_close_values(freqs[max_freq_ind], 0.2)
        
#     while len(freqs_w) > n:
#         freqs_w = freqs_w[:-1]
        
    freqs_w = [fs % 14 for fs in freqs_w]
    return np.array(freqs_w[:n], dtype=np.float64)

def get_frequencies_per_week_long(v, train_data, args, n, Origin=False):
    
    Origin_norm, REC_norm, _ = rebuild_TS_non_overlapping(v, train_data, args, keep_norm=True)
    denoised = denoise_data(Origin_norm.T.cpu()) if Origin else denoise_data(REC_norm.T.cpu())
    
    freqs = np.empty((args.n_channels, n))
    for i in range(args.n_channels):
        fs = find_freqs(denoised[i], n)
        freqs[i] = fs
        
    return fs



# def compare(dataset, model, VQ=True):
#     model.eval()
#     rec = []
#     x = []
#     with torch.no_grad():
#         for i, data in enumerate(dataset):
#             if VQ:
#                 x_rec, loss, mu, logvar = model(data)
#             else:
#                 x_rec, mu, logvar = model(data)
#             z = model.reparametrization_trick(mu, logvar)
#             x.extend(data[:,:,0].detach().numpy())
#             rec.extend(x_rec.detach().numpy())
#
#     plt.plot(rec, "r--")
#     plt.plot(x[:], "b-")
#     # plt.ylim(0,100)
#     plt.grid(True)
#     plt.show()
    
# def compare_dist(dataset, encoder, decoder):
#     encoder.eval()
#     decoder.eval()
#     rec = []
#     x = []
#     with torch.no_grad():
#         for i, data in enumerate(dataset):
#              # Forward pass through the encoder and compute the latent variables
#             qnet = encoder(data)
#
#             # Sample from the latent variables and forward pass through the decoder
#             z_sample = qnet['z'].rsample()
#             pnet = decoder(z_sample)
#             x_rec = pnet['x'].rsample()
# #             z = model.reparametrization_trick(mu, logvar)
#
#             x.extend(data[:,:,0].detach().numpy())
#             rec.extend(x_rec[:,:,0].detach().numpy())
#
#     plt.plot(rec, "r--")
#     plt.plot(x[:], "b-")
#     plt.ylim(0,100)
#     plt.grid(True)


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


#     def save(text):

#         print(text)
#         torch.save(model, r'modules\{}.pt'.format(text))

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

def no_effects(effects):
    occ = "occurances"
    for effect in effects:
        effects[effect][occ] = 0
        
def set_effect(effect, effects, n):
    
    occ = "occurances"
    considered_effects = ["no_effect", "Std_variation", "Pulse", "Trend", "Seasonality", "both", "all"]
    no_effects(effects)
    
    if effect == "random":
        effect = random.choice(considered_effects[1:])
    
    if effect not in considered_effects:
        print(effect, "is not in the list of effects")
        return effects
    
    if effect == "no_effect":
        return effects
            
    elif effect == "both":
        effects["Trend"][occ] = n
        effects["Seasonality"][occ] = n
    elif effect == "all":
        effects["Pulse"][occ] = n
        effects["Trend"][occ] = n
        effects["Seasonality"][occ] = n
        effects["Std_variation"][occ] = n
        
    else:
        effects[effect][occ] = n
    return effects

def generate_data(args, effects, effect="both", occurance=1):
    effects = set_effect(effect, effects, occurance)
    X = Gen(args, effects)
    x, params, e_params = X.parameters()

    train_data_long, val_data_long, test_data_long = create_loader_Window(x_long.squeeze(0), args=args)

    n = x.shape[1]

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


def train_on_effect(model, opt, args, effects, effect='no_effect', epochs_per_sample=50):
    L = args.L
    latent_dims = args.latent_dims
    n_channels = args.n_channels
    for i in range(args.n_samples):
        X, train_data, val_data, test_data = generate_long_data(args, effects, periode_factor=1, effect=effect)
        x, params, e_params = X.parameters()

        for epoch in range(1, epochs_per_sample):
            train(model, train_data, criterion, opt, args.device, epoch, VQ=True)
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

# def save(obj, name, effect, args, i):
#     torch.save(obj, r'modules\vq_vae_{}_{}_{}channels_{}latent_{}window_{}.pt'.format(name, effect, args.n_channels,
#                                                                                       args.latent_dims, args.L, i))


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


def get_index_from_date(date, ref_time='2023-03-01T00:00:00', step=5):
    
    # Convert the Ref time and the given date to a datetime variable
    reference_time = np.datetime64(ref_time)
    reference_time = datetime.strptime(reference_time.astype(str), "%Y-%m-%dT%H:%M:%S")
    date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
    
    # Calculate the diff between the input date and the Ref Time, convert to minutes and get the index using the step.
    # step is the number of minutes between each sample
    index = int((date - reference_time).total_seconds() // (60 * step)) 
    
    return index

# look for the index of the first zero element in the given tensor
@suppress_prints
def get_next_empty_index(param_tensor):
    print("the tensor in question: ", param_tensor)
    print("step1 :", param_tensor == 0)
    print("step2 :", (param_tensor == 0).nonzero())
    print("step3 :", (param_tensor == 0).nonzero()[0])
    print("step4 :", (param_tensor == 0).nonzero()[0].item())
    value = (param_tensor == 0).nonzero()[0].item()
    return value

@suppress_prints
def extract_param_per_effect(labels, e_params, effect_n, effect):
    print("###################")
    print("Effect: ", effect)
    if labels.dim() == 2: labels.unsqueeze(0)
    n_samples= labels.shape[0]
    
    for sample in range(n_samples):
        print("########Sample########### -> ",sample) 
        # for the givven effect, first loop through the channel numbers where this effect occures
        for (i, channel) in enumerate(e_params[effect]["channel"][sample]):
            print("//////////////////")            
            print("Channel: ", channel)
            
            # for each occurance of this effect loop through all the parameters of this effect 
            for param_type in e_params[effect]:
                # skip the channel list
                if param_type not in ["channel", "phaseshift", "index", "interval"]:
                    print("----------------------")

                    print("Parameter: ", param_type)
                    print("label shape: ", labels.shape)

                    # look for the next empty slot in the labels tensor
                    next_idx = get_next_empty_index(labels[sample][channel][effect_n])
#                     print("Correspending tensor: ", labels[sample][channel][effect_n]) 
                    print("Next Index: ", next_idx)

                    #get the value of the parameter
                    val = e_params[effect][param_type][sample][i]                  

                    # if the parameter is the index where it happens (date string) transform it to a an int
#                     if param_type == "index": val = get_index_from_date(val)
                    
                    print("Value:", val)
                    if isinstance(val, np.str_):
                        print("here")
                        val = np.inf

                    #fill the labels tensor
                    labels[sample][channel][effect_n][next_idx] = val
                
def get_max_occ(effects):
    max_occ = 0
    for effect in effects:
        if effects[effect]["occurances"] > max_occ:
            max_occ = effects[effect]["occurances"]

    return max_occ
@suppress_prints
# def squeeze_labels(labels):
#     new_labels = []
#     idxs = labels.nonzero()
    
#     for index in idxs:
#         print(index)
#         sample = index[0]
#         ch = index[1]    
#         new_val = [sample, ch, labels[index[0], index[1], index[2], index[3]]]
#         new_labels.append(new_val)
        
#     new_labels = torch.tensor(new_labels)
#     return new_labels
def squeeze_labels_per_channel(labels, args, max_occ):
    
    # New labels of shape [Samples, Channels, values]
    new_labels = torch.zeros(labels.shape[0], args.n_channels, max_occ)
    labels = labels.view(labels.shape[0], labels.shape[1], -1)
    print(labels.shape)
    
    # extract the relevant values
    for i, sample in enumerate(labels):
        for j, channel in enumerate(sample):
            idxs = channel.nonzero()
            for k, idx in enumerate(idxs):
                new_labels[i, j, k] = channel[idxs[k]]
    # remove extra zeros to reduce last dim of new labels
    max_count = (new_labels != 0).sum(dim=2).max()
    new_labels = new_labels[...,:max_count].view(labels.shape[0], -1)     
    
    return new_labels
@suppress_prints
def squeeze_labels(labels, args, max_occ):
    # New labels of shape [Samples, values]
    new_labels = torch.zeros(labels.shape[0], args.n_channels * max_occ)
    labels = labels.view(labels.shape[0], -1)
    
    # extract the relevant values
    for i, sample in enumerate(labels):
        idxs = sample.nonzero()
        for k, idx in enumerate(idxs):
            new_labels[i, k] = sample[idx]
            
    # remove extra zeros to reduce last dim of new labels
    max_count = (new_labels != 0).sum(dim=1).max()
    new_labels = new_labels[...,:max_count]     
    
    return new_labels

def extract_parameters(args, e_params, effects):
    # create the labels tensor
    n_effects = len(effects)
    max_occ = 4 * get_max_occ(effects)
    n_channels = args.n_channels
    n_samples = args.n_samples * args.samples_factor
    if n_samples==None:
        labels = torch.zeros((n_channels, n_effects, max_occ))
    else:
        labels = torch.zeros((n_samples, n_channels, n_effects, 4 * max_occ))
    
    # loop through the effects and extract their paramaters
    for (effect_n, effect) in enumerate(e_params):
        if effect != "Channels_Coupling" and e_params[effect]["channel"] !=[]:
            extract_param_per_effect(labels, e_params, effect_n, effect)
    labels = squeeze_labels(labels, args, max_occ)    
        
    return labels

def reshape_params(parameter):
    # Get the shape of the original array
    original_shape = parameter.shape

    # Get the number of rows and columns in the original array
    num_rows, num_cols = original_shape

    # Create indices arrays
    row_indices = torch.arange(num_rows).unsqueeze(1).repeat(1, num_cols)
    col_indices = torch.arange(num_cols).unsqueeze(0).repeat(num_rows, 1)

    # Flatten the original array and concatenate indices
    flat_array = parameter.flatten()
    indices_array = torch.stack((row_indices.flatten(), col_indices.flatten()), dim=1)

    # Concatenate indices array with the flat array
    combined_array = torch.cat((indices_array, flat_array.unsqueeze(1)), dim=1)
    
    return combined_array

def add_mu_std(labels, params): 
    mu = torch.FloatTensor(params["mu"]).mean(dim=2)
    mu = reshape_params(mu)

    std = torch.FloatTensor(params["cov"]).mean(dim=2)
    std = reshape_params(std)

    mu_std = torch.cat((mu, std), dim=0).view(-1,3)
    labels = torch.cat((mu_std, labels), dim=0)

    labels = labels[labels[:, 1].sort(stable=True)[1]]
    labels = labels[labels[:, 0].sort(stable=True)[1]]
    n_samples = len(labels[:,0].unique())

    labels = torch.tensor_split(labels, n_samples, dim=0)
    labels = torch.stack(labels)[..., 1:]
    
    return labels