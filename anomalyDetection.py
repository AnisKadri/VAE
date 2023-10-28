import torch
import numpy as np
import matplotlib.pyplot as plt
from dataGen import Gen, FastGen, Gen2
from train import *
from utils import *
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
import umap
import scipy.stats as st
from torch.nn.functional import normalize
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, roc_auc_score, auc
import copy


def sample_anomaly(X_normal, anomaly_type, anomaly):
    X = copy.deepcopy(X_normal)
    if anomaly["occurances"] == 0:
        return
    if anomaly_type == "Pulse":
        X.add_pulse(anomaly)
    elif anomaly_type == "Trend":
        X.add_trend(anomaly)
    elif anomaly_type == "Seasonality":
        X.add_seasonality(anomaly)
    elif anomaly_type == "std_variation":
        X.add_std_variation(anomaly)
    elif anomaly_type == "channels_coupling":
        X.add_channels_coupling(anomaly)
    X.sample()
    return X
    
def add_anomaly_long(X, args, anomaly_type, anomaly, periode_factor=182, return_gen=False):
    args.periode *= periode_factor
    n_samples= args.n_samples
    args.n_samples = 1
    
    X_long = sample_anomaly(X, anomaly_type, anomaly)
    
    x_long, params_long, e_params_long = X_long.parameters()
    X_long.show()
    args.periode = args.periode // periode_factor
    args.n_samples = n_samples

    train_data_long, val_data_long, test_data_long = create_loader_Window(x_long.squeeze(0), args=args)
    
    if return_gen:
        return train_data_long, val_data_long, test_data_long, X_long
    else:
        return train_data_long, val_data_long, test_data_long
        
def get_pulse_anomalies_labels_mask(X, args):
    
    x, params, e_params = X.parameters()
    
    if e_params['Pulse']["channel"] == []:
        print("There's no Pulse Anomalies in the given data")
        return

    channels = e_params['Pulse']["channel"][0]
    indexes = e_params['Pulse']["index"][0]
    
    #Create a mask for pulse anomalies
    mask = np.full((args.n_channels, x.shape[-1]), False)
    mask[channels, indexes] = True
    
    return mask

def get_other_anomalies_labels_mask(X, X_an, args):
    
    x, params, e_params = X.parameters()
    x_an, params_an, e_params_an = X_an.parameters()
    
    an = np.abs(x_an-x)
    mask = np.full((args.n_channels, x_an.shape[-1]), False)

    for channel in range(args.n_channels):
        anomalies = np.where(an[:, channel] > 5)[1]
        mask[channel, anomalies] = True
    
    return mask
def get_anomalies_label_mask(X, X_an, args):
    
    pulse_mask = get_pulse_anomalies_labels_mask(X, args)
    other_mask = get_other_anomalies_labels_mask(X, X_an, args)
    anomalies_mask = pulse_mask | other_mask
    
    return anomalies_mask

def get_anomalies_label(mask, args, n, test=True):
    mask = mask[:, -n:] if test else mask[:, :n]
    
    # Reduce the size of mask (grouped values)
    reshaped_mask = mask.reshape(args.n_channels, -1, 1)
    
    # Get labels
    labels = np.any(reshaped_mask, axis=2)
    return labels



def get_anomalies_prediction(X, model, data, args, norm=True, threshold=0.03, group_size=3):
    
    x, params, e_params = X.parameters()
    Origin, REC, _ = rebuild_TS_non_overlapping(model, data, args, keep_norm=norm)
#     plt.plot(Origin.shape)
#     plt.plot(REC)
    
    # Calculate rec loss
    rec_loss = ((REC - Origin)**2).cpu()
    
    #Create a mask for the anomalies
    anomalies_mask = np.full(Origin.T.shape, False)
    
    for channel in range(args.n_channels):
        anomalies = np.where(rec_loss[:, channel] > threshold)[0]
        anomalies_mask[channel, anomalies] = True

    # Reduce the size of mask (grouped values)
    anomalies_reshaped_mask = anomalies_mask.reshape(args.n_channels, -1, group_size)
    predictions = np.any(anomalies_reshaped_mask, axis=2)
    
    return predictions

def get_rec_loss(model, data, args, keep_norm=True):
    Origin, REC, _ = rebuild_TS_non_overlapping(model, data, args, keep_norm=keep_norm)
    rec_loss = (REC - Origin)**2
    return rec_loss

def get_y_scores(rec_loss, args):
    y_scores = np.empty((args.n_channels, rec_loss.shape[0]))
    for i in range(args.n_channels):
        rec =rec_loss[:,i].reshape(-1, 1).cpu().numpy()
        gmm = GaussianMixture(n_components=2, random_state=0)  
        gmm.fit(rec)

        # Get the estimated probabilities of being in each component
        outlier_probabilities = gmm.predict_proba(rec)[:, 0]  # Probability of being in the first component
        y_scores[i] = 1-outlier_probabilities
    return y_scores

def get_anomalies_from_y_scores(y_scores, args):
    #Create a mask for the anomalies
    anomalies_mask = np.full(y_scores.shape, False)

    for i in range(args.n_channels):
        anomalies = np.where(y_scores[i] > 0.99)[0]
        anomalies_mask[i, anomalies] = True

    return anomalies, anomalies_mask

def get_anomalies_sector(anomalies, max_length=500):
    some_anomalies = np.where(anomalies<max_length)[0]
    sector_anomalies = anomalies[some_anomalies]
    return sector_anomalies