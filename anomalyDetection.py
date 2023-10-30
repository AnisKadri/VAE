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
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection


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
    
class AD:
    def __init__(self, model, X, X_an, args, data, 
                 norm=True, 
                 n_comp=2,
                 true_anomalies_threshold=5, 
                 gaussian_threshold=0.9,
                 k=500):
        
        self.model = model
        self.X = X                     # Length of time series in days.
        self.X_an = X_an                           # Messing tackt in mins.
        self.args = args                             # Max value when generating the time series.
        self.data = data               # Number of channels in each time series.
        self.norm = norm
        self.n_comp = n_comp
        self.true_anomalies_threshold = true_anomalies_threshold
        self.gaussian_threshold = gaussian_threshold
        self.k = k
        
        self.n = X.n
        self.n_channels = args.n_channels
        train_split, test_split = args.split
        self.train_split, self.test_split = int(train_split * self.n), int(test_split * self.n)
        self.train_split = self.train_split - self.train_split % args.L
        self.test_split_end = self.n -(self.n - self.test_split) % args.L # drop last batch
             
        
        self.x, self.params, self.e_params = X.parameters()
        self.x_an, self.params_an, self.e_params_an = X_an.parameters()
        
        self.true_mask_full = self.get_true_mask()  
        
        self.pred_mask, self.y_scores = self.get_pred_mask(model, data, args)
        
        self.n_pred = self.pred_mask.shape[-1]
        self.true_mask = self.select_from_true_mask()
        
        label = self.get_model_label(model)  
        
        self.rec_loss = self.get_rec_loss()
        self.plot_rec_loss(self.rec_loss)
        
        self.fig, self.roc_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        self.plot_roc(self.true_mask, self.pred_mask, self.y_scores, label)
        
    def get_model_label(self, model):
        
        m_mod = model._modified
        m_red = model._reduction
        m_rob = model._robust
        
        l_type = model.model_type
        l_mod = ": modified-" if m_mod else ": non modified-"
        l_red = "reduced-" if m_red else "non reduced-"
        l_rob = "robust" if m_rob else "non robust"
        
        label = l_type + l_mod + l_red + l_rob
        
        return label
        
    def plot_roc_compare(self, model, args):
        
        new_pred_mask, new_y_scores = self.get_pred_mask(model, self.data, args)
        label = self.get_model_label(model) 
        
        self.plot_roc(self.true_mask, new_pred_mask, new_y_scores, label)
        self.fig
        
    
     
    def plot_roc(self, true_mask, pred_mask, y_scores, model_label=""):
        
        for i, (label, pred) in enumerate(zip(true_mask, pred_mask)):

            accuracy = accuracy_score(label, pred)
            print("Accuracy: ", accuracy)
            recall = recall_score(label.T, pred.T, average='macro')
            print("Recall: ", recall)
            f1 = f1_score(label.T, pred.T, average='macro')
            print("F1 score: ", f1)
            
        
        true_merged = np.any(true_mask, axis=0)
        pred_merged = np.any(pred_mask, axis=0)
        y_scores_merged = np.max(y_scores, axis=0)    
        print(model_label, true_merged.shape, pred_merged.shape, y_scores_merged.shape)

        fpr, tpr, thresholds = roc_curve(true_merged, y_scores_merged, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)   
        
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        self.roc_ax.plot(fpr, tpr, lw=2, label= model_label + f': (AUC = {roc_auc:.2f})')
        self.roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        self.roc_ax.set_xlim([0.0, 1.0])
        self.roc_ax.set_ylim([0.0, 1.05])
        self.roc_ax.set_xlabel('False Positive Rate')
        self.roc_ax.set_ylabel('True Positive Rate')
        self.roc_ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        self.roc_ax.legend(loc='lower right')
        plt.show()
        
    def plot_true_pred_anomalies(self, start=0, end=500):
        
        if start >= end:
            print("Start > end")
            return
        start = sorted((0, start, end))[1]
        end = sorted((start, end, self.n))[1]
        
        x_plot_origin = self.Origin[start:end]
        x_plot_rec = self.REC[start:end]
        
        true_mask_plot = self.true_mask[..., start:end]        
        true_idxs_plot = self.get_index_from_mask(true_mask_plot)
        
        pred_mask_plot = self.pred_mask[..., start:end]        
        pred_idxs_plot = self.get_index_from_mask(pred_mask_plot) 
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 10))
        
        line_handles = []
        scatter_handles = []
        
        for channel in range(self.n_channels):
            if channel == 0:
                line1, = ax[0].plot(x_plot_origin[:, channel], 
                                    "b", alpha=0.7, label='Original TS')
                line2, = ax[0].plot(x_plot_rec[:, channel],
                                    "--r", label='Reconstructed TS')
                scatter1 =ax[0].scatter(true_idxs_plot[channel], x_plot_origin[true_idxs_plot[channel], channel],
                                        c="r", alpha=1, label='True Anomalies')
                scatter2= ax[0].scatter(pred_idxs_plot[channel], x_plot_origin[pred_idxs_plot[channel], channel],
                                        c="g", marker='x', alpha=0.7, label='Predicted Anomalies')
            else:
                line1, = ax[0].plot(x_plot_origin[:, channel], "b", alpha=0.7)
                line2, = ax[0].plot(x_plot_rec[:, channel], "--r")
                scatter1 =ax[0].scatter(true_idxs_plot[channel], x_plot_origin[true_idxs_plot[channel], channel],
                                        c="r", alpha=1)
                scatter2= ax[0].scatter(pred_idxs_plot[channel], x_plot_origin[pred_idxs_plot[channel], channel],
                                        c="g", alpha=0.7, marker='x')              
            
        ax[0].set_title("True anomalies vs predicted anomalies")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)
        
        ax[1].plot(true_mask_plot.T, "b", alpha=0.7, label="True anomalies mask")
        ax[1].plot(pred_mask_plot.T, "--r", label="Predicted anomalies mask")
        ax[1].set_title("True anomalies Mask vs Predicted Mask")
        ax[1].legend(loc='upper right')
        ax[1].grid()
        
        plt.show()
        
    def plot_true_anomalies(self, start=0, end=500):
        
        if start >= end:
            print("Start > end")
            return
        start = sorted((0, start, end))[1]
        end = sorted((start, end, self.n))[1]
        
        x_plot = self.x_an[...,start:end].squeeze(0).T
        
        true_mask_plot = self.true_mask_full[..., start:end]        
        true_idxs_plot = self.get_index_from_mask(true_mask_plot)
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
        
        for channel in range(self.n_channels):
            ax[0].plot(x_plot[:,channel], "b", alpha=0.7, label="Original TS")
            ax[0].scatter(true_idxs_plot[channel], x_plot[true_idxs_plot[channel], channel], c="r", marker="x", label="Anomalies")
        ax[0].set_title("Anomalies on the Original TS")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)
        
        ax[1].plot(true_mask_plot.T)
        ax[1].set_title("Anomalies Mask")
        
        plt.show()
        
    def plot_y_scores(self,rec_loss, y_scores):
        scale_factor = 1#/torch.max(rec_loss, axis=0)[0]
        rec_loss_rescaled = scale_factor * rec_loss
        
         # Plotting the scores
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 4))

        for channel in range(self.n_channels):
            ax.plot(rec_loss_rescaled[:self.k, channel], "orange", alpha=1, label="Rec loss channel")# {}".format(channel+1))
        for channel in range(self.n_channels):
            ax.plot(y_scores.T[:self.k, channel], "--r", alpha=0.8, label="y_score channel")# {}".format(channel+1))

        ax.legend(loc="upper right")
        ax.grid()
        ax.set_title("Rec loss vs y_scores from GM")
        
        plt.show()
        
    def plot_rec_loss(self,rec_loss, start=0, end=500):
        
        if start >= end:
            print("Start > end")
            return
        start = sorted((0, start, end))[1]
        end = sorted((start, end, self.n))[1]
        
        x_plot_origin = self.Origin[start:end]
        x_plot_rec = self.REC[start:end]
        x_rec_loss = rec_loss[start:end]
        
         # Plotting the scores
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 10))
        
        line_handles = []
        scatter_handles = []
        
        for channel in range(self.n_channels):
            if channel == 0:
                line1, = ax[0].plot(x_plot_origin[:, channel], 
                                    "b", alpha=0.7, label='Original TS')
                line2, = ax[0].plot(x_plot_rec[:, channel],
                                    "--r", label='Reconstructed TS')
                line3, = ax[1].plot(x_rec_loss[:, channel],
                                    "orange", label='Reconstruction loss')
            else:
                line1, = ax[0].plot(x_plot_origin[:, channel], "b", alpha=0.7)
                line2, = ax[0].plot(x_plot_rec[:, channel], "--r")
                line3, = ax[1].plot(x_rec_loss[:, channel], "orange")


        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        ax[0].grid()
        ax[0].set_title("Original data vs Reconstruction")
        ax[1].set_title("Reconstruction loss")
        plt.grid()
        
        plt.show()
        
    def get_index_from_mask(self, mask):
        
        idxs = []
        for channel_mask in mask:
            index = np.where(channel_mask > 0)[0]
            idxs.append(index)
            
        return idxs
    
    # Input Shape of mask and scores is [channels, scores]
    def fill_mask(self, scores, mask, threshold):
        
        for channel in range(self.n_channels):
            anomalies = np.where(scores[channel] > threshold)[0]
            mask[channel, anomalies] = True
            
        return mask
        
    def get_pulse_anomalies_labels_mask(self):
    
        if self.e_params['Pulse']["channel"] == []:
            print("There's no Pulse Anomalies in the given data")
            return

        channels = self.e_params['Pulse']["channel"][0]
        indexes = self.e_params['Pulse']["index"][0]

        #Create a mask for pulse anomalies
        mask = np.full((self.n_channels, self.n), False)
        mask[channels, indexes] = True

        return mask
            
    def get_other_anomalies_labels_mask(self):        

        an = np.abs(self.x_an-self.x)[0]
        
        mask = np.full((self.n_channels, self.n), False)        
        mask = self.fill_mask(an, mask, threshold=self.true_anomalies_threshold)

        return mask
    
    def get_true_mask(self):

        pulse_mask = self.get_pulse_anomalies_labels_mask()
        other_mask = self.get_other_anomalies_labels_mask()
        true_mask = pulse_mask | other_mask

        return true_mask
    
    def test(self):
        
        test=False        
        if (self.n_pred / self.n) <= 0.4:
            test = True
        else:
            test = False
            
        return test

    def select_from_true_mask(self):
        
        test = self.test()        
        if self.test():
            selected = self.true_mask_full[:, self.test_split : self.test_split_end] 
        else:
            selected = self.true_mask_full[:, :self.train_split]
        
#         print(self.n, self.test_split, self.test_split_end)

        return selected

    def get_rec_loss(self):
        rec_loss = (self.REC - self.Origin)**2
        return rec_loss

    def get_y_scores(self, rec_loss, n_comp):
        
        rec_shape = rec_loss.shape[0]
        y_scores = np.empty((self.n_channels, rec_loss.shape[0]))
        
        for i in range(self.n_channels):
            rec = rec_loss[:,i].reshape(-1, 1)
            gmm = GaussianMixture(n_components=n_comp, random_state=0)  
            gmm.fit(rec)

            # Get the estimated probabilities of being in each component
            observations = gmm.predict_proba(rec)[:, 0]
            observations_inverse = gmm.predict_proba(rec)[:, 1]
            ones, ones_inverse = np.sum(observations == 1), np.sum(observations_inverse == 1)
            y_scores[i] = observations if ones >= ones_inverse else observations_inverse
#             y_scores[i] = gmm.predict_proba(rec)[:, 1]
        
            
        return y_scores

    def get_mask_from_y_scores(self, y_scores):
        #Create a mask for the anomalies
        pred_mask = np.full(y_scores.shape, False)
        pred_mask = self.fill_mask(y_scores, pred_mask, threshold=self.gaussian_threshold)        

        return pred_mask
    
    def get_pred_mask(self, model, data, args, norm=True):
        
        Origin, REC, _ = rebuild_TS_non_overlapping(model, data, args, keep_norm=norm)
        self.Origin, self.REC = Origin.cpu(), REC.cpu()
        
        rec_loss = self.get_rec_loss()
        y_scores = self.get_y_scores(rec_loss, n_comp=self.n_comp)
        pred_mask = self.get_mask_from_y_scores(y_scores)
        
#         self.plot_y_scores(rec_loss, y_scores)
    
#         fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(12, 6))
#         ax.plot(pred_mask.T[:self.k], "b", alpha=0.4, label="Pred Mask")
#         ax.plot(y_scores.T[:self.k], "--r", alpha=0.4, label="y_score")
        
#         ax.legend(loc="upper right")
#         ax.set_title("Prediction Mask vs y_scores")
#         plt.show()
        
        return pred_mask, y_scores

#     def get_anomalies_sector(anomalies, max_length=500):
#         some_anomalies = np.where(anomalies<max_length)[0]
#         sector_anomalies = anomalies[some_anomalies]
#         return sector_anomalies


        
# def get_pulse_anomalies_labels_mask(X, args):
    
#     x, params, e_params = X.parameters()
    
#     if e_params['Pulse']["channel"] == []:
#         print("There's no Pulse Anomalies in the given data")
#         return

#     channels = e_params['Pulse']["channel"][0]
#     indexes = e_params['Pulse']["index"][0]
    
#     #Create a mask for pulse anomalies
#     mask = np.full((args.n_channels, x.shape[-1]), False)
#     mask[channels, indexes] = True
    
#     return mask

# def get_other_anomalies_labels_mask(X, X_an, args):
    
#     x, params, e_params = X.parameters()
#     x_an, params_an, e_params_an = X_an.parameters()
    
#     an = np.abs(x_an-x)
#     mask = np.full((args.n_channels, x_an.shape[-1]), False)

#     for channel in range(args.n_channels):
#         anomalies = np.where(an[:, channel] > 5)[1]
#         mask[channel, anomalies] = True
    
#     return mask
# def get_anomalies_label_mask(X, X_an, args):
    
#     pulse_mask = get_pulse_anomalies_labels_mask(X, args)
#     other_mask = get_other_anomalies_labels_mask(X, X_an, args)
#     anomalies_mask = pulse_mask | other_mask
    
#     return anomalies_mask

# def get_anomalies_label(mask, args, n, test=True):
#     mask = mask[:, -n:] if test else mask[:, :n]
    
#     # Reduce the size of mask (grouped values)
#     reshaped_mask = mask.reshape(args.n_channels, -1, 1)
    
#     # Get labels
#     labels = np.any(reshaped_mask, axis=2)
#     return labels



# def get_anomalies_prediction(X, model, data, args, norm=True, threshold=0.03, group_size=3):
    
#     x, params, e_params = X.parameters()
#     Origin, REC, _ = rebuild_TS_non_overlapping(model, data, args, keep_norm=norm)
# #     plt.plot(Origin.shape)
# #     plt.plot(REC)
    
#     # Calculate rec loss
#     rec_loss = ((REC - Origin)**2).cpu()
    
#     #Create a mask for the anomalies
#     anomalies_mask = np.full(Origin.T.shape, False)
    
#     for channel in range(args.n_channels):
#         anomalies = np.where(rec_loss[:, channel] > threshold)[0]
#         anomalies_mask[channel, anomalies] = True

#     # Reduce the size of mask (grouped values)
#     anomalies_reshaped_mask = anomalies_mask.reshape(args.n_channels, -1, group_size)
#     predictions = np.any(anomalies_reshaped_mask, axis=2)
    
#     return predictions

# def get_rec_loss(model, data, args, keep_norm=True):
#     Origin, REC, _ = rebuild_TS_non_overlapping(model, data, args, keep_norm=keep_norm)
#     rec_loss = (REC - Origin)**2
#     return rec_loss

# def get_y_scores(rec_loss, args):
#     y_scores = np.empty((args.n_channels, rec_loss.shape[0]))
#     for i in range(args.n_channels):
#         rec =rec_loss[:,i].reshape(-1, 1).cpu().numpy()
#         gmm = GaussianMixture(n_components=2, random_state=0)  
#         gmm.fit(rec)

#         # Get the estimated probabilities of being in each component
#         outlier_probabilities = gmm.predict_proba(rec)[:, 0]  # Probability of being in the first component
#         y_scores[i] = 1-outlier_probabilities
#     return y_scores

# def get_anomalies_from_y_scores(y_scores, args):
#     #Create a mask for the anomalies
#     anomalies_mask = np.full(y_scores.shape, False)

#     for i in range(args.n_channels):
#         anomalies = np.where(y_scores[i] > 0.99)[0]
#         anomalies_mask[i, anomalies] = True

#     return anomalies, anomalies_mask

# def get_anomalies_sector(anomalies, max_length=500):
#     some_anomalies = np.where(anomalies<max_length)[0]
#     sector_anomalies = anomalies[some_anomalies]
#     return sector_anomalies