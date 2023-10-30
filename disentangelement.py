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
import umap
import scipy.stats as st
from torch.nn.functional import normalize
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy
TINY = 1e-12
# from hinton import *

class identifier(nn.Module):
    def __init__(self, args, n_layers, output_size):
        super(identifier, self).__init__()   
        
        self.n_channels =  args.n_channels      
        self.slope = args.slope
        self.n_layers = n_layers
        self.input_size = args.enc_out * args.latent_dims
        self.output_size = output_size 
        self.lin_layers = nn.ModuleList()
#         self.n = 1
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)
                m.bias.data.fill_(0.01)
                
        # Linear Layers
        for i in range(0, n_layers):
            
            if i == self.n_layers -1:
                self.lin_layers.append(nn.Linear(self.input_size, self.output_size))
                self.lin_layers.append(nn.ReLU(True))
                self.lin_layers.append(nn.BatchNorm1d(self.output_size))
            else:
                self.lin_layers.append(nn.Linear(self.input_size, self.input_size))
                self.lin_layers.append(nn.ReLU(True))
                self.lin_layers.append(nn.BatchNorm1d(self.output_size))
        
        self.lin_layers.apply(init_weights)
            
         
    def forward(self, x):
#         print("x input", x.shape)
#         x_shape = x.shape
        x = x.view(x.size(0), -1)
#         print("x before lin", x.shape)
        for i, lin in enumerate(self.lin_layers):
            x = lin(x)        
#         print("x after lin", x.shape)
#         print("x out", x.shape)
#         x = x.view(x.shape[0], -1, self.n_channels)
        return x

def train_identifier_modified(model, id_model, train_loader, optimizer, args, epoch):
    
    device=args.device
    labels_size = id_model.output_size
    n_channels = args.n_channels
    
    id_model.train()
    for p in id_model.parameters():
        p.requires_grad = True
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    train_loss = 0
    y_pred = torch.empty((0, labels_size), device=device)
    for batch_idx, (data_tup, label, norm) in enumerate(train_loader):

        data_tup = [data.to(device) for data in data_tup]
        data = pick_data(data_tup, args)
        norm = [n.to(device) for n in norm]
        bs   = data.shape[0]  
        
        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e, indices = model(data, ouput_indices=True)
        label_pred = id_model(e)

        optimizer.zero_grad()
#         label = normalize(label, p=1, dim=0).to(device)
        loss = F.mse_loss(label_pred, label.to(device), reduction='mean')

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        y_pred = torch.cat((y_pred, label_pred), dim = 0) 
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    print("True Loss: ", train_loss)

    return y_pred



def calc_P_matrix(R):
    # Calculate the denominator: PK-1 Rik for each row
    denominator = np.sum(R, axis=1)
#     print(denominator)

    # Ensure there are no division by zero errors
    denominator[denominator == 0] = 1

    # Calculate the P matrix using broadcasting
    P = R / denominator[:, np.newaxis]

    return P

def calc_H(P):

    # Calculate the entropy formula using broadcasting
    K = P.shape[1]
    D = P.shape[0]
#     H = -np.sum(P * np.log(K) * P, axis=1)
    H = np.empty(D)
    for i in range(D):
#         print(entropy(P[i], base=K))
        H[i] = entropy(P[i], base=K)

    return H
def calc_D(H):
    return 1-H
def norm_entropy(p):
    '''p: probabilities '''
    n = p.shape[0]
    return - p.dot(np.log(p + TINY) / np.log(n + TINY))

def entropic_scores(r):
#     '''r: relative importances '''
    r = np.abs(r)
    ps = r / np.sum(r, axis=0) # 'probabilities'
    hs = [1-norm_entropy(p) for p in ps.T]
    return hs

def print_table_pretty(name, values, factor_label, model_names):
    headers = [factor_label + str(i) for i in range(len(values[0]))]
    headers[-1] = "Avg."
    headers = "\t" + "\t".join(headers)
    print("{0}:\n{1}".format(name, headers))
    
    for i, values in enumerate(values):
        value = ""
        for v in values:
            value +=  "{0:.2f}".format(v) + "&\t"
        print("{0}\t{1}".format(model_names[i], value))
    print("")
def hinton(matrix, title, max_weight=None, ax=None):
    
    if ax is None:
        ax = plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    
    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    # Add labels to the axes
    ax.set_xticks(np.arange(matrix.shape[0]), minor=False)
    ax.set_yticks(np.arange(matrix.shape[1]), minor=False)
    ax.set_xticklabels(range(1, matrix.shape[0] + 1))
    ax.set_yticklabels(range(1, matrix.shape[1] + 1))
    ax.set_ylabel('C')
    ax.set_xlabel('Z')
    ax.set_title(title)

def scores_cd(R):
    R= np.abs(R)

    disent_scores = entropic_scores(R)
    c_rel_importance = np.sum(R.T,1) / np.sum(R) # relative importance of each code variable
    disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)
#     disent_scores.append(disent_w_avg)
#     print("Disentangelement scores: ", disent_scores)
#     m_disent_scores.append(disent_scores)
    
    # completeness
    complete_scores = entropic_scores(R.T)
    complete_avg = np.mean(complete_scores)
    
    return disent_scores, complete_scores
    
def importance_codes(R):
    R= np.abs(R)

    D_coefs = R.sum(axis=1)/R.sum()
    C_coefs = R.sum(axis=0)/R.sum()

    Pi = calc_P_matrix(R)
    Pj = calc_P_matrix(R.T)

    Hd = calc_H(Pi)
    Hc = calc_H(Pj)
    
    D = calc_D(Hd)# * D_coefs
    C = calc_D(Hc)# * C_coefs


    diagram = D[:, np.newaxis] * C
    important_codes = np.argmax(diagram, axis=0)
    
#     plt.figure(figsize=(6, 10))    
#     hinton(R.T, title, ax=ax)
#     plt.show()
    
    return important_codes
        
def extract_id_data(model, train_loader, args, norm_labels=True, labels_size=3):
    device=args.device
    n_channels = args.n_channels

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    X = torch.empty((0, args.enc_out, args.latent_dims), device = device) 
    Y = torch.empty((0, labels_size), device=device)

    for batch_idx, (data_tup, label, norm) in enumerate(train_loader):

        data = pick_data(data_tup, args)
        norm = [n.to(device) for n in norm]
        bs   = data.shape[0]  
        label = label.to(device)
        
        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e, indices = model(data, ouput_indices=True)
        
        # Save X and y 
        X = torch.cat((X, e), dim=0)
        Y = torch.cat((Y, label), dim=0)
        
    # Reshape and normalize
    X = X.view(X.shape[0], -1)
    if norm_labels:
        X = normalize(X, p=1, dim=0)
        Y = normalize(Y, p=1, dim=0)

    return X.cpu().detach().numpy(), Y.cpu().detach().numpy()

def get_loss(model, id_model, train_loader, args, norm_labels=True, labels_size=3):
    device=args.device
    n_channels = args.n_channels

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    X = torch.empty((0, args.enc_out, args.latent_dims), device = device) 
    Y = torch.empty((0, labels_size), device=device)
    Loss =[]

    for batch_idx, (data_tup, label, norm) in enumerate(train_loader):

        data = pick_data(data_tup, args)
        norm = [n.to(device) for n in norm]
        bs   = data.shape[0]  
        label = label.to(device)
        
        x_rec, loss, mu, logvar, mu_rec, logvar_rec, e, indices = model(data, ouput_indices=True)
        
        label_pred = id_model(e)

#         label = normalize(label, p=1, dim=0).to(device)
        loss = F.mse_loss(label_pred, label.to(device), reduction='mean')
        Loss.append(loss.item())
        
#         # Save X and y 
#         X = torch.cat((X, e), dim=0)
#         Y = torch.cat((Y, label), dim=0)
        
#     # Reshape and normalize
#     X = X.view(X.shape[0], -1)
#     if norm_labels:
#         X = normalize(X, p=1, dim=0)
#         Y = normalize(Y, p=1, dim=0)

    return Loss

def get_R_from_forest(X, y, n_estimators=10):
    R = []
    for i in range(y.shape[-1]):
        y_values = y[:, i]
        # Create a random forest model with the desired number of estimators
        rf_model = RandomForestRegressor(n_estimators=n_estimators)
        rf_model.fit(X, y_values) 

        # Get feature importances for this y value
        feature_importances = rf_model.feature_importances_
        R.append(feature_importances)
        
    return np.array(R)