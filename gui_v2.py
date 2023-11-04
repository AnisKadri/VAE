import torch
import numpy as np
from torch.utils.data import DataLoader
from tkinter.messagebox import showinfo
import tkinter as tk
from tkinter import ttk
from tkinter import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from VQ_EMA_fn import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Cursor
from copy import deepcopy
from sklearn.metrics import mutual_info_score
import warnings
warnings.simplefilter("ignore", UserWarning)

class VQ_gui(tk.Tk):
  @torch.no_grad()
  def __init__(self, model, data, args, repetition, VQ):
    super().__init__()
    # Init variable vor the Gui
    self.model, self.data, self.args, self.repetition, self.VQ = model, data, args, repetition, VQ
    self.model.eval()
    self.spinboxs, self.double_vars = [], []
    self.n = 0


    # Get the model parameters
    # self.code_book      = deepcopy(self.avg_latents)
    # self.code_book_np   = self.code_book.detach().numpy()
    self.changed = []

    # Sample x, x_rec and the latent space
    self.x, self.x_rec, self.latents = rebuild_TS(model, data, args, keep_norm=False)
    if VQ:
        self.codebook = model.quantizer._embedding.weight
        indices = self.latent.mean(axis=-1).type(torch.int32)
        self.latents = self.codebook.index_select(0, indices)
    self.latent = self.latents[n]

    # create window and frames layout
    button_frame, grid_frame, heatmap_frame, plot_frame, latent_plot_frame= self.create_frames()

    # Create the text input widgets
    self.create_spinboxes(grid_frame)

    # create buttons for Reset and Save
    self.create_buttons(button_frame)

    # create plot
    self.ax_plot, self.plot_canvas = self.create_plot(plot_frame, self.plot_reconstruction, self.x, self.x_rec, self.n)
    # self.latent_plot, self.plot_canvas = self.create_plot(latent_plot_frame, self.plot_latents, self.mu, self.logvar, self.norm)

    # create heatmap
    self.ax_heatmap, self.heatmap, self.heatmap_canvas = self.create_heatmap(heatmap_frame)

    # create scatter
    # self.ax_scatter, self.scatter, self.scatter_canvas = self.create_scatter(scatter_frame, 10)

  def get_average_norm_scale(self,train_data, model):
      n_channels = model._n_channels
      norm = torch.empty(n_channels, 0)

      for i, (data, norm_scale) in enumerate(train_data):
        reshaped_norm = norm_scale.permute(1, 0, 2).flatten(1)
        norm = torch.cat((norm, reshaped_norm), 1)

      avg_norm = torch.mean(norm, dim=1)
      print(norm.shape)
      return  norm, avg_norm

  def get_latent_variables(self, train_data, model):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_channels = model._n_channels
    latent_dims = model._latent_dims
    x_len = len(train_data.dataset)

    latents = torch.empty(n_channels, latent_dims, 0, device=device)

    for i, (data, norm_scale) in enumerate(train_data):
      data = data.to(device)

      mu, logvar = model.encoder(data)
      z = model.reparametrization_trick(mu, logvar)

      reshaped_mu, reshaped_logvar, reshaped_z = (t.permute(1, 2, 0) for t in [mu, logvar, z])

      latents = torch.cat((latents, reshaped_z), 2)

    avg_latents = torch.mean(latents, dim=2)
    latents = latents.view(-1, x_len)
    print("latents", latents.shape)
    return latents, avg_latents


  def create_heatmap(self, heatmap_frame):
    self.heatmap_fig = Figure(figsize=(4, 6), dpi=100)
    ax_heatmap = self.heatmap_fig.add_subplot(111)
    heatmap = self.plot_heatmap(ax_heatmap, self.latent)
    heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=heatmap_frame)
    heatmap_canvas.draw()
    heatmap_canvas.get_tk_widget().pack(fill="both", expand=True)
    self.cbar = self.heatmap_fig.colorbar(heatmap)
    ax_heatmap.set_xlabel('Num of Embeddings')
    ax_heatmap.set_ylabel('Latent Dimensions')

    return ax_heatmap, heatmap, heatmap_canvas

  def create_plot(self, plot_frame, plot_fn, *args):
    fig = Figure(figsize=(8,4), dpi=100)
    ax_plot = fig.add_subplot(111)
    plot_fn(ax_plot, *args)
    ax_plot.set_xlabel('Time')
    ax_plot.set_ylabel('Values')

    plot_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    # # create a Matplotlib navigation toolbar and add it to the Tkinter window
    toolbar = NavigationToolbar2Tk(plot_canvas, plot_frame)
    toolbar.update()

    cursor = Cursor(ax_plot, useblit=True, color='red', linewidth=1)

    # bind the Matplotlib events to the Tkinter canvas
    plot_canvas.mpl_connect("motion_notify_event", toolbar._update_cursor)
    plot_canvas.mpl_connect("scroll_event", toolbar.zoom)
    plot_canvas.mpl_connect("key_press_event", toolbar.pan)
    plot_canvas.mpl_connect("key_release_event", toolbar.pan)

    return ax_plot, plot_canvas

  def create_scatter(self, scatter_frame, k):
    fig = Figure(figsize=(6,4), dpi=100)
    ax_scatter = fig.add_subplot(111)
    # flattend = self.MI_scores.flatten()
    # n = flattend.size(0)
    #
    # # Set the x-axis to the indices of arr
    # indices = np.arange(n)
    #
    # print(n)
    # print(flattend)
    # scatter = ax_scatter.scatter(indices, flattend)
    # ax_scatter.set_xlabel('Embedding Vec')
    # ax_scatter.set_ylabel('MI Score')
    # ax_scatter.set_title("MI Score of Embeding Vec and mean")

    scatter = self.plot_MI_scores(ax_scatter, self.MI_scores, k)

    # Add interactivity
    scatter_canvas = FigureCanvasTkAgg(fig, master=scatter_frame)
    scatter_canvas.draw()
    scatter_canvas.get_tk_widget().pack(fill="both", expand=True)

    # # create a Matplotlib navigation toolbar and add it to the Tkinter window
    toolbar = NavigationToolbar2Tk(scatter_canvas, scatter_frame)
    toolbar.update()

    cursor = Cursor(ax_scatter, useblit=True, color='red', linewidth=1)

    # bind the Matplotlib events to the Tkinter canvas
    scatter_canvas.mpl_connect("motion_notify_event", toolbar._update_cursor)
    scatter_canvas.mpl_connect("scroll_event", toolbar.zoom)
    scatter_canvas.mpl_connect("key_press_event", toolbar.pan)
    scatter_canvas.mpl_connect("key_release_event", toolbar.pan)
    fig.tight_layout()

    return ax_scatter, scatter, scatter_canvas

  def create_buttons(self, button_frame):
    # Reset Button
    self.reset_button = tk.Button(button_frame, text="Reset Values")
    self.reset_button.pack(side="top", padx=5, pady=5, fill="x")
    self.reset_button.bind("<Button-1>", self.reset)

    # File name for saving
    save_label = tk.Label(button_frame, text="model_name")
    save_label.pack(side=TOP)
    self.file_name = tk.Entry(button_frame)
    self.file_name.pack(side="top", padx=5, pady=5, fill="x")


    # Save Button
    self.save_button = tk.Button(button_frame, text="Save Model")
    self.save_button.pack(side="top", padx=5, pady=5, fill="x")
    self.save_button.bind("<Button-1>", self.save)

    # parameters for loading model: n_channel
    n_channels_label = tk.Label(button_frame, text="n_channels")
    n_channels_label.pack(side=TOP)
    self.load_channels = tk.Entry(button_frame)
    self.load_channels.insert(0, "1")
    self.load_channels.pack(side="top", padx=5, pady=5, fill="x")

    window_label = tk.Label(button_frame, text="window Length")
    window_label.pack(side=TOP)
    self.load_window = tk.Entry(button_frame)
    self.load_window.insert(0, "600")
    self.load_window.pack(side="top", padx=5, pady=5, fill="x")

    effect_label = tk.Label(button_frame, text="Effect")
    effect_label.pack(side=TOP)
    self.load_effect = tk.Entry(button_frame)
    self.load_effect.insert(0, "trend")
    self.load_effect.pack(side="top", padx=5, pady=5, fill="x")

    latent_label = tk.Label(button_frame, text="Latent space Dim")
    latent_label.pack(side=TOP)
    self.load_dims = tk.Entry(button_frame)
    self.load_dims.insert(0, "7")
    self.load_dims.pack(side="top", padx=5, pady=5, fill="x")

    number_label = tk.Label(button_frame, text="Model Number")
    number_label.pack(side=TOP)
    self.load_number = tk.Entry(button_frame)
    self.load_number.insert(0, "0")
    self.load_number.pack(side="top", padx=5, pady=5, fill="x")

    sample_number = tk.Label(button_frame, text="Model Number")
    sample_number.pack(side=TOP)
    var = StringVar()
    var.set(1)
    self.sample =Spinbox(button_frame, from_=0, to=4000, increment=1, textvariable=var,
            command= self.change_sample)
    # sample = tk.Label(button_frame, text="Sample Number")
    # self.sample_n = tk.Entry(button_frame)
    # self.sample_n.insert(0, "0")
    self.sample.pack(side="top", padx=5, pady=5, fill="x")

    # Load Button
    self.load_button = tk.Button(button_frame, text="Load Model")
    self.load_button.pack(side="top", padx=5, pady=5, fill="x")
    self.load_button.bind("<Button-1>", self.load)

  def change_sample(self):
      self.n = int(self.sample.get())

      self.x, self.x_rec, self.latents = rebuild_TS(self.model, self.data, args, keep_norm=False)
      if self.VQ:
          self.codebook = self.model.quantizer._embedding.weight
          indices = self.latent.mean(axis=-1).type(torch.int32)
          self.latents = self.codebook.index_select(0, indices)
      self.latent = self.latents[n]

      self.ax_plot.clear()
      self.plot_reconstruction(self.ax_plot, self.x, self.x_rec, self.n)
      self.plot_canvas.draw()
      heatmap = self.plot_heatmap(self.ax_heatmap, self.latent)
      self.cbar.update_normal(heatmap)
      self.heatmap_canvas.draw()
      print("replotted")
      for col in range(self.args.enc_out):
          for row in range(self.args.latent_dims):
              idx = row * self.args.enc_out + col
              self.double_vars[idx].set(self.latent[col, row].item())


  def create_spinboxes(self, grid_frame):
    for col in range(self.args.enc_out):
      for row in range(self.args.latent_dims):
        self.double_var = tk.DoubleVar()
        # create the Spinbox and link it to the DoubleVar
        self.spinbox = Spinbox(grid_frame, from_=-5, to=5, increment=0.001, textvariable=self.double_var,
                               command=lambda idx=row, idy=col:  self.sample_from_codebook(idx, idy) if self.VQ else self.sample_vae(idx, idy))
        # self.spinbox.bind('<Return>', self.set_spinbox_value(row, col))
        self.spinbox.grid(row=row, column=col, sticky="NSEW")

        # set initial value
        initial_value = self.latent[col][row].item()

        self.double_var.set(initial_value)

        # add the spinbox double var (variable to change the value) to a list
        self.spinboxs.append(self.spinbox)
        self.double_vars.append(self.double_var)

        grid_frame.grid_columnconfigure(col, weight=1)
      grid_frame.grid_rowconfigure(row, weight=1)

  def create_frames(self):
    self.title('My Awesome App')
    self.geometry('1800x800')


    # create parent frame
    parent_frame = tk.Frame(self)
    parent_frame.pack(side="top", fill="both", expand=True)

    # create plot frame
    plot_frame = tk.Frame(parent_frame)
    plot_frame.grid(row=0, column=0, sticky="nsew")

    latent_plot_frame = tk.Frame(parent_frame)
    latent_plot_frame.grid(row=0, column=2, sticky="nsew")

    # create buttons frame
    button_frame = tk.Frame(parent_frame)
    button_frame.grid(row=0, column=4, sticky="nsew")

    # create bottom frame for inputs and heatmap
    bottom_frame = tk.Frame(self)
    bottom_frame.pack(side="bottom", fill="both", expand=False)

    # create left and right subframes for inputs and heatmap
    inputs_frame = tk.Frame(bottom_frame)
    inputs_frame.pack(side="left", fill="both", expand=True)
    heatmap_frame = tk.Frame(bottom_frame)
    heatmap_frame.pack(side="right", fill="both", expand=True)

    # create grid of input labels and entries
    grid_frame = tk.Frame(bottom_frame, padx=10, pady=10)
    grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    return button_frame, grid_frame, heatmap_frame, plot_frame, latent_plot_frame#, scatter_frame

  def plot_heatmap(self, ax_heatmap, codebook):
    ax_heatmap.clear()
    heatmap = ax_heatmap.imshow(codebook.T)
    ax_heatmap.set_title('Codebook Heatmap')
    ax_heatmap


    return heatmap

  def plot_MI_scores(self, ax_scatter, MI_scores, k):
    ax_scatter.clear()
    # Get the indices and values of the top 10 elements across all dimensions
    top_values, top_indices = torch.topk(MI_scores.view(-1), k=k, dim=0)

    # Reshape the indices to match the shape of MI_scores and set them in tupples
    top_indices = np.unravel_index(top_indices, MI_scores.shape)
    top_indices = list(zip(top_indices[0], top_indices[1]))

    for el in top_indices:
      col, row = el[0], el[1]
      idx = row * self.num_embed + col
      self.spinboxs[idx].configure(background='yellow')

    # define the x axis ticks
    indices = np.arange(k)
    xtick_labels = [str(t[::-1])for t in top_indices]

    # Create the Scatterplot
    scatter = ax_scatter.scatter(indices, top_values, c='y')

    ax_scatter.set_xticks(indices)
    ax_scatter.set_xticklabels(xtick_labels, rotation=45)
    ax_scatter.set_xlabel('Embedding value: (Row, Col)')
    ax_scatter.set_ylabel('MI Score')
    ax_scatter.set_title("MI Score of Embedding   and mean")
    ax_scatter.grid()

    return scatter

  def plot_reconstruction(self, ax_plot, x, x_rec, n=0):
    ax_plot.clear()
    ax_plot.plot(x[n].T, "b", alpha=0.2)

    ax_plot.plot(x_rec[n].T, "r", alpha=0.2)
    # Create custom legend handles and labels
    blue_handle = plt.Line2D([], [], color='b', label='Original Data')
    red_handle = plt.Line2D([], [], color='r', label='Reconstructions mean')
    ax_plot.set_title('Reconstruction')
    ax_plot.legend(handles=[blue_handle, red_handle], loc="upper right")
    ax_plot.grid()

  def plot_latents(self, latent_plot, mus, logvars, norm):
    mus = mus.view(-1, len(self.data.dataset))
    logvars = logvars.view(-1, len(self.data.dataset))
    print(mus.shape)
    # print(norm.shape)
    # print("latent mal norm", (latents* norm.squeeze(0)).shape)
    # print(latents)
    latent_plot.clear()
    # for latent in latents:
    latent_plot.plot((mus* norm.squeeze(0)).T, alpha = 0.3)
    latent_plot.plot((logvars * norm.squeeze(0)).T, alpha=0.3)

    # rec_lines = latent_plot.plot(x_rec.T, "orange", alpha=0.2)
    # rec_lines_mean = latent_plot.plot(x_rec_mean, "r")
    # Create custom legend handles and labels
    # blue_handle = plt.Line2D([], [], color='b', label='Original Data')
    # red_handle = plt.Line2D([], [], color='r', label='Reconstructions mean')
    latent_plot.set_title('latent_variables')
    # latent_plot.legend(handles=[blue_handle, red_handle], loc="upper right")
    latent_plot.grid()


  def set_spinbox_value(self, event, row, col):
    try:
      idx = row * self.num_embed + col
      spin = self.spinboxs[idx]
      value = int(spin.get())
      spin.get().delete(0, tk.END)
      spin.insert(0, value)
      self.sample_from_codebook(row, col)
    except ValueError:
      pass

  def save(self, event):
    torch.save(self.model, r'modules\{}.pt'.format(self.file_name.get()))
    print("Saved")

  def load(self, event):
    print(r'modules\data_{}_{}channels_{}latent_{}window_{}.pt'.format(self.load_effect.get(), self.load_channels.get(), self.load_dims.get(), self.load_window.get(), self.load_number.get()))
    try:
      x = torch.load(r'modules\data_{}_{}channels_{}latent_{}window_{}.pt'.format(self.load_effect.get(), self.load_channels.get(), self.load_dims.get(), self.load_window.get(), self.load_number.get()))
      params = torch.load(r'modules\params_{}_{}channels_{}latent_{}window_{}.pt'.format(self.load_effect.get(), self.load_channels.get(), self.load_dims.get(), self.load_window.get(), self.load_number.get()))
      v = torch.load(r'modules\model_{}_{}channels_{}latent_{}window_{}.pt'.format(self.load_effect.get(), self.load_channels.get(), self.load_dims.get(), self.load_window.get(), self.load_number.get()))
    except:
      print("model doesn't exist")
      return
    x = torch.FloatTensor(x)

    L = v._L
    latent_dims = v._latent_dims
    batch_size = self.batch_size
    n = x.shape[1]

    train_ = x[:, :int(0.8 * n)]
    val_ = x[:, int(0.8 * n):int(0.9 * n)]
    test_ = x[:, int(0.9 * n):]
    train_data = DataLoader(slidingWindow(train_, L),
                            batch_size=batch_size,
                            shuffle=False
                            )
    val_data = DataLoader(slidingWindow(val_, L),
                          batch_size=batch_size,
                          shuffle=False
                          )
    test_data = DataLoader(slidingWindow(test_, L),
                           batch_size=batch_size,
                           shuffle=False
                           )
    self.destroy()  # Destroy the existing frame and its contents
    self.__init__(v, train_data, params, self.repetition)
    print("Loaded")

  def reset(self, event):
    self.model.quantizer._embedding.weight = nn.Parameter(self.code_book)

    # resample data with the initial Codebook
    self.x, self.mu, self.logvar, self.z, self.embed, self.x_rec, self.x_rec_mean, _ = self.sample_from_data_VQ(self.model,
                                                                                                             self.data,
                                                                                                             self.repetition)
    # Redraw the plot and heatmap
    self.plot_reconstruction(self.ax_plot, self.x, self.x_rec, self.x_rec_mean)
    self.plot_heatmap(self.ax_heatmap, self.code_book_np)
    self.plot_canvas.draw()
    self.heatmap_canvas.draw()
    print("plots reset")

    # Reset Spinboxes values
    for row in range(self.latent_dims):
      for col in range(self.num_embed):
        value = self.code_book[col][row].detach().numpy()
        idx = row * self.num_embed + col
        self.double_vars[idx].set(value)

    # Reset the list Tracking changes
    self.changed = []

  @torch.no_grad()
  def sample_from_codebook(self, row, col):
    # Get the current codebook
    # code = self.model.quantizer._embedding.weight.detach().numpy()
    # add the new value to the codebook
    # new_code_book = self.set_new_val(code)
    new_val_idx = row * self.num_embed + col
    new_val = torch.tensor(float(self.spinboxs[new_val_idx].get()))
    # new_code_book = code[row, col]
    self.model.quantizer._embedding.weight[row, col] = new_val
    new_code_book = self.model.quantizer._embedding.weight.detach().numpy()
    # init the batch index in the main reconstruction (idx += batchsize in each loop)
    idx = 0

    # Init tensors to store results
    x_rec = torch.empty(self.repetition, self.L, self.n_channels)

    # Loop through data n times
    for i, ((_mu, mu_norm), (_logvar, logvar_norm)) in enumerate(zip(self.mu_loader, self.logvar_loader)):
      # get batch size (last batch may have a different size)
      bs = _mu.shape[0]
      bs = self.L
      print(bs)
      for j in range(self.repetition):
        # generate the batch reconstruction
        _z = self.model.reparametrization_trick(_mu[..., 0], _logvar[..., 0])
        _embed, _ = self.model.quantizer(_z)
        rec, mu_dec, logvar_dec = self.model.decoder(_embed)


        # add the batch reconstruction to the main rec
        # x_rec[j, idx: idx + bs, :] = (rec * self.norm_vec[i])[:, :, 0]
        rec = torch.permute(rec, (0, 2, 1))
        print("shape of x-rec", x_rec[j, idx: idx + bs, :].shape)
        print("the rec to assign", rec[0, :, :].shape)
        print("idx", idx, idx+bs)
        x_rec[j, idx: idx + bs, :] = rec[0, :, :]
      idx += bs

    # Calculate the mean for mu, logvar, z and x_rec
    x_rec_mean = torch.mean(x_rec, dim=0)

    # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    x_rec = torch.permute(x_rec, (1, 0, 2))
    x_rec = x_rec.reshape(self.T, -1)

    # Redray the plots
    self.plot_reconstruction(self.ax_plot, self.x, x_rec, x_rec_mean)
    self.plot_canvas.draw()
    self.plot_heatmap(self.ax_heatmap, new_code_book)
    self.heatmap_canvas.draw()
    print("replotted")

  def set_new_val(self, code):
    # loop through all the spinboxes to check for the changed one
    for i, spin in enumerate(self.spinboxs):
      col = i // self.num_embed
      row = i % self.num_embed
      new_val = np.float32(spin.get())
      new_code_book = code

      # modify the codebook in case of change
      if new_val != code[row, col] and (i, new_val) not in self.changed:
        # if a change is found, then add it to the list for tracking the changes
        self.changed.append((i, new_val))
        self.model.quantizer._embedding.weight[row, col] = torch.tensor(new_val)
        new_code_book = self.model.quantizer._embedding.weight
        print("changed row {}, col {} val {}".format(row, col, new_val))
    return new_code_book

  @torch.no_grad()
  def sample_vae(self, row, col):
      decoder = self.model.decoder
      n_channels = self.args.n_channels
      device = self.args.device
      latent_dims = self.args.latent_dims
      enc_out = self.args.enc_out
      data_shape = self.data.dataset.data.shape
      new_val_idx = row * enc_out + col
      new_val = torch.tensor(float(self.spinboxs[new_val_idx].get()))
      self.latent[col, row] = new_val

      Origin = torch.empty(data_shape)
      REC = torch.empty(data_shape)
      idx = 0
      x_rec, mu_rec, logvar_rec = decoder(self.latent.unsqueeze(0))
      print(self.x_rec.shape)
      print(x_rec.shape)
      self.x_rec[self.n] = x_rec

      self.ax_plot.clear()
      self.plot_reconstruction(self.ax_plot, self.x, self.x_rec, self.n)
      self.plot_canvas.draw()
      heatmap = self.plot_heatmap(self.ax_heatmap, self.latent)
      self.cbar.update_normal(heatmap)
      self.heatmap_canvas.draw()
      print("replotted")
      # for sample_idx, (data_tup, label, norm) in enumerate(self.data):
      #     data = pick_data(data_tup, args)
      #     norm = [n.to(device) for n in norm]
      #     bs = data.shape[0]
      #
      #     x_rec, mu_rec, logvar_rec = decoder(self.latent)
      #
      #     denorm_data = revert_min_max_s(data, norm) if args.min_max else revert_standarization(data, norm)
      #     denorm_rec = revert_min_max_s(x_rec, norm) if args.min_max else revert_standarization(x_rec, norm)
      #
      #     Origin[idx: idx + bs] = denorm_data
      #     REC[idx: idx + bs] = denorm_rec
      #     idx += bs

      # self.x, self.x_rec = Origin, REC
  # @torch.no_grad()
  # def sample_vae(self):
  #   x_len = len(self.data.dataset)
  #
  #   x_rec, mu_dec, logvar_dec, norm_vec = (torch.empty(0, self.n_channels, x_len, device=self.device) for _ in range(4))
  #   mu_enc, logvar_enc = (torch.empty(0, self.n_channels, self.latent_dims, x_len, device=self.device) for _ in range(2))
  #
  #   for j in range(self.repetition):
  #     # create temp tensors to store data in each repetition
  #     x, x_rec_temp, mu_rec_temp, logvar_rec_temp, norm_scale_temp = (torch.empty(self.n_channels, 0, device=self.device) for _ in range(5))
  #     mu_temp, logvar_temp = (torch.empty(self.n_channels, self.latent_dims, 0, device=self.device) for _ in range(2))
  #
  #     for i, (batch, norm_scale) in enumerate(self.data):
  #       batch = batch.to(self.device)
  #       # sample from model
  #       rec, loss, _mu, _logvar, mu_rec, logvar_rec = self.model(batch)
  #
  #       # reshape data -> (Channel, latent_dims, BS) or (Channel, BS)
  #       _mu, _logvar = (t.permute(1, 2, 0) for t in [_mu, _logvar])
  #       batch, rec, norm_scale, mu_rec, logvar_rec = (t.permute(1, 0, 2)[:, :, 0] for t in
  #                                              [batch, rec, norm_scale, mu_rec, logvar_rec])
  #
  #       # Temp store data
  #       mu_temp = torch.cat((mu_temp, _mu), dim=2)
  #       logvar_temp = torch.cat((logvar_temp, _logvar), dim=2)
  #       mu_rec_temp = torch.cat((mu_rec_temp, mu_rec * norm_scale), dim=1)
  #       logvar_rec_temp = torch.cat((logvar_rec_temp, logvar_rec * norm_scale), dim=1)
  #       x_rec_temp = torch.cat((x_rec_temp, rec * norm_scale), dim=1)
  #       norm_scale_temp = torch.cat((norm_scale_temp,  norm_scale), dim=1)
  #       x = torch.cat((x, batch*norm_scale), dim=1)
  #
  #     # Store data after each reconstruction
  #     x_rec = torch.cat((x_rec, x_rec_temp.unsqueeze(0)), dim=0)
  #     mu_dec = torch.cat((mu_dec, mu_rec_temp.unsqueeze(0)), dim=0)
  #     logvar_dec = torch.cat((logvar_dec, logvar_rec_temp.unsqueeze(0)), dim=0)
  #     norm_vec = torch.cat((norm_vec, norm_scale_temp.unsqueeze(0)), dim=0)
  #     mu_enc = torch.cat((mu_enc, mu_temp.unsqueeze(0)), dim=0)
  #     logvar_enc = torch.cat((logvar_enc, logvar_temp.unsqueeze(0)), dim=0)
  #
  #   # join repetition and Channels and get the mean
  #   x_rec, mu_dec, logvar_dec, norm_vec = (t.view(-1, x_len) for t in [x_rec, mu_dec, logvar_dec, norm_vec])
  #   mu_enc, logvar_enc =(t.view(-1, latent_dims, x_len) for t in [mu_enc, logvar_enc])
  #   x_rec_mean, mu_enc_mean, logvar_enc_mean, mu_dec_mean, logvar_dec_mean, norm_vec = (torch.mean(t, dim=0, keepdim=True) for t in
  #                                                                             [x_rec, mu_enc, logvar_enc, mu_dec,
  #                                                                              logvar_dec, norm_vec])
  #   x_rec_mean = x_rec.mean(0)
  #
  #   return x, x_rec, x_rec_mean, mu_enc_mean, logvar_enc_mean, mu_dec_mean, logvar_dec_mean, norm_vec

  @torch.no_grad()
  def sample_vae_latent(self, row, col):
    print(self.mu.shape, self.logvar.shape)
    print(torch.ones_like(self.mu[row, col, :]).shape)
    x_len = len(self.data.dataset)
    new_val_idx = row * self.num_embed + col
    new_val = torch.tensor(float(self.spinboxs[new_val_idx].get()))
    print(new_val)
    increment = new_val - torch.mean(self.mu[row, col, :])
    print(increment)
    self.mu[row, col, :] += torch.ones_like(self.mu[row, col, :]) * increment
    self.mu = torch.FloatTensor(self.mu)

    print(self.mu[row, col, :])

    x_rec, mu_dec, logvar_dec = (torch.empty(0, self.n_channels, x_len, device=self.device) for _ in range(3))

    z_enc = torch.empty(0, self.n_channels, self.latent_dims, x_len, device=self.device)

    mu_loader = DataLoader(slidingWindow(self.mu, self.L),
               batch_size=self.batch_size,
               shuffle=False
               )
    logvar_loader = DataLoader(slidingWindow(self.logvar, self.L),
               batch_size=self.batch_size,
               shuffle=False
               )
    # Loop through data n times
    for j in range(self.repetition):
      # create temp tensors to store data in each repetition
      x_rec_temp, mu_rec_temp, logvar_rec_temp = (torch.empty(self.n_channels, 0, device=self.device) for _ in
                                                     range(3))
      z_temp = torch.empty(self.n_channels, self.latent_dims, 0, device=self.device)
      idx = 0

      for i, ((_mu,norm_mu), (_logvar, norm_logvar)) in enumerate(zip(mu_loader, logvar_loader)):
        bs = _mu.shape[0]
        # print(_mu.shape)
        # print(_logvar.shape)
        # norm_mu = norm_mu.squeeze(-1)
        # norm_logvar = norm_logvar.squeeze(-1)
        _mu = _mu.to(self.device)
        _logvar = _logvar.to(self.device)
        # print("mu", _mu[0, row, col, :])
        z = self.model.reparametrization_trick(_mu * norm_mu, _logvar * norm_logvar)
        # print("z", z[0, row, col, :])
        # sample from model
        rec, mu_rec, logvar_rec = self.model.decoder(z[..., 0])
        # rec, loss, _mu, _logvar, mu_rec, logvar_rec = self.model(batch)

        # reshape data -> (Channel, latent_dims, BS) or (Channel, BS)
        if _mu.shape[0] != self.batch_size:
          rec_last, mu_rec_last, logvar_rec_last = (t.permute(1, 0, 2)[:,-1, :] for t in
                                                           [rec, mu_rec, logvar_rec])
          norm_last = self.norm_vec[:, idx+bs:]
          z_last = z.permute(1, 2, 0, 3)[:, :, -1, :]

          # print("############################Last############################")
          # print("norm, idx", norm_last.shape, idx)
          # print("z after permute", z.shape)
          # print("rec ", rec_last.shape)
          # print("mu_rec ", mu_rec_last.shape)
          # print("logvar_rec ", logvar_rec_last.shape)

        # print("z before permute", z.shape)
        z = z.permute(1, 2, 0, 3)[..., 0]
        # print("############################Before############################")
        # print("norm", self.norm_vec.shape)
        #
        # print("z after permute", z.shape)
        # print("rec ", rec.shape)
        # print("mu_rec ", mu_rec.shape)
        # print("logvar_rec ", logvar_rec.shape)

        rec,  mu_rec, logvar_rec = (t.permute(1, 0, 2)[:, :, 0] for t in
                                                    [rec, mu_rec, logvar_rec])
        norm = self.norm_vec[:, idx]


        # print("############################After############################")
        # print("norm, idx", norm.shape, idx)
        # print("z after permute", z.shape)
        # print("rec ", rec.shape)
        # print("mu_rec ", mu_rec.shape)
        # print("logvar_rec ", logvar_rec.shape)

        # Temp store data
        z_temp = torch.cat((z_temp, z), dim=2)
        mu_rec_temp = torch.cat((mu_rec_temp, mu_rec * norm), dim=1)
        logvar_rec_temp = torch.cat((logvar_rec_temp, logvar_rec * norm), dim=1)
        x_rec_temp = torch.cat((x_rec_temp, rec * norm), dim=1)

        if _mu.shape[0] != self.batch_size:
          z_temp = torch.cat((z_temp, z_last), dim=2)
          mu_rec_temp = torch.cat((mu_rec_temp, mu_rec_last * norm_last), dim=1)
          logvar_rec_temp = torch.cat((logvar_rec_temp, logvar_rec_last * norm_last), dim=1)
          x_rec_temp = torch.cat((x_rec_temp, rec_last * norm_last), dim=1)

        idx += bs


      # Store data after each reconstruction
      x_rec = torch.cat((x_rec, x_rec_temp.unsqueeze(0)), dim=0)
      mu_dec = torch.cat((mu_dec, mu_rec_temp.unsqueeze(0)), dim=0)
      logvar_dec = torch.cat((logvar_dec, logvar_rec_temp.unsqueeze(0)), dim=0)
      z_enc = torch.cat((z_enc, z_temp.unsqueeze(0)), dim=0)
      # logvar_enc = torch.cat((logvar_enc, logvar_temp.unsqueeze(0)), dim=0)

    # join repetition and Channels and get the mean
    x_rec, mu_dec, logvar_dec = (t.view(-1, x_len) for t in [x_rec, mu_dec, logvar_dec])
    z_enc = z_enc.view(-1, latent_dims, x_len)
    x_rec_mean, z_enc_mean, mu_dec_mean, logvar_dec_mean = (torch.mean(t, dim=0, keepdim=True) for t
                                                                              in
                                                                              [x_rec, z_enc, mu_dec,
                                                                               logvar_dec])
    x_rec_mean = x_rec.mean(0)
    avg_latents = torch.mean(z_enc_mean, dim=2)
    print(avg_latents)

    # Redray the plots
    print(type(self.ax_plot))
    self.ax_plot.clear()
    self.plot_reconstruction(self.ax_plot, self.x, x_rec, x_rec_mean)
    self.plot_canvas.draw()
    heatmap = self.plot_heatmap(self.ax_heatmap, avg_latents)
    self.cbar.update_normal(heatmap)
    self.heatmap_canvas.draw()
    print("replotted")
    print(x_rec_mean)

    # return x, x_rec, x_rec_mean, mu_dec_mean, logvar_dec_mean


  @torch.no_grad()
  def sample_from_data_VQ(self, model, data, n):
    # Init tensors to store results
    model.eval()
    x = torch.empty((self.L, self.n_channels))
    mu, logvar, z, embed = (torch.empty((self.repetition, self.L, self.num_embed, self.latent_dims)) for _ in range(4))
    x_rec = torch.empty(self.repetition, self.L, self.n_channels)
    print(x_rec.shape)

    # Init the normalisation vector and batch index
    norm_vec = []
    idx = 0

    # Loop through data n times
    for i, (batch, v) in enumerate(data):
      # get batch size
      bs = batch.shape[0]
      # bs = self.L
      for j in range(n):
        # generate reconstruction and latent space over the x axis
        rec, loss, _mu, _logvar, mu_rec, logvar_rec = model(batch)
        # _z = model.reparametrization_trick(_mu, _logvar)
        # _embed, _ = model.quantizer(_z)
        # print(v.shape)
        # normalization
        if v.dim() == 1:
          v = v.unsqueeze(-1)
          v = v.unsqueeze(-1)
        # print(v.shape)
        rec = torch.permute(rec, (0, 2, 1))

        print("rec", rec.shape)
        print(bs)


        # Fill the Tensors with data Shape (mu, logvar,z): n*T*K*D, with K num of embeddings and D latent dims
        # x_rec = n*T*C
        mu[     j, idx: idx + bs, :] = _mu
        logvar[ j, idx: idx + bs, :] = _logvar
        # z[      j, idx: idx + bs, :] = _z
        # embed[  j, idx: idx + bs, :] = _embed
        x_rec[  j, idx: idx + bs, :] = (rec )[:, :, :]
      batch = torch.permute(batch, (0, 2, 1))
      x[           idx: idx + bs, :] = (batch )[:, :, :]  # Shape T*C

      idx += bs
      # store the normalisation for each batch
      if self.norm_vec_filled == False:
        norm_vec.append(v)
    print("rec1", x_rec.shape)
    # Calculate the mean for mu, logvar, z and x_rec
    mu, logvar, z, embed, x_rec_mean = (torch.mean(t, dim=0) for t in [mu, logvar, z, embed, x_rec])
    print("rec2", x_rec_mean.shape)
    # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    x_rec = torch.permute(x_rec, (1, 0, 2))
    print("rec3", x_rec.shape)
    x_rec = x_rec.reshape(self.L, -1)
    print("rec4", x_rec.shape)

    # convert to numpy, print shapes and output
    x, z, x_rec = (t.detach().numpy() for t in [x, z, x_rec])
    print("Tensors x: {}, mu: {}, logvar: {}, z: {}, x_rec: {}".format(x.shape, mu.shape, logvar.shape, z.shape,
                                                                       x_rec.shape))
    self.norm_vec_filled = True

    return x, mu, logvar, z, embed, x_rec, x_rec_mean, norm_vec

  @torch.no_grad()
  def calculate_MI_score(self, var):
    MI_scores = torch.empty((self.num_embed, self.latent_dims))
    MI_scores_vec = torch.empty((self.latent_dims))
    print(self.num_embed)
    print(self.latent_dims)
    print(self.embed.shape)

    for col in range(self.num_embed):
      for row in range(self.latent_dims):
        # for channel in range(self.n_channels):
          # Calculate mutual Info
        embed_vec = self.embed[:, row, col].numpy()
        mutual_info = mutual_info_score(var, embed_vec, contingency=None)

        # fill the MI_scores Tensor
        MI_scores[row, col] = torch.tensor(mutual_info)

    # print(self.embed.shape)
    # print(self.latent_dims)
    # for row in range(self.num_embed):
    #     # Calculate mutual Info
    #     embed_vec = self.embed[:, row, :].view(embed_vec.shape[0], -1).numpy()
    #     new_var = np.expand_dims(var, 1)
    #     print(embed_vec.shape)
    #     print(new_var.shape)
    #     mutual_info = mutual_info_score(embed_vec, var, contingency=None)
    #
    #     # fill the MI_scores Tensor
    #     MI_scores_vec[col] = torch.tensor(mutual_info)

    # print(MI_scores)

    # print(var.shape)
    # reshaped_embed = self.embed.view(var.shape[0], -1).T.detach().numpy()
    # print("reshaped Embedding", reshaped_embed.shape)
    #
    # # Broadcast the input var to match the shape of the embedding
    # broadcast_shape = (reshaped_embed.shape[0], reshaped_embed.shape[1])
    # var_broadcasted = np.broadcast_to(var, broadcast_shape)
    # print("var_broadcasted", var_broadcasted.shape)
    # mutual_info = mutual_info_score(var, reshaped_embed, contingency=None)

    return MI_scores



if __name__ == "__main__":
  i = 2
  p = 2
  effect = "both"
  args = GENV(n_channels=1, latent_dims=5, n_samples=100, shuffle=False, periode=p, L=288 * p, min_max=True,
              num_layers=3, robust=False, first_kernel=288, num_embed=512, modified=False)
  n_channels = args.n_channels
  latent_dims = args.latent_dims
  L = args.L
  batch_size = args.bs
  effect = "Seasonality" # trend, seasonality, std_variation, trend_seasonality, no_effect


  # x = torch.load(r'modules\data_{}channels_{}latent_{}window.pt'.format(n_channels, latent_dims, L))
  # params = torch.load(r'modules\params_{}channels_{}latent_{}window.pt'.format(n_channels, latent_dims, L))
  # v = torch.load(r'modules\vq_ema_{}channels_{}latent_{}window.pt'.format(n_channels, latent_dims, L))

  x = torch.load(r'modules\vq_vae_data_{}_{}channels_{}latent_{}window_{}.pt'.format(effect, n_channels,latent_dims, L, i))
  params = torch.load(r'modules\vq_vae_params_{}_{}channels_{}latent_{}window_{}.pt'.format(effect, n_channels,latent_dims, L, i))
  e_params = torch.load(r'modules\vq_vae_e_params_{}_{}channels_{}latent_{}window_{}.pt'.format(effect, n_channels, latent_dims, L, i))
  vae = torch.load(r'modules\vq_vae_vae_{}_{}channels_{}latent_{}window_{}.pt'.format(effect, n_channels,latent_dims, L, i))


  print(x.shape)
  # print(params)
  # print(vae)
  x = torch.FloatTensor(x)

  # L = v._L
  # latent_dims = v._latent_dims
  batch_size = 22
  n = x.shape[1]

  effects = {
    "Pulse": {
      "occurances": 1,
      "max_amplitude": 5,
      "interval": 40,
      "start": None
    },
    "Trend": {
      "occurances": 10,
      "max_slope": 0.002,
      "type": "linear",
      "start": None
    },
    "Seasonality": {
      "occurances": 10,
      "frequency_per_week": (14, 21),  # min and max occurances per week
      "amplitude_range": (5, 20),
      "start": -5
    },
    "std_variation": {
      "occurances": 0,
      "max_value": 10,
      "interval": 30,
      "start": None
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
  effects = set_effect(effect, effects, 2)
  labels = extract_parameters(args, e_params=e_params, effects=effects)
  # labels = add_mu_std(labels, params)

  train_data, val_data, test_data = create_loader_noWindow(x, args, labels, norm=True)


  app = VQ_gui(vae, train_data, args, repetition=3, VQ=False)

  app.mainloop()
