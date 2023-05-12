import torch
import numpy as np
from torch.utils.data import DataLoader
from tkinter.messagebox import showinfo
import tkinter as tk
from tkinter import ttk
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from VQ_EMA_fn import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Cursor
from copy import deepcopy

class VQ_gui(tk.Tk):
  @torch.no_grad()
  def __init__(self, model, data, repetition):
    super().__init__()
    # Init variable vor the Gui
    self.model, self.data, self.repetition = model, data, repetition
    self.model.eval()
    self.spinboxs, self.double_vars = [], []
    self.norm_vec_filled = False

    # Get the model parameters
    self.L              = self.model._L
    self.T              = self.data.dataset.data.shape[1]- self.L     # last window is removed because for each window we sample only 1 mu/logvar/z etc
    self.batch_size     = self.data.batch_size
    # self.n_channels     = self.model._n_channels
    self.num_embed       = self.model._num_embed
    self.latent_dims    = self.model._latent_dims
    self.code_book      = deepcopy(self.model.quantizer._embedding.weight.data)
    self.code_book_np   = self.code_book.detach().numpy()
    self.changed = []

    # Sample x, x_rec and the latent space
    self.x, self.mu, self.logvar, self.z, self.embed, self.x_rec, self.x_rec_mean, self.norm_vec = self.sample_from_data_VQ(model, data, repetition)

    # create a DataLoader from mu and logvar for later generation
    self.mu_loader = DataLoader(self.mu, #slidingWindow(self.mu, self.L),
                           batch_size=self.batch_size,
                           shuffle=False
                           )
    self.logvar_loader = DataLoader(self.logvar, #slidingWindow(self.logvar, self.L),
                               batch_size=self.batch_size,
                               shuffle=False
                               )

    # create window and frames layout
    button_frame, grid_frame, heatmap_frame, plot_frame = self.create_frames()

    # Create the text input widgets
    self.create_spinboxes(grid_frame)

    # create buttons for Reset and Save
    self.create_buttons(button_frame)

    # create plot
    self.ax_plot, self.rec_lines, self.rec_lines_mean, self.plot_canvas = self.create_plot(plot_frame)

    # create heatmap
    self.ax_heatmap, self.heatmap, self.heatmap_canvas = self.create_heatmap(heatmap_frame)


  def create_heatmap(self, heatmap_frame):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax_heatmap = fig.add_subplot(111)
    heatmap = self.plot_heatmap(ax_heatmap, self.code_book)
    heatmap_canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
    heatmap_canvas.draw()
    heatmap_canvas.get_tk_widget().pack(fill="both", expand=True)

    return ax_heatmap, heatmap, heatmap_canvas

  def create_plot(self, plot_frame):
    fig = Figure(figsize=(16, 4), dpi=100)
    ax_plot = fig.add_subplot(111)
    rec_lines, rec_lines_mean = self.plot_reconstruction(ax_plot, self.x, self.x_rec, self.x_rec_mean)
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

    return ax_plot, rec_lines, rec_lines_mean, plot_canvas


  def create_buttons(self, button_frame):
    # Reset Button
    self.reset_button = tk.Button(button_frame, text="Reset Values")
    self.reset_button.pack(side="top", padx=20, pady=15, fill="x")
    self.reset_button.bind("<Button-1>", self.reset)

    # Save Button
    self.save_button = tk.Button(button_frame, text="Save Model")
    self.save_button.pack(side="top", padx=20, pady=15, fill="x")
    self.save_button.bind("<Button-1>", self.save)

    # File name for saving
    self.file_name = tk.Entry(button_frame)
    self.file_name.pack(side="top", padx=20, pady=30, fill="x")

  def create_spinboxes(self, grid_frame):
    for row in range(self.latent_dims):
      for col in range(self.num_embed):
        self.double_var = tk.DoubleVar()
        # create the Spinbox and link it to the DoubleVar
        self.spinbox = Spinbox(grid_frame, from_=-50.0, to=50, textvariable=self.double_var,
                               command=lambda: self.sample_from_codebook(row, col))
        self.spinbox.grid(row=row, column=col, sticky="NSEW")

        # set initial value
        initial_value = self.code_book_np[row][col]
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

    # create buttons frame
    button_frame = tk.Frame(parent_frame)
    button_frame.grid(row=0, column=1, sticky="nsew")

    # create bottom frame for inputs and heatmap
    bottom_frame = tk.Frame(self)
    bottom_frame.pack(side="bottom", fill="both", expand=True)

    # create left and right subframes for inputs and heatmap
    inputs_frame = tk.Frame(bottom_frame)
    inputs_frame.pack(side="left", fill="both", expand=True)
    heatmap_frame = tk.Frame(bottom_frame)
    heatmap_frame.pack(side="right", fill="both", expand=True)

    # create grid of input labels and entries
    grid_frame = tk.Frame(bottom_frame, padx=10, pady=10)
    grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    return button_frame, grid_frame, heatmap_frame, plot_frame

  def plot_heatmap(self, ax_heatmap, codebook):
    ax_heatmap.clear()
    heatmap = ax_heatmap.imshow(codebook)
    ax_heatmap.set_title('Codebook Heatmap')

    return heatmap

  def plot_reconstruction(self, ax_plot, x, x_rec, x_rec_mean):
    ax_plot.clear()
    ax_plot.plot(x, "b")
    rec_lines = ax_plot.plot(x_rec, "orange", alpha=0.2)
    rec_lines_mean = ax_plot.plot(x_rec_mean, "r")
    ax_plot.set_title('Reconstruction')
    ax_plot.grid()

    return rec_lines, rec_lines_mean


  def save(self, event):
    torch.save(self.model, r'modules\{}.pt'.format(self.file_name.get()))
    print("Saved")

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
        value = self.code_book[row][col]
        idx = row * self.num_embed + col
        self.double_vars[idx].set(value)

    # Reset the list Tracking changes
    self.changed = []

  @torch.no_grad()
  def sample_from_codebook(self, row, col):
    # Get the current codebook
    code = self.model.quantizer._embedding.weight.detach().numpy()
    # add the new value to the codebook
    new_code_book = self.set_new_val(code)
    # init the batch index in the main reconstruction (idx += batchsize in each loop)
    idx = 0

    # Init tensors to store results
    x_rec = torch.empty(self.repetition, self.T, self.num_embed)

    # Loop through data n times
    for i, (_mu, _logvar) in enumerate(zip(self.mu_loader, self.logvar_loader)):
      # get batch size (last batch may have a different size)
      bs = _mu.shape[0]
      for j in range(self.repetition):
        # generate the batch reconstruction
        _z = self.model.reparametrization_trick(_mu, _logvar)
        _embed, _ = self.model.quantizer(_z)
        rec = self.model.decoder(_embed)

        # add the batch reconstruction to the main rec
        x_rec[j, idx: idx + bs, :] = (rec * self.norm_vec[i])[:, :, 0]
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
      row = i // self.num_embed
      col = i % self.num_embed
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
  def sample_from_data_VQ(self, model, data, n):
    # Init tensors to store results
    x = torch.empty((self.T, self.num_embed))
    mu, logvar, z, embed = (torch.empty((self.repetition, self.T, self.num_embed , self.latent_dims)) for _ in range(4))
    x_rec = torch.empty(self.repetition, self.T, self.num_embed)

    # Init the normalisation vector and batch index
    norm_vec = []
    idx = 0

    # Loop through data n times
    for i, (batch, v) in enumerate(data):
      # get batch size
      bs = batch.shape[0]
      for j in range(n):
        # generate reconstruction and latent space over the x axis
        rec, loss, _mu, _logvar = model(batch)
        _z = model.reparametrization_trick(_mu, _logvar)
        _embed, _ = model.quantizer(_z)

        # normalization
        if v.dim() == 1:
          v = v.unsqueeze(-1)
          v = v.unsqueeze(-1)

        # Fill the Tensors with data Shape (mu, logvar,z): n*T*K*D, with K num of embeddings and D latent dims
        # x_rec = n*T*C
        mu[     j, idx: idx + bs, :] = _mu
        logvar[ j, idx: idx + bs, :] = _logvar
        z[      j, idx: idx + bs, :] = _z
        embed[  j, idx: idx + bs, :] = _embed
        x_rec[  j, idx: idx + bs, :] = (rec * v)[:, :, 0]
      x[           idx: idx + bs, :] = (batch * v)[:, :, 0]  # Shape T*C

      idx += bs
      # store the normalisation for each batch
      if self.norm_vec_filled == False:
        norm_vec.append(v)

    # Calculate the mean for mu, logvar, z and x_rec
    mu, logvar, z, embed, x_rec_mean = (torch.mean(t, dim=0) for t in [mu, logvar, z, embed, x_rec])

    # reshape and squeeze x_rec so that n and C are merged and final shape is T * (C*n)
    x_rec = torch.permute(x_rec, (1, 0, 2))
    x_rec = x_rec.reshape(self.T, -1)

    # convert to numpy, print shapes and output
    x, z, x_rec = (t.detach().numpy() for t in [x, z, x_rec])
    print("Tensors x: {}, mu: {}, logvar: {}, z: {}, x_rec: {}".format(x.shape, mu.shape, logvar.shape, z.shape,
                                                                       x_rec.shape))
    self.norm_vec_filled = True

    return x, mu, logvar, z, embed, x_rec, x_rec_mean, norm_vec

if __name__ == "__main__":
  #r'modules\data_{}channels_{}latent.pt'
  x = torch.load(r'modules\data_1channels_6latent.pt')
  x = torch.FloatTensor(x)

  v = torch.load(r'modules\vq_ema_1channels_6latent.pt')
  print(v)
  L = v._L
  latent_dims = v._latent_dims
  batch_size = 22
  n = x.shape[1]

  train_ = x[:, :int(0.8 * n)]
  val_ = x[:, int(0.8 * n):int(0.9 * n)]
  test_ = x[:, int(0.9 * n):]
  print(train_.shape)
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

  app = VQ_gui(v, train_data, 10)

  app.mainloop()
