import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
# import torch
np.random.seed(10)

class Gen():
    def __init__(self, periode = 30, step = 5, val = 1000, nchannels = 3, effects = None):              
        # get the parameters
        self.step = step
        self.effects = effects
        self.nchannels = nchannels
        self.val = val
        
        # generate the time axis
        min_per_day = 1440
        self.periode = periode * min_per_day # convert in minutes
        self.t = np.arange(self.periode, step = self.step)        
        self.n = self.t.shape[0]
        
        # generate y values
        self.mu = np.random.randint(self.val, size=nchannels)
        self.mu = np.tile(self.mu, (self.n,1)).T.astype(np.float32)
        self.cov = np.diag(np.ones(self.n))
#         self.cov = np.diag(np.random.uniform(high=1, size=nchannels))
        self.effects_params = {
            "Pulse":{
                "channel":[],
                "index":[],
                "scale":[]
            },
            "Trend":{
                "channel":[],
                "index":[],
                "slope":[]
            },
            "Seasonality":{
                "channel":[],
                "frequency":[],
                "amplitude":[],
                "phaseshift":[]
            }            
        } 
        
        # add effects (noise)
        self.add_effects(self.effects) 
        
        # generate the different timeseries: multivariate normal dist
        self. x = np.array([np.random.multivariate_normal(self.mu[:,obs], np.diag(np.ones_like(self.mu[:,obs]))) for obs in range(self.n)]).T            
    
    #plots the generated data
    def show(self): 
        self.plot_time_series(self.t, self.x.T, "Generated MTS")
#         plt.plot(self.t, self.x.T)
#         plt.grid(True)
        
    #returns the Time series and their parameters
    def parameters(self):        
        params = {
            "T": len(self.t),
            "nchannels":len(self.mu),
#             "effects":self.effects,
            "mu":self.mu,
            "cov":self.cov
        }
        return self.x, params, self.effects_params
    
    # loops through all the input effects and calls the respective function for each effect
    def add_effects(self, effects):        
        if self.effects is not None:
            for effect, params in self.effects.items(): 
                if params["number"] == 0:
                    continue
                if effect == "Pulse":
                    self.add_pulse(params)                
                elif effect == "Trend":
                    self.add_trend(params)
                elif effect == "Seasonality":
                    self.add_seasonality(params)
    
    # adds a pulse effect
    def add_pulse(self, params):
        # extract and generate the parameters: 
        # extract:
        n = params["number"]
        gain = params["max_gain"]
        
        ### create randomised Pulses parameters
        # channel: On which channel will the effect be applied.
        # idxs: At which index will the effect be applied.
        # scale: How strong is the Pulse. 
        channels = np.random.randint(self.nchannels, size=n)
        values = np.random.randint(self.t[-1], size=n)
        idxs = (values*self.step**-1).astype(int)
#         idxs = np.array([np.where(self.t == value) for value in values]).squeeze()
        scale = np.random.uniform(low = -gain, high = gain, size=n)
        
        #save the Pulses parameters
        self.effects_params["Pulse"]["channel"].extend(channels)
        self.effects_params["Pulse"]["index"].extend(values)
        self.effects_params["Pulse"]["scale"].extend(scale)
        
        # generate the pulses
        ground_val = self.mu[channels, idxs]
        k = np.random.uniform(ground_val, ground_val*scale)

        # add it to the channels
        self.mu[channels,idxs] += k
        
    
    def add_trend(self, params):
        # extract and generate the parameters:
        # extract:
        n = params["number"]   
        slope = params["max_slope"]      
        
        ### create randomised Trends parameters
        # channels: On which channel will the effect be applied.
        # idxs: At which index will the effect be applied.
        # slope: Slopses of the different trends. 
        channels = np.random.randint(self.nchannels, size=n)
        channels_nodup = list(dict.fromkeys(channels))
        values = np.random.randint(self.t[-1], size=n)
        idxs = (values*self.step**-1).astype(int)        
        slopes = np.random.uniform(low = -slope, high = slope, size=n)  
        
        #save the Trends parameters
        self.effects_params["Trend"]["channel"].extend(channels)
        self.effects_params["Trend"]["index"].extend(values)
        self.effects_params["Trend"]["slope"].extend(slopes)
        
        # generate the trends
        trends = np.zeros_like(self.mu)
        for channel, idx in enumerate(idxs):
            shifted = len(self.t) - idx
            ch = channels[channel]

            trends[ch, idx :] += np.linspace(0,  slopes[channel]* self.step*shifted, shifted)

        # add it to the channels
        self.mu += trends
        
    def add_seasonality(self,params):
        # extract and generate the parameters:
        # extract:
        n = params["number"]
        freq = params["frequency_range"]
        amp = params["amplitude_range"]
#         w = 1/self.t[-1]
        
        ### create randomised Seasonalities parameters
        # channels: On which channel will the effect be applied.
        # freqs: Frequency of each seasonality.
        # amps: Amplitude of each seasonality. 
        # phases: Phase of each seasonality
        channels = np.random.randint(self.nchannels, size=n)
        freqs = np.random.uniform(low = freq[0], high = freq[1], size=n)
        amps = np.random.randint(low = amp[0], high = amp[1], size=n)
        phases = np.random.randint(180, size=n)

        #save the Trends parameters
        self.effects_params["Seasonality"]["channel"].extend(channels)
        self.effects_params["Seasonality"]["frequency"].extend(freqs)
        self.effects_params["Seasonality"]["amplitude"].extend(amps)
        self.effects_params["Seasonality"]["phaseshift"].extend(phases)
        
        # generate the seasonalites
        g2r =  np.pi/180
        seas = np.zeros_like(self.mu)
        for idx, channel in enumerate(channels):
            seas[channel] = np.maximum(seas[channel], np.sin(2 * np.pi * self.t * freqs[idx] / (24 * 60 * 7) + phases[idx])* amps[idx])
        
         # add it to the channels
        self.mu += seas
        
    def plot_time_series(self, x, y, title):
        """
        Plots a time series with dates on the x-axis using matplotlib.

        Args:
        x (list): List of datetime objects representing the x-axis.
        y (list): List of values representing the y-axis.
        title (str): The title of the plot.

        Returns:
        None
        """        
        reference_time = np.datetime64('2023-03-01T00:00:00')
        date_array = reference_time + np.array(x, dtype='timedelta64[m]')
    
        duration = np.max(date_array) - np.min(date_array)
        one_month = np.timedelta64(1, 'M').astype('timedelta64[m]')
        
        formater = mdates.DateFormatter('%A')
        locator = mdates.DayLocator()
        if duration > one_month/2 and duration <= 2 * one_month:
            locator = mdates.DayLocator(interval=2)
        elif duration > 2 * one_month:
            locator = mdates.WeekdayLocator()
            formater = mdates.DateFormatter('%Y-%m-%d')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(date_array, y)
        ax.set(xlabel='Date', ylabel='Value', title=title)
        ax.xaxis.set_major_formatter(formater)
        ax.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()