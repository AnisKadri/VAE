import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates


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
        self.t = np.arange(self.periode, step = self.step) # time axis
        self.n = self.t.shape[0] # number of points in time axis
        self.reference_time = np.datetime64('2023-03-01T00:00:00') # Reference time (for plausibility and plots)
        
        
        # generate y values
        self.mu = np.random.randint(self.val, size=self.nchannels) # mean values for each channel
        self.mu = np.tile(self.mu, (self.n,1)).T.astype(np.float32) # expand the means over the time axis

        self.cov = np.diag(np.ones(self.nchannels)) # diag cov matrix for each channel
        self.cov = np.tile(self.cov, (self.n,1,1)).T.astype(np.float32) # expand the covs over the time axis

        
        self.effects_params = {
            "Pulse":{
                "channel":[],
                "index":[],
                "amplitude":[]
            },
            "Trend":{
                "channel":[],
                "index":[],
                "slope":[]
            },
            "Seasonality":{
                "channel":[],
                "frequency_per_week":[],
                "amplitude":[],
                "phaseshift":[]
            },  
            "Std_variation":{
                "channel":[],
                "interval":[],
                "amplitude":[]
            },
            "Channels_Coupling":{
                "channels":[],
                "amplitude":[]
            },
            "Noise":{
                "channel":[],
                "index":[],
                "slope":[]
            }
        } 
        
        # add effects (noise)
        self.add_effects(self.effects) 
        
        # generate the different timeseries: multivariate normal dist
        self.x = np.array([np.random.multivariate_normal(self.mu[:,obs], self.cov[:,:,obs]) for obs in range(self.n)]).T    
        
        self.add_noise()
    
    
    #plots the generated data
    def show(self): 
        self.plot_time_series("Generated MTS")
        
        
    #returns the Time series and their parameters
    def parameters(self): 
        
        params = {
            "n": self.n,
            "nchannels":len(self.mu),

            "mu":self.mu,
            "cov":self.cov
        }
        return self.x, params, self.effects_params
    
    
    # loops through all the input effects and calls the respective function for each effect
    def add_effects(self, effects): 
        
        if self.effects is not None:
            for effect, params in self.effects.items(): 
                if params["occurances"] == 0:
                    continue
                if effect == "Pulse":
                    self.add_pulse(params)                
                elif effect == "Trend":
                    self.add_trend(params)
                elif effect == "Seasonality":
                    self.add_seasonality(params)
                elif effect == "std_variation":
                    self.add_std_variation(params)
                elif effect == "channels_coupling":
                    self.add_channels_coupling(params)
    
    
    # adds a pulse effect
    def add_pulse(self, params):
        
        # extract parameters: 
        occ = params["occurances"] # number of Pulses.
        amp = params["max_amplitude"] # max amplitude of the pulse
        interval = params["interval"] # length of interval on which pulse will be applied
        
        
        ### create randomised Pulses parameters       
        channels = np.random.randint(self.nchannels, size=occ) # On which channel will the effect be applied.        
        start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the Pulse start.
        end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the Pulse end.
        amplitude = np.random.uniform(low = -amp, high = amp, size=occ) # How strong is the Pulse.
        
        
        #save the Pulses parameters
        idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
        self.effects_params["Pulse"]["channel"].extend(channels)
        self.effects_params["Pulse"]["index"].extend(idxs_to_time.astype('str'))
        self.effects_params["Pulse"]["amplitude"].extend(amplitude)
        
        
        # generate the pulses
        ground_val = self.mu[channels, start_idxs] # original value at the pulse indexes
        k = np.random.uniform(ground_val, ground_val*amplitude) # new values 

        # add it to the channels
        for i in range(occ):
            self.mu[channels[i],start_idxs[i]: end_idxs[i]] += k[i]
        
    
    def add_trend(self, params):
        
        # extract parameters:
        occ = params["occurances"]  # number of Trends.   
        slope = params["max_slope"] # Max slope of the Trends  
        trend_type = params["type"] # linear or quadratic or mixed trends
        
        
        ### create randomised Trends parameters
        channels = np.random.randint(self.nchannels, size=occ) # On which channel will the Trend be applied.
        idxs = np.random.randint(self.n, size=occ) # At which index will the Trend start.        
        slopes = np.random.uniform(low = -slope, high = slope, size=occ)  # Slope of the Trend.
        
        
        #save the Trends parameters
        idxs_to_time = self.reference_time + (idxs * self.step).astype('timedelta64[m]')
        self.effects_params["Trend"]["channel"].extend(channels)
        self.effects_params["Trend"]["index"].extend(idxs_to_time.astype('str'))
        self.effects_params["Trend"]["slope"].extend(slopes)
        
        
        # generate the trends
        trends = np.zeros_like(self.mu)
        for channel, idx in enumerate(idxs):
            shifted = len(self.t) - idx
            ch = channels[channel]
            if trend_type == "linear":
                trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)
            elif trend_type == "quadratic":
                trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**2
            elif trend_type == "mixed":
                trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**((channel%2)+1)

        # add it to the channels
        self.mu += trends
        
    def add_seasonality(self,params):
        
        # extract parameters:
        occ = params["occurances"] # number of Seasonalities.
        freq = params["frequency_per_week"] # frequency per Week
        amp = params["amplitude_range"] # max amplitudes per week

        
        ### create randomised Seasonalities parameters
        channels = np.random.randint(self.nchannels, size=occ) # On which channel will the seasonality be applied.
        freqs = np.random.uniform(low = freq[0], high = freq[1], size=occ) # frequency per week
        amps = np.random.randint(low = amp[0], high = amp[1], size=occ) # max amplitude 
        phases = np.random.randint(180, size=occ) # shift to be applied
        

        #save the Trends parameters
        self.effects_params["Seasonality"]["channel"].extend(channels)
        self.effects_params["Seasonality"]["frequency_per_week"].extend(freqs)
        self.effects_params["Seasonality"]["amplitude"].extend(amps)
        self.effects_params["Seasonality"]["phaseshift"].extend(phases)
        
        
        # generate the seasonalites
        seas = np.zeros_like(self.mu)
        for idx, channel in enumerate(channels):
            seas[channel] = np.maximum(seas[channel], np.sin(2 * np.pi * self.t * freqs[idx] / (24 * 60 * 7) + phases[idx])* amps[idx])
        
         # add it to the channels
        self.mu += seas
        
    def add_std_variation(self, params):
        
        # extract parameters:
        occ = params["occurances"] # number of std variations.
        max_value = params["max_value"] # max values of std.
        interval = params["interval"] # length of the interval on which the std variates
        
        
        ### create randomised std variations parameters
        channels = np.random.randint(self.nchannels, size=occ) #  # On which channel will the seasonality be applied.     
        start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the std variation start.
        end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the std variation end.
        intervals = end_idxs - start_idxs
        
        
        #save std variations parameters
        start_idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
        end_idxs_to_time = self.reference_time + (end_idxs * self.step).astype('timedelta64[m]')
        durations = (end_idxs_to_time- start_idxs_to_time).astype('timedelta64[D]')
        
        self.effects_params["Std_variation"]["channel"].extend(channels)
        self.effects_params["Std_variation"]["interval"].extend(durations.astype('str'))        
        
        
        # add it to the channels
        for i in range(occ):
            ch = channels[i]
            amplitude = np.random.uniform(high = max_value, size = (1, intervals[i]))
            
            self.cov[ch, ch, start_idxs[i]: end_idxs[i]] = amplitude
            self.effects_params["Std_variation"]["amplitude"].extend(amplitude)
            
    def add_channels_coupling(self, params):
        
        # extract parameters:
        occ = params["occurances"] # number of channels_coupling.
        max_value = params["coupling_strengh"] # max values of std.
#         interval = params["interval"] # length of the interval on which the std variates
        
        
        ### create randomised std variations parameters
        channels = np.random.randint(self.nchannels, size=(occ, 2)) #  # On which channel will the seasonality be applied.     
#         start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the std variation start.
#         end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the std variation end.
#         intervals = end_idxs - start_idxs
        
        
        #save std variations parameters
#         start_idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
#         end_idxs_to_time = self.reference_time + (end_idxs * self.step).astype('timedelta64[m]')
#         durations = (end_idxs_to_time- start_idxs_to_time).astype('timedelta64[D]')
        
        self.effects_params["Channels_Coupling"]["channels"].extend(channels)
#         self.effects_params["Std_variation"]["interval"].extend(durations.astype('str'))        
        
        
        # add it to the channels
        for i in range(occ):
            ch = channels[i]
            amplitude = np.random.uniform(high = max_value)
            
            self.cov[ch[0], ch[1], :] = amplitude
            self.effects_params["Channels_Coupling"]["amplitude"].append(amplitude)
            
    
    def add_noise(self):
        
        if self.effects is not None:
            params = self.effects["Noise"]
        
            if params["occurances"] != 0:
                
                # extract parameters:
                occ = params["occurances"]  # number of Trends.   
                slope = params["max_slope"] # Max slope of the Trends  
                noise_type = params["type"] # linear or quadratic or mixed trends


                ### create randomised Noise parameters
                channels = np.random.randint(self.nchannels, size=occ) # On which channel will the Trend be applied.
                idxs = np.random.randint(self.n, size=occ) # At which index will the Trend start.        
                slopes = np.random.uniform(low = -slope, high = slope, size=occ)  # Slope of the Trend.


                #save the Noise parameters
                idxs_to_time = self.reference_time + (idxs * self.step).astype('timedelta64[m]')
                self.effects_params["Noise"]["channel"].extend(channels)
                self.effects_params["Noise"]["index"].extend(idxs_to_time.astype('str'))
                self.effects_params["Noise"]["slope"].extend(slopes)


                # generate the trends
                noises = np.zeros_like(self.x)
                for channel, idx in enumerate(idxs):
                    shifted = len(self.t) - idx
                    ch = channels[channel]
                    if noise_type == "linear":
                        noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)
                    elif noise_type == "quadratic":
                        noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**2
                    elif noise_type == "mixed":
                        noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**((channel%2)+1)

                # add it to the channels
                self.x += noises
        
        
    def plot_time_series(self, title): 
        
        date_array = self.reference_time + np.array(self.t, dtype='timedelta64[m]')
    
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
        ax.plot(date_array, self.x.T)
        ax.set(xlabel='Date', ylabel='Value', title=title)
        
        ax.xaxis.set_major_formatter(formater)
        ax.xaxis.set_major_locator(locator)
        
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()