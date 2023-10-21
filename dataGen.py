import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from mergedeep import merge, Strategy
np.random.seed(10)


class Gen2():
    def __init__(self, args, n_samples=10, periode=30, step=5, val=1000, nchannels=3, effects=None, fast=True):

        # get the parameters
        self.step = args.step
        self.effects = effects
        self.nchannels = args.n_channels
        self.val = args.val
        self.n_samples = args.n_samples
        self.fast = fast
        
        self.effects_params = {
            "Pulse": {
                "channel": [],
                "index": [],
                "amplitude": []
            },
            "Trend": {
                "channel": [],
                "index": [],
                "slope": []
            },
            "Seasonality": {
                "channel": [],
                "frequency_per_week": [],
                "amplitude": [],
                "phaseshift": []
            },
            "Std_variation": {
                "channel": [],
                "start": [],
                "end": [],
                "amplitude": []
            },
            "Channels_Coupling": {
                "channels": [],
                "amplitude": []
            },
            "Noise": {
                "channel": [],
                "index": [],
                "slope": []
            }
        }

        # generate the time axis
        min_per_day = 1440
        self.periode = args.periode * min_per_day // self.step # convert in minutes
        self.t = np.arange(self.periode)  # time axis
        self.n = self.t.shape[0]  # number of points in time axis
        self.reference_time = np.datetime64('2023-03-01T00:00:00')  # Reference time (for plausibility and plots)

        # generate y values
        self.mu = np.random.randint(self.val, size=self.nchannels * self.n_samples).astype(
            np.float16)  # mean values for each channel
        self.cov = np.ones(self.nchannels * self.n_samples).astype(np.float16)  # diag cov matrix for each channel
        if self.fast:
            self.add_std_variation(effects["std_variation"])
            self.x = np.random.multivariate_normal(self.mu, np.diag(self.cov).astype(np.float16), size=self.n).T
        else:
            self.mu = np.tile(self.mu, (self.n, 1)).T.astype(np.float16)  # expand the means over the time axis
            self.cov = np.tile(self.cov, (self.n, 1)).T.astype(np.float16)  # expand the covs over the time axis

        # add effects (noise)
        self.add_effects(effects)

        # generate the different timeseries: multivariate normal dist
        self.sample()

    def sample(self):
        if not self.fast:
#             self.x = np.random.multivariate_normal(self.mu, np.diag(self.cov).astype(np.float16), size=self.n).T
#         else:
            self.x = np.array([np.random.multivariate_normal(self.mu[:, obs], np.diag(self.cov[:, obs]).astype(np.float16))
                           for obs in range(self.n)]).T

        self.add_noise()
        self.x = np.reshape(self.x, (self.n_samples, self.nchannels, -1))

        self.params = {
            "n": self.n,
            "nchannels": self.nchannels,
            "n_samples": self.n_samples,

            "mu": np.reshape(self.mu, (self.n_samples, self.nchannels, -1)),
            "cov": np.reshape(self.cov, (self.n_samples, self.nchannels, -1))
        }

    # plots the generated data
    def show(self, n_samples=None):
        self.plot_time_series("Generated MTS", n_samples)

    # returns the Time series and their parameters
    def parameters(self):
        return self.x, self.params, self.effects_params

    # loops through all the input effects and calls the respective function for each effect
    def add_effects(self, effects):

        if effects is not None:
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
        occ = params["occurances"]  # number of Pulses.
        amp = params["max_amplitude"]  # max amplitude of the pulse
        interval = params["interval"]  # length of interval on which pulse will be applied
        pulse_start = params["start"]  # start index of the pulse
        n_occ = self.n_samples * occ

        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

        ### create randomised Pulses parameters
        channels = np.random.randint(self.nchannels,
                                     size=n_occ) + transform  # On which channel will the effect be applied.
        print(pulse_start)
        interval = self.check_interval_length(interval)  # Length of the generated Pulse
        start_idxs = self.set_start_idxs(pulse_start, occ, interval)  # At which index will the Pulse end.
        end_idxs = start_idxs + np.random.randint(low=1, high=interval,
                                                  size=n_occ)  # At which index will the Pulse end.
        amplitude = np.random.uniform(low=-amp, high=amp, size=n_occ)  # How strong is the Pulse.

        # save the Pulses parameters
        self.save_generated_effect("Pulse", [channels - transform, start_idxs, amplitude])

        # generate the pulses
        # original value at the pulse indexes
        print(start_idxs.shape)
        print(self.x.shape)
        print(self.mu.shape)
        print(self.x.shape)
        print(channels.shape)
        ground_val = self.x[channels, start_idxs].astype(np.int8) if self.fast else self.mu[channels, start_idxs].astype(np.int8)  
        
        amp_sign = 0.01 * np.sign(amp)
        k = np.random.uniform(ground_val + amp_sign, ground_val * amplitude)  # new values

        # add it to the channels
        for i in range(n_occ):
            if self.fast:
                self.x[channels[i], start_idxs[i]: end_idxs[i]] += k[i].astype(np.float16)
            else:
                self.mu[channels[i], start_idxs[i]: end_idxs[i]] += k[i].astype(np.float16)
                
    def add_random_pulse(self, percentage, amp):
        # Merge samples and channels
        if self.fast:
            self.x = np.reshape(self.x, (self.n_samples * self.nchannels, -1))

        # extract parameters:        
        occ, start_idxs, end_idxs = self.set_random_start(percentage)
        n_occ = self.n_samples * occ

        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

        ### create randomised Pulses parameters
        channels = np.random.randint(self.nchannels,
                                     size=n_occ) + transform  # On which channel will the effect be applied.
        amplitude = np.random.uniform(low=-amp, high=amp, size=n_occ)  # How strong is the Pulse.

        # save the Pulses parameters
        self.save_generated_effect("Pulse", [channels - transform, start_idxs, amplitude])

        # generate the pulses
        # original value at the pulse indexes
        ground_val = self.x[channels, start_idxs].astype(np.int8) if self.fast else self.mu[channels, start_idxs].astype(np.int8)  
        
        amp_sign = 0.01 * np.sign(amp)
        k = np.random.uniform(ground_val + amp_sign, ground_val * amplitude)  # new values

        # add it to the channels
        for i in range(n_occ):
            if self.fast:
                self.x[channels[i], start_idxs[i]: end_idxs[i]] += k[i].astype(np.float16)
            else:
                self.mu[channels[i], start_idxs[i]: end_idxs[i]] += k[i].astype(np.float16)
        if self.fast:
            self.x = np.reshape(self.x, (self.n_samples, self.nchannels, -1))

    def add_trend(self, params):

        # extract parameters:
        occ = params["occurances"]  # number of Trends.
        slope = params["max_slope"]  # Max slope of the Trends
        trend_type = params["type"]  # linear or quadratic or mixed trends
        trend_start = params["start"]
        n_occ = self.n_samples * occ

        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

        ### create randomised Trends parameters
        channels = np.random.randint(self.nchannels,
                                     size=n_occ) + transform  # On which channel will the Trend be applied.
        start_idxs = self.set_start_idxs(trend_start, occ)
        slopes = np.random.uniform(low=-slope, high=slope, size=n_occ)  # Slope of the Trend.

        # save the Trends parameters
        # idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
        self.save_generated_effect("Trend", [channels - transform, start_idxs, slopes])

        # generate the trends
        trends = np.zeros_like(self.x) if self.fast else np.zeros_like(self.mu)
        for channel, idx in enumerate(start_idxs):
            shifted = len(self.t) - idx
            ch = channels[channel]
            if trend_type == "linear":
                trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)
            elif trend_type == "quadratic":
                trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted) ** 2
            elif trend_type == "mixed":
                trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted) ** (
                            (channel % 2) + 1)

        # add it to the channels
        if self.fast:
            self.x += trends
        else:
            self.mu += trends

    def add_seasonality(self, params):

        # extract parameters:
        occ = params["occurances"]  # number of Seasonalities.
        freq = params["frequency_per_week"]  # frequency per Week
        amp = params["amplitude_range"]  # max amplitudes per week
        season_start = params["start"]
        n_occ = self.n_samples * occ

        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

        ### create randomised Seasonalities parameters
        channels = np.random.randint(self.nchannels,
                                     size=n_occ) + transform  # On which channel will the seasonality be applied.
        freqs = np.random.uniform(low=freq[0], high=freq[1], size=n_occ)  # frequency per week
        amps = np.random.randint(low=amp[0], high=amp[1], size=n_occ)  # max amplitude
        phases = np.random.randint(180, size=n_occ)  # shift to be applied
        start_idxs = self.set_start_idxs(season_start, occ)

        # save the Trends parameters
        self.save_generated_effect("Seasonality", [channels - transform, freqs, amps, phases])

        # generate the seasonalites
        seas = np.zeros_like(self.x) if self.fast else np.zeros_like(self.mu)
        for idx, channel in enumerate(channels):
            index = start_idxs[idx]
            seas[channel, index:] = np.maximum(seas[channel, index:], np.sin(
                2 * np.pi * self.t[index:] * freqs[idx] / (24 * 12 * 7) + phases[idx]) * amps[idx])

        # add it to the channels
        if self.fast:
            self.x += seas
        else:
            self.mu += seas


    def add_std_variation(self, params):

        # extract parameters:
        occ = params["occurances"]  # number of std variations.
        max_value = params["max_value"]  # max values of std.
        interval = params["interval"]  # length of the interval on which the std variates
        start = params["start"]
        n_occ = self.n_samples * occ

        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

        ### create randomised std variations parameters
        channels = np.random.randint(self.nchannels,
                                     size=n_occ) + transform  # # On which channel will the seasonality be applied.
        interval = self.check_interval_length(interval)
        start_idxs = self.set_start_idxs(start, occ, interval)  # At which index will the std variation start.
        end_idxs = start_idxs + np.random.randint(interval, size=occ)  # At which index will the std variation end.
        amps = np.random.uniform(high=max_value, size=n_occ)
        intervals = end_idxs - start_idxs

        # save std variations parameters
        self.save_generated_effect("Std_variation", [channels - transform, start_idxs, end_idxs, amps])

        # add it to the channels
        for i in range(n_occ):
            ch = channels[i]
            # amplitude = np.random.uniform(high = max_value, size = (1, intervals[i]))

            self.cov[ch, ch, start_idxs[i]: end_idxs[i]] = amps[i]
            # self.effects_params["Std_variation"]["amplitude"].extend(amplitude)

    def add_channels_coupling(self, params):

        # extract parameters:
        occ = params["occurances"]  # number of channels_coupling.
        max_value = params["coupling_strengh"]  # max values of std.
        #         interval = params["interval"] # length of the interval on which the std variates

        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

        ### create randomised std variations parameters
        channels = np.random.randint(self.nchannels,
                                     size=(occ, 2))  # # On which channel will the seasonality be applied.
        #         start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the std variation start.
        #         end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the std variation end.
        #         intervals = end_idxs - start_idxs

        # save std variations parameters
        #         start_idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
        #         end_idxs_to_time = self.reference_time + (end_idxs * self.step).astype('timedelta64[m]')
        #         durations = (end_idxs_to_time- start_idxs_to_time).astype('timedelta64[D]')

        self.effects_params["Channels_Coupling"]["channels"].extend(channels)
        #         self.effects_params["Std_variation"]["interval"].extend(durations.astype('str'))

        # add it to the channels
        for i in range(occ):
            ch = channels[i]
            amplitude = np.random.uniform(high=max_value)

            self.cov[ch[0], ch[1], :] = amplitude
            self.effects_params["Channels_Coupling"]["amplitude"].append(amplitude)

    def add_noise(self):

        if self.effects is not None:
            params = self.effects["Noise"]

            if params["occurances"] != 0:

                # extract parameters:
                occ = params["occurances"] * self.n_samples  # number of Trends.
                slope = params["max_slope"]  # Max slope of the Trends
                noise_type = params["type"]  # linear or quadratic or mixed trends

                # transformation array to map the effect on the corresponding channels in each sample
                transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

                ### create randomised Noise parameters
                channels = np.random.randint(self.nchannels,
                                             size=self.n_samples * occ) + transform  # On which channel will the Trend be applied.
                idxs = np.random.randint(self.n, size=occ)  # At which index will the Trend start.
                slopes = np.random.uniform(low=-slope, high=slope, size=occ)  # Slope of the Trend.

                # save the Noise parameters
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
                        noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted) ** 2
                    elif noise_type == "mixed":
                        noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted) ** (
                                    (channel % 2) + 1)

                # add it to the channels
                self.x += noises

    def plot_time_series(self, title, n_samples):

        date_array = self.reference_time + np.array(self.t * self.step, dtype='timedelta64[m]')

        combined_channel_samples = self.x.shape[0] * self.nchannels
        x_reshaped = np.reshape(self.x, (combined_channel_samples, -1)).T

        if n_samples != None:
            x_reshaped = x_reshaped[..., :n_samples]

        duration = np.max(date_array) - np.min(date_array)
        one_month = np.timedelta64(1, 'M').astype('timedelta64[m]')

        formater = mdates.DateFormatter('%A')
        locator = mdates.DayLocator()
        if duration > one_month / 2 and duration <= 2 * one_month:
            locator = mdates.DayLocator(interval=2)
        elif duration > 2 * one_month:
            locator = mdates.WeekdayLocator()
            formater = mdates.DateFormatter('%Y-%m-%d')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(date_array, x_reshaped)
        ax.set(xlabel='Date', ylabel='Value', title=title)

        ax.xaxis.set_major_formatter(formater)
        ax.xaxis.set_major_locator(locator)

        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
    def save_generated_effect(self, effect, params):
        current_effect = self.effects_params[effect]
        if len(params) != len(current_effect):
            print("number of parameters doesn't match parameters in ", effect)
            return
        for i, existing_param in enumerate(current_effect):

            new_param = list(np.reshape(params[i], (self.n_samples, -1)))
            current_effect[existing_param].extend(new_param)

    # Adjusts the start index to be in range [0, (self.n-interval)] and generates a list of start indexes
    # containing occ * n_samples random indexes
    # start:    float in range [0, 1]  where the effect should start
    #           0 = at 0%
    #           1 = 100 % of time series
    #           None = random value
    # occ: number of occ per sample
    # interval: the interval where effect is applied, if not 0 then max value of start = length - interval
    def set_start_idxs(self, start, occ, interval=0):
        max_length = self.n - interval
        n_occ = self.n_samples * occ
        if start == None:
            start = np.random.uniform(size=n_occ)
        elif start > 1:
            start = np.ones(n_occ, dtype=np.int8)
        elif start < 0:
            start = np.zeros(n_occ, dtype=np.int8)
            
        start_idxs = np.array( start * max_length, dtype=np.int32)        
        return start_idxs
    
    def set_random_start(self, percentage):
        # Define the desired percentage (x)

        # Calculate the total number of indices you want
        total_indices = int(percentage * self.n)  # Adjust 1000 based on your total number of points

        # Define the minimum and maximum group size
        min_group_size = 1
        max_group_size = 5
        mean_group_size = (max_group_size + min_group_size)//2
        max_length = self.n - max_group_size

        # Calculate the number of groups based on the total indices and the maximum group size
        num_groups = int(total_indices / mean_group_size)

        # Generate random group sizes between min and max
        group_sizes = np.random.randint(min_group_size, max_group_size + 1, num_groups)

        # Ensure the total number of indices is within the desired range
        while sum(group_sizes) != total_indices:
            diff = total_indices - sum(group_sizes)
            grp_choice = np.random.choice(num_groups)
            if diff > 0:                
                diff_to_add = max_group_size - group_sizes[grp_choice]
                group_sizes[grp_choice] += diff_to_add
            else:
                diff_to_add = group_sizes[grp_choice] - min_group_size
                group_sizes[grp_choice] -= diff_to_add
                
        
        n_occ = self.n_samples * num_groups
        start = np.random.uniform(size=n_occ)
        start_idxs = np.array( start * max_length, dtype=np.int32)
        print(start_idxs.shape)
        
        group_sizes = np.tile(group_sizes, self.n_samples)
        end_idxs = start_idxs + group_sizes
            
        return num_groups, start_idxs, end_idxs
        
        

    def check_interval_length(self, interval, treshold=1):
        if treshold == 0:
            treshold = 1
        if interval >= self.n // treshold:
            print("Adjusting the Interval")
            print("old one", interval)
            print("limit", self.n//treshold)
            interval = self.n // treshold
            print("new one", interval)
        elif interval < 0:
            interval = 0
        return interval

    def merge(self, Y):
        y, params_y, e_params_y = Y.parameters()

        self.x = np.concatenate((self.x, y), axis=0)
        self.params["n_samples"] += params_y["n_samples"]
        self.n_samples += params_y["n_samples"]
        self.params["mu"] = np.concatenate((self.params["mu"], params_y["mu"]), axis=0)
        self.params["cov"] = np.concatenate((self.params["cov"], params_y["cov"]), axis=0)

        self.e_params = merge(self.effects_params, e_params_y, strategy=Strategy.ADDITIVE)
class Gen():
    def __init__(self, args, n_samples = 10, periode = 30, step = 5, val = 1000, nchannels = 3, effects = None):
        
        # get the parameters
        self.step = args.step
        self.effects = effects
        self.nchannels = args.nchannels
        self.val = args.val
        self.n_samples = args.n_samples
        
        # generate the time axis
        min_per_day = 1440
        self.periode = args.periode * min_per_day # convert in minutes
        self.t = np.arange(self.periode, step = self.step) # time axis
        self.n = self.t.shape[0] # number of points in time axis
        self.reference_time = np.datetime64('2023-03-01T00:00:00') # Reference time (for plausibility and plots)
        
        
        # generate y values
        self.mu = np.random.randint(self.val, size=self.nchannels * self.n_samples).astype(np.float16) # mean values for each channel
        self.mu = np.tile(self.mu, (self.n,1)).T.astype(np.float16) # expand the means over the time axis
#         self.mu = np.lib.stride_tricks.as_strided(self.mu, shape=(len(self.mu), self.n), strides=(0, self.mu.itemsize))
     
        self.cov = np.ones(self.nchannels * self.n_samples).astype(np.float16) # diag cov matrix for each channel        
        self.cov = np.tile(self.cov, (self.n,1)).T.astype(np.float16) # expand the covs over the time axis
#         self.cov = np.lib.stride_tricks.as_strided(self.cov, shape=(self.cov.shape[0], self.cov.shape[1], self.n),
#                                          strides=(self.cov.strides[0], self.cov.strides[1], 0))
       
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
                "start":[],
                "end":[],
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
        self.add_effects(effects) 
        
        # generate the different timeseries: multivariate normal dist
        self.sample()
    
    def sample(self):
        self.x = np.array([np.random.multivariate_normal(self.mu[:,obs], np.diag(self.cov[:,obs]).astype(np.float16))
                           for obs in range(self.n)]).T
        
        self.add_noise()
        self.x = np.reshape(self.x, (self.n_samples, self.nchannels, -1))
        
        self.params = {
            "n": self.n,
            "nchannels":self.nchannels,
            "n_samples":self.n_samples,

            "mu":np.reshape(self.mu, (self.n_samples, self.nchannels, -1)),
            "cov":np.reshape(self.cov, (self.n_samples, self.nchannels, -1))
        }
        
    #plots the generated data
    def show(self, n_samples=None): 
        self.plot_time_series("Generated MTS", n_samples)

    #returns the Time series and their parameters
    def parameters(self):         
        return self.x, self.params, self.effects_params

    # loops through all the input effects and calls the respective function for each effect
    def add_effects(self, effects): 
        
        if effects is not None:
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
        occ = params["occurances"]    # number of Pulses.
        amp = params["max_amplitude"] # max amplitude of the pulse
        interval = params["interval"] # length of interval on which pulse will be applied
        pulse_start = params["start"] # start index of the pulse
        n_occ = self.n_samples* occ
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        ### create randomised Pulses parameters       
        channels = np.random.randint(self.nchannels, size=n_occ) + transform # On which channel will the effect be applied.
        interval = self.check_interval_length(interval) # Length of the generated Pulse
        start_idxs = self.set_start_idxs(pulse_start, occ, interval) # At which index will the Pulse end.
        end_idxs = start_idxs + np.random.randint(low = 1, high=interval, size=n_occ) # At which index will the Pulse end.
        amplitude = np.random.uniform(low = -amp, high = amp, size=n_occ) # How strong is the Pulse.
        
        #save the Pulses parameters
        self.save_generated_effect("Pulse", [channels - transform, start_idxs, amplitude])

        # generate the pulses
        ground_val = self.mu[channels, start_idxs].astype(np.int8) # original value at the pulse indexes
        amp_sign = 0.01 * np.sign(amp)
        k = np.random.uniform(ground_val + amp_sign, ground_val*amplitude) # new values

        # add it to the channels
        for i in range(n_occ):
            self.mu[channels[i],start_idxs[i]: end_idxs[i]] += k[i].astype(np.float16)

    def save_generated_effect(self, effect, params):
        current_effect = self.effects_params[effect]
        if len(params) != len(current_effect):
            print("number of parameters doesn't match parameters in ", effect)
            return
        for i, existing_param in enumerate(current_effect):

            new_param = list(np.reshape(params[i], (self.n_samples, -1)))
            current_effect[existing_param].extend(new_param)

    # Adjusts the start index to be in range [0, (self.n-interval)] and generates a list of start indexes
    # containing occ * n_samples random indexes
    # start:    float in range [0, 1]  where the effect should start
    #           0 = at 0%
    #           1 = 100 % of time series
    #           None = random value
    # occ: number of occ per sample
    # interval: the interval where effect is applied, if not 0 then max value of start = length - interval
    def set_start_idxs(self, start, occ, interval=0):
        max_length = self.n - interval
        n_occ = self.n_samples * occ
        if start == None:
            start = np.random.uniform()
        elif start > 1:
            start = 1
        elif start < 0:
            start = 0

        start_idx = int(start * max_length)
        start_idxs = int(start_idx) * np.ones(n_occ, dtype=np.int8)
        return start_idxs

    def check_interval_length(self, interval, treshold=1):
        if treshold == 0:
            treshold = 1
        if interval >= self.n // treshold:
            print("Adjusting the Interval")
            interval = self.n // treshold
        elif interval < 0:
            interval = 0
        return interval

    def add_trend(self, params):
        
        # extract parameters:
        occ = params["occurances"]  # number of Trends.   
        slope = params["max_slope"] # Max slope of the Trends  
        trend_type = params["type"] # linear or quadratic or mixed trends
        trend_start= params["start"]
        n_occ = self.n_samples * occ
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        ### create randomised Trends parameters
        channels = np.random.randint(self.nchannels, size=n_occ) + transform # On which channel will the Trend be applied.
        start_idxs = self.set_start_idxs(trend_start, occ)
        slopes = np.random.uniform(low = -slope, high = slope, size=n_occ)  # Slope of the Trend.
        
        
        #save the Trends parameters
        # idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
        self.save_generated_effect("Trend", [channels - transform, start_idxs, slopes])
        
        # generate the trends
        trends = np.zeros_like(self.mu)
        for channel, idx in enumerate(start_idxs):
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
        season_start = params["start"]
        n_occ = self.n_samples* occ
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        ### create randomised Seasonalities parameters
        channels = np.random.randint(self.nchannels, size=n_occ) + transform # On which channel will the seasonality be applied.
        freqs = np.random.uniform(low = freq[0], high = freq[1], size=n_occ) # frequency per week
        amps = np.random.randint(low = amp[0], high = amp[1], size=n_occ) # max amplitude
        phases = np.random.randint(180, size=n_occ) # shift to be applied
        start_idxs = self.set_start_idxs(season_start, occ)

        #save the Trends parameters
        self.save_generated_effect("Seasonality", [channels - transform, freqs, amps, phases])
        
        # generate the seasonalites
        seas = np.zeros_like(self.mu)
        for idx, channel in enumerate(channels):
            index = start_idxs[idx]
            seas[channel, index:] = np.maximum(seas[channel, index:], np.sin(2 * np.pi * self.t[index:] * freqs[idx] / (24 * 60 * 7) + phases[idx])* amps[idx])
        
         # add it to the channels
        self.mu += seas
        
    def add_std_variation(self, params):
        
        # extract parameters:
        occ = params["occurances"] # number of std variations.
        max_value = params["max_value"] # max values of std.
        interval = params["interval"] # length of the interval on which the std variates
        start = params["start"]
        n_occ = self.n_samples * occ
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        ### create randomised std variations parameters
        channels = np.random.randint(self.nchannels, size=n_occ) + transform #  # On which channel will the seasonality be applied.
        interval = self.check_interval_length(interval)
        start_idxs = self.set_start_idxs(start, occ, interval)  # At which index will the std variation start.
        end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the std variation end.
        amps = np.random.uniform( high = max_value, size=n_occ)
        intervals = end_idxs - start_idxs
        
        
        #save std variations parameters
        self.save_generated_effect("Std_variation", [channels - transform, start_idxs, end_idxs, amps])
        
        # add it to the channels
        for i in range(n_occ):
            ch = channels[i]
            # amplitude = np.random.uniform(high = max_value, size = (1, intervals[i]))
            
            self.cov[ch, ch, start_idxs[i]: end_idxs[i]] = amps[i]
            # self.effects_params["Std_variation"]["amplitude"].extend(amplitude)
            
    def add_channels_coupling(self, params):
        
        # extract parameters:
        occ = params["occurances"] # number of channels_coupling.
        max_value = params["coupling_strengh"] # max values of std.
#         interval = params["interval"] # length of the interval on which the std variates
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
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
                occ = params["occurances"] * self.n_samples  # number of Trends.   
                slope = params["max_slope"] # Max slope of the Trends  
                noise_type = params["type"] # linear or quadratic or mixed trends
                
                # transformation array to map the effect on the corresponding channels in each sample
                transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

                ### create randomised Noise parameters
                channels = np.random.randint(self.nchannels, size=self.n_samples* occ) + transform # On which channel will the Trend be applied.
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
        
        
    def plot_time_series(self, title, n_samples): 
        
        date_array = self.reference_time + np.array(self.t, dtype='timedelta64[m]')        
        
        combined_channel_samples = self.x.shape[0]*self.nchannels        
        x_reshaped = np.reshape(self.x, (combined_channel_samples, -1)).T
        
        if n_samples != None:
            x_reshaped = x_reshaped[..., :n_samples]
            
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
        ax.plot(date_array, x_reshaped)
        ax.set(xlabel='Date', ylabel='Value', title=title)
        
        ax.xaxis.set_major_formatter(formater)
        ax.xaxis.set_major_locator(locator)
        
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    def merge(self, Y):
        y, params_y, e_params_y = Y.parameters()

        self.x = np.concatenate((self.x, y), axis=0)
        self.params["n_samples"] += params_y["n_samples"]
        self.params["mu"] = np.concatenate((self.params["mu"], params_y["mu"]), axis=0)
        self.params["cov"] = np.concatenate((self.params["cov"], params_y["cov"]), axis=0)
        
        self.e_params = merge(self.effects_params, e_params_y, strategy=Strategy.ADDITIVE)
        
        
class FastGen():
    def __init__(self, args, effects = None):  
        
        # get the parameters
        self.step = args.step
        self.effects = effects
        self.nchannels = args.nchannels
        self.val = args.val
        self.n_samples = args.n_samples       
        
        # generate the time axis
        min_per_day = 1440
        self.periode = args.periode * min_per_day # convert in minutes
        self.t = np.arange(self.periode, step = self.step) # time axis
        self.n = self.t.shape[0] # number of points in time axis
        self.reference_time = np.datetime64('2023-03-01T00:00:00') # Reference time (for plausibility and plots)
        
        
        # generate y values
        self.mu = np.random.randint(self.val, size=self.nchannels * self.n_samples).astype(np.float16) # mean values for each channel
     
        self.cov = np.ones(self.nchannels * self.n_samples).astype(np.float16) # diag cov matrix for each channel        
#         self.cov = np.tile(self.cov, (self.n,1)).T.astype(np.float16) # expand the covs over the time axis
#         self.cov = np.lib.stride_tricks.as_strided(self.cov, shape=(self.cov.shape[0], self.cov.shape[1], self.n),
#                                          strides=(self.cov.strides[0], self.cov.strides[1], 0))
       
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
#         print(self.mu.shape, self.mu)
#         print(self.cov.shape, self.cov)
#         self.add_std_variation(self.effects["std_variation"])
        self.x = np.random.multivariate_normal(self.mu, np.diag(self.cov).astype(np.float16), size=self.n).T
#         print(self.x.shape)
        
        # add effects (noise)
        self.add_effects(self.effects) 
        
        # generate the different timeseries: multivariate normal dist
#         self.x = np.array([np.random.multivariate_normal(self.mu[:,obs], np.diag(self.cov[:,obs]).astype(np.float16)) for obs in range(self.n)]).T
#         print(self.x.shape)
        self.add_noise()
        self.x = np.reshape(self.x, (self.n_samples, self.nchannels, -1))
        
        self.params = {
            "n": self.n,
            "nchannels":self.nchannels,
            "n_samples":self.n_samples,

            "mu":np.reshape(self.mu, (self.n_samples, self.nchannels, -1)),
            "cov":np.reshape(self.cov, (self.n_samples, self.nchannels, -1))
        }

    def sample(self):
        self.x = np.array([np.random.multivariate_normal(self.mu[:, obs], np.diag(self.cov[:, obs]).astype(np.float16))
                           for obs in range(self.n)]).T
        self.x = np.random.multivariate_normal(self.mu, np.diag(self.cov).astype(np.float16), size=self.n).T

        self.add_noise()
        self.x = np.reshape(self.x, (self.n_samples, self.nchannels, -1))

        self.params = {
            "n": self.n,
            "nchannels": self.nchannels,
            "n_samples": self.n_samples,

            "mu": np.reshape(self.mu, (self.n_samples, self.nchannels, -1)),
            "cov": np.reshape(self.cov, (self.n_samples, self.nchannels, -1))
        }
    
    
    #plots the generated data
    def show(self, n_samples=None): 
        self.plot_time_series("Generated MTS", n_samples)
        
        
    #returns the Time series and their parameters
    def parameters(self):         
        return self.x, self.params, self.effects_params
    
    
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
        occ = params["occurances"]    # number of Pulses.
        amp = params["max_amplitude"] # max amplitude of the pulse
        interval = params["interval"] # length of interval on which pulse will be applied
        pulse_start = params["start"]
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        
        ### create randomised Pulses parameters       
        channels = np.random.randint(self.nchannels, size=self.n_samples* occ) + transform # On which channel will the effect be applied.            
        if pulse_start == None:
            start_idxs = np.random.randint(self.n - interval, size=self.n_samples* occ) # At which index will the Pulse start.
        else:
            if pulse_start + interval > self.n:
                print("pulse start index adjusted to the max value (start + interval exceed time series range!)")
                pulse_start = self.n - interval
            if pulse_start < 0:
                print("pulse start index adjusted to 0 (start is negative!)")
                pulse_start = 0
            start_idxs = int(pulse_start) * np.ones(self.n_samples* occ, dtype=np.int8)
            
#         start_idxs = np.random.randint(self.n - interval, size=self.n_samples* occ) # At which index will the Pulse start.
        end_idxs = start_idxs + np.random.randint(interval, size=self.n_samples* occ) # At which index will the Pulse end.
        amplitude = np.random.uniform(low = -amp, high = amp, size=self.n_samples* occ) # How strong is the Pulse.
        
        
        #save the Pulses parameters
        idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
        self.effects_params["Pulse"]["channel"].extend(np.reshape(channels - transform, (self.n_samples, -1)))
        self.effects_params["Pulse"]["index"].extend(np.reshape(idxs_to_time.astype('str'), (self.n_samples, -1) ))
        self.effects_params["Pulse"]["amplitude"].extend(np.reshape(amplitude, (self.n_samples, -1)))
        
        
        # generate the pulses
        ground_val = self.x[channels, start_idxs].astype(np.int8) # original value at the pulse indexes
        k = np.random.uniform(ground_val, ground_val*amplitude) # new values 

        # add it to the channels
        for i in range(self.n_samples* occ):
            self.x[channels[i],start_idxs[i]: end_idxs[i]] += k[i].astype(np.float16)
        
    
    def add_trend(self, params):
        
        # extract parameters:
        occ = params["occurances"]  # number of Trends.   
        slope = params["max_slope"] # Max slope of the Trends  
        trend_type = params["type"] # linear or quadratic or mixed trends
        trend_start = params["start"]
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        ### create randomised Trends parameters
        channels = np.random.randint(self.nchannels, size=self.n_samples* occ) + transform # On which channel will the Trend be applied.
        if trend_start == None:
            idxs = np.random.randint(self.n, size=self.n_samples* occ) # At which index will the Trend start.
        else:
            if trend_start > self.n:
                print("trend start index adjusted to 80% of time series (start + interval exceed time series range!)")
                trend_start = self.n*0.8
            if trend_start < 0:
                print("trend start index adjusted to 0 (start is negative!)")
                trend_start = 0
            idxs = int(trend_start) * np.ones(self.n_samples* occ, dtype=np.int8)
            
#         idxs = np.random.randint(self.n, size=self.n_samples* occ) # At which index will the Trend start.        
        slopes = np.random.uniform(low = -slope, high = slope, size=self.n_samples* occ)  # Slope of the Trend.
        
        
        #save the Trends parameters
        idxs_to_time = self.reference_time + (idxs * self.step).astype('timedelta64[m]')
        self.effects_params["Trend"]["channel"].extend(np.reshape(channels - transform, (self.n_samples, -1)))
        self.effects_params["Trend"]["index"].extend(np.reshape(idxs_to_time.astype('str'), (self.n_samples, -1)))
        self.effects_params["Trend"]["slope"].extend(np.reshape(slopes, (self.n_samples, -1)))
        
        
        # generate the trends
        trends = np.zeros_like(self.x)
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
        self.x += trends
        
    def add_seasonality(self,params):
        
        # extract parameters:
        occ = params["occurances"] # number of Seasonalities.
        freq = params["frequency_per_week"] # frequency per Week
        amp = params["amplitude_range"] # max amplitudes per week
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        ### create randomised Seasonalities parameters
        channels = np.random.randint(self.nchannels, size=self.n_samples* occ) + transform # On which channel will the seasonality be applied.
        freqs = np.random.uniform(low = freq[0], high = freq[1], size=self.n_samples* occ) # frequency per week
        amps = np.random.randint(low = amp[0], high = amp[1], size=self.n_samples* occ) # max amplitude 
        phases = np.random.randint(180, size=self.n_samples* occ) # shift to be applied
        

        #save the Trends parameters
        self.effects_params["Seasonality"]["channel"].extend(np.reshape(channels - transform, (self.n_samples, -1)))
        self.effects_params["Seasonality"]["frequency_per_week"].extend(np.reshape(freqs, (self.n_samples, -1)))
        self.effects_params["Seasonality"]["amplitude"].extend(np.reshape(amps, (self.n_samples, -1)))
        self.effects_params["Seasonality"]["phaseshift"].extend(np.reshape(phases, (self.n_samples, -1)))
        
        
        # generate the seasonalites
        seas = np.zeros_like(self.x)
        for idx, channel in enumerate(channels):
            seas[channel] = np.maximum(seas[channel], np.sin(2 * np.pi * self.t * freqs[idx] / (24 * 60 * 7) + phases[idx])* amps[idx])
        
         # add it to the channels
        self.x += seas
        
    def add_std_variation(self, params):
        
        # extract parameters:
        occ = params["occurances"] # number of std variations.
        max_value = params["max_value"] # max values of std.
        interval = params["interval"] # length of the interval on which the std variates
        std_var_start = params["start"] # where the std variation starts
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
        ### create randomised std variations parameters
        channels = np.random.randint(self.nchannels, size=self.n_samples* occ) + transform #  # On which channel will the seasonality be applied.
        if std_var_start == None and interval == None:
            std_var_start = 0
            interval = self.n
        if std_var_start == None:
            start_idxs = np.random.randint(self.n - interval, size=self.n_samples* occ) # At which index will the Pulse start.
        else:
            if std_var_start + interval > self.n:
                print("std var start index adjusted to the max value (start + interval exceed time series range!)")
                std_var_start = self.n - interval
            if std_var_start < 0:
                print("std var start index adjusted to 0 (start is negative!)")
                std_var_start = 0
            start_idxs = int(std_var_start) * np.ones(self.n_samples* occ, dtype=np.int8)
        if interval == None:
            interval = self.n - std_var_start
#         start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the std variation start.
        end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the std variation end.
        intervals = end_idxs - start_idxs
        
        
        #save std variations parameters
        start_idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
        end_idxs_to_time = self.reference_time + (end_idxs * self.step).astype('timedelta64[m]')
        durations = (end_idxs_to_time- start_idxs_to_time).astype('timedelta64[D]')
        
        self.effects_params["Std_variation"]["channel"].extend(np.reshape(channels, (self.n_samples, -1)))
        self.effects_params["Std_variation"]["interval"].extend(np.reshape(durations.astype('str'), (self.n_samples, -1)))        
        
        
        # add it to the channels
        for i in range(self.n_samples* occ):
            ch = channels[i]
#             amplitude = np.random.uniform(high = max_value, size = (1, intervals[i]))
#             print(self.cov.shape)
#             print(
            amplitude = np.random.uniform(high = max_value)
            self.cov[ch] = amplitude
#             self.cov[ch, ch, start_idxs[i]: end_idxs[i]] = amplitude
            self.effects_params["Std_variation"]["amplitude"].extend([amplitude])
            
    def add_channels_coupling(self, params):
        
        # extract parameters:
        occ = params["occurances"] # number of channels_coupling.
        max_value = params["coupling_strengh"] # max values of std.
#         interval = params["interval"] # length of the interval on which the std variates
        
        # transformation array to map the effect on the corresponding channels in each sample
        transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)
        
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
                occ = params["occurances"] * self.n_samples  # number of Trends.   
                slope = params["max_slope"] # Max slope of the Trends  
                noise_type = params["type"] # linear or quadratic or mixed trends
                
                # transformation array to map the effect on the corresponding channels in each sample
                transform = np.repeat(np.arange(0, self.n_samples * self.nchannels, step=self.nchannels), occ)

                ### create randomised Noise parameters
                channels = np.random.randint(self.nchannels, size=self.n_samples* occ) + transform # On which channel will the Trend be applied.
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
        
        
    def plot_time_series(self, title, n_samples): 
        
        date_array = self.reference_time + np.array(self.t, dtype='timedelta64[m]')        
        
        combined_channel_samples = self.x.shape[0]*self.nchannels        
        x_reshaped = np.reshape(self.x, (combined_channel_samples, -1)).T
        
        if n_samples != None:
            x_reshaped = x_reshaped[..., :n_samples]
            
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
        ax.plot(date_array, x_reshaped)
        ax.set(xlabel='Date', ylabel='Value', title=title)
        
        ax.xaxis.set_major_formatter(formater)
        ax.xaxis.set_major_locator(locator)
        
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    def merge(self, Y):
        y, params_y, e_params_y = Y.parameters()

        self.x = np.concatenate((self.x, y), axis=0)
        self.params["n_samples"] += params_y["n_samples"]
        self.params["mu"] = np.concatenate((self.params["mu"], params_y["mu"]), axis=0)
        self.params["cov"] = np.concatenate((self.params["cov"], params_y["cov"]), axis=0)
        
        self.e_params = merge(self.effects_params, e_params_y, strategy=Strategy.ADDITIVE)
        
# class Gen():
#     def __init__(self, periode = 30, step = 5, val = 1000, nchannels = 3, effects = None):  
        
#         # get the parameters
#         self.step = step
#         self.effects = effects
#         self.nchannels = nchannels
#         self.val = val
        
        
#         # generate the time axis
#         min_per_day = 1440
#         self.periode = periode * min_per_day # convert in minutes
#         self.t = np.arange(self.periode, step = self.step) # time axis
#         self.n = self.t.shape[0] # number of points in time axis
#         self.reference_time = np.datetime64('2023-03-01T00:00:00') # Reference time (for plausibility and plots)
        
        
#         # generate y values
#         self.mu = np.random.randint(self.val, size=self.nchannels) # mean values for each channel
#         self.mu = np.tile(self.mu, (self.n,1)).T.astype(np.float32) # expand the means over the time axis

#         self.cov = np.diag(np.ones(self.nchannels)) # diag cov matrix for each channel
#         self.cov = np.tile(self.cov, (self.n,1,1)).T.astype(np.float32) # expand the covs over the time axis

        
#         self.effects_params = {
#             "Pulse":{
#                 "channel":[],
#                 "index":[],
#                 "amplitude":[]
#             },
#             "Trend":{
#                 "channel":[],
#                 "index":[],
#                 "slope":[]
#             },
#             "Seasonality":{
#                 "channel":[],
#                 "frequency_per_week":[],
#                 "amplitude":[],
#                 "phaseshift":[]
#             },  
#             "Std_variation":{
#                 "channel":[],
#                 "interval":[],
#                 "amplitude":[]
#             },
#             "Channels_Coupling":{
#                 "channels":[],
#                 "amplitude":[]
#             },
#             "Noise":{
#                 "channel":[],
#                 "index":[],
#                 "slope":[]
#             }
#         } 
        
#         # add effects (noise)
#         self.add_effects(self.effects) 
        
#         # generate the different timeseries: multivariate normal dist
#         self.x = np.array([np.random.multivariate_normal(self.mu[:,obs], self.cov[:,:,obs]) for obs in range(self.n)]).T    
        
#         self.add_noise()
    
    
#     #plots the generated data
#     def show(self): 
#         self.plot_time_series("Generated MTS")
        
        
#     #returns the Time series and their parameters
#     def parameters(self): 
        
#         params = {
#             "n": self.n,
#             "nchannels":len(self.mu),

#             "mu":self.mu,
#             "cov":self.cov
#         }
#         return self.x, params, self.effects_params
    
    
#     # loops through all the input effects and calls the respective function for each effect
#     def add_effects(self, effects): 
        
#         if self.effects is not None:
#             for effect, params in self.effects.items(): 
#                 if params["occurances"] == 0:
#                     continue
#                 if effect == "Pulse":
#                     self.add_pulse(params)                
#                 elif effect == "Trend":
#                     self.add_trend(params)
#                 elif effect == "Seasonality":
#                     self.add_seasonality(params)
#                 elif effect == "std_variation":
#                     self.add_std_variation(params)
#                 elif effect == "channels_coupling":
#                     self.add_channels_coupling(params)
    
    
#     # adds a pulse effect
#     def add_pulse(self, params):
        
#         # extract parameters: 
#         occ = params["occurances"] # number of Pulses.
#         amp = params["max_amplitude"] # max amplitude of the pulse
#         interval = params["interval"] # length of interval on which pulse will be applied
        
        
#         ### create randomised Pulses parameters       
#         channels = np.random.randint(self.nchannels, size=occ) # On which channel will the effect be applied.        
#         start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the Pulse start.
#         end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the Pulse end.
#         amplitude = np.random.uniform(low = -amp, high = amp, size=occ) # How strong is the Pulse.
        
        
#         #save the Pulses parameters
#         idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
#         self.effects_params["Pulse"]["channel"].extend(channels)
#         self.effects_params["Pulse"]["index"].extend(idxs_to_time.astype('str'))
#         self.effects_params["Pulse"]["amplitude"].extend(amplitude)
        
        
#         # generate the pulses
#         ground_val = self.mu[channels, start_idxs] # original value at the pulse indexes
#         k = np.random.uniform(ground_val, ground_val*amplitude) # new values 

#         # add it to the channels
#         for i in range(occ):
#             self.mu[channels[i],start_idxs[i]: end_idxs[i]] += k[i]
        
    
#     def add_trend(self, params):
        
#         # extract parameters:
#         occ = params["occurances"]  # number of Trends.   
#         slope = params["max_slope"] # Max slope of the Trends  
#         trend_type = params["type"] # linear or quadratic or mixed trends
        
        
#         ### create randomised Trends parameters
#         channels = np.random.randint(self.nchannels, size=occ) # On which channel will the Trend be applied.
#         idxs = np.random.randint(self.n, size=occ) # At which index will the Trend start.        
#         slopes = np.random.uniform(low = -slope, high = slope, size=occ)  # Slope of the Trend.
        
        
#         #save the Trends parameters
#         idxs_to_time = self.reference_time + (idxs * self.step).astype('timedelta64[m]')
#         self.effects_params["Trend"]["channel"].extend(channels)
#         self.effects_params["Trend"]["index"].extend(idxs_to_time.astype('str'))
#         self.effects_params["Trend"]["slope"].extend(slopes)
        
        
#         # generate the trends
#         trends = np.zeros_like(self.mu)
#         for channel, idx in enumerate(idxs):
#             shifted = len(self.t) - idx
#             ch = channels[channel]
#             if trend_type == "linear":
#                 trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)
#             elif trend_type == "quadratic":
#                 trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**2
#             elif trend_type == "mixed":
#                 trends[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**((channel%2)+1)

#         # add it to the channels
#         self.mu += trends
        
#     def add_seasonality(self,params):
        
#         # extract parameters:
#         occ = params["occurances"] # number of Seasonalities.
#         freq = params["frequency_per_week"] # frequency per Week
#         amp = params["amplitude_range"] # max amplitudes per week

        
#         ### create randomised Seasonalities parameters
#         channels = np.random.randint(self.nchannels, size=occ) # On which channel will the seasonality be applied.
#         freqs = np.random.uniform(low = freq[0], high = freq[1], size=occ) # frequency per week
#         amps = np.random.randint(low = amp[0], high = amp[1], size=occ) # max amplitude 
#         phases = np.random.randint(180, size=occ) # shift to be applied
        

#         #save the Trends parameters
#         self.effects_params["Seasonality"]["channel"].extend(channels)
#         self.effects_params["Seasonality"]["frequency_per_week"].extend(freqs)
#         self.effects_params["Seasonality"]["amplitude"].extend(amps)
#         self.effects_params["Seasonality"]["phaseshift"].extend(phases)
        
        
#         # generate the seasonalites
#         seas = np.zeros_like(self.mu)
#         for idx, channel in enumerate(channels):
#             seas[channel] = np.maximum(seas[channel], np.sin(2 * np.pi * self.t * freqs[idx] / (24 * 60 * 7) + phases[idx])* amps[idx])
        
#          # add it to the channels
#         self.mu += seas
        
#     def add_std_variation(self, params):
        
#         # extract parameters:
#         occ = params["occurances"] # number of std variations.
#         max_value = params["max_value"] # max values of std.
#         interval = params["interval"] # length of the interval on which the std variates
        
        
#         ### create randomised std variations parameters
#         channels = np.random.randint(self.nchannels, size=occ) #  # On which channel will the seasonality be applied.     
#         start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the std variation start.
#         end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the std variation end.
#         intervals = end_idxs - start_idxs
        
        
#         #save std variations parameters
#         start_idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
#         end_idxs_to_time = self.reference_time + (end_idxs * self.step).astype('timedelta64[m]')
#         durations = (end_idxs_to_time- start_idxs_to_time).astype('timedelta64[D]')
        
#         self.effects_params["Std_variation"]["channel"].extend(channels)
#         self.effects_params["Std_variation"]["interval"].extend(durations.astype('str'))        
        
        
#         # add it to the channels
#         for i in range(occ):
#             ch = channels[i]
#             amplitude = np.random.uniform(high = max_value, size = (1, intervals[i]))
            
#             self.cov[ch, ch, start_idxs[i]: end_idxs[i]] = amplitude
#             self.effects_params["Std_variation"]["amplitude"].extend(amplitude)
            
#     def add_channels_coupling(self, params):
        
#         # extract parameters:
#         occ = params["occurances"] # number of channels_coupling.
#         max_value = params["coupling_strengh"] # max values of std.
# #         interval = params["interval"] # length of the interval on which the std variates
        
        
#         ### create randomised std variations parameters
#         channels = np.random.randint(self.nchannels, size=(occ, 2)) #  # On which channel will the seasonality be applied.     
# #         start_idxs = np.random.randint(self.n - interval, size=occ) # At which index will the std variation start.
# #         end_idxs = start_idxs + np.random.randint(interval, size=occ) # At which index will the std variation end.
# #         intervals = end_idxs - start_idxs
        
        
#         #save std variations parameters
# #         start_idxs_to_time = self.reference_time + (start_idxs * self.step).astype('timedelta64[m]')
# #         end_idxs_to_time = self.reference_time + (end_idxs * self.step).astype('timedelta64[m]')
# #         durations = (end_idxs_to_time- start_idxs_to_time).astype('timedelta64[D]')
        
#         self.effects_params["Channels_Coupling"]["channels"].extend(channels)
# #         self.effects_params["Std_variation"]["interval"].extend(durations.astype('str'))        
        
        
#         # add it to the channels
#         for i in range(occ):
#             ch = channels[i]
#             amplitude = np.random.uniform(high = max_value)
            
#             self.cov[ch[0], ch[1], :] = amplitude
#             self.effects_params["Channels_Coupling"]["amplitude"].append(amplitude)
            
    
#     def add_noise(self):
        
#         if self.effects is not None:
#             params = self.effects["Noise"]
        
#             if params["occurances"] != 0:
                
#                 # extract parameters:
#                 occ = params["occurances"]  # number of Trends.   
#                 slope = params["max_slope"] # Max slope of the Trends  
#                 noise_type = params["type"] # linear or quadratic or mixed trends


#                 ### create randomised Noise parameters
#                 channels = np.random.randint(self.nchannels, size=occ) # On which channel will the Trend be applied.
#                 idxs = np.random.randint(self.n, size=occ) # At which index will the Trend start.        
#                 slopes = np.random.uniform(low = -slope, high = slope, size=occ)  # Slope of the Trend.


#                 #save the Noise parameters
#                 idxs_to_time = self.reference_time + (idxs * self.step).astype('timedelta64[m]')
#                 self.effects_params["Noise"]["channel"].extend(channels)
#                 self.effects_params["Noise"]["index"].extend(idxs_to_time.astype('str'))
#                 self.effects_params["Noise"]["slope"].extend(slopes)


#                 # generate the trends
#                 noises = np.zeros_like(self.x)
#                 for channel, idx in enumerate(idxs):
#                     shifted = len(self.t) - idx
#                     ch = channels[channel]
#                     if noise_type == "linear":
#                         noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)
#                     elif noise_type == "quadratic":
#                         noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**2
#                     elif noise_type == "mixed":
#                         noises[ch, idx:] += np.linspace(0, slopes[channel] * self.step * shifted, shifted)**((channel%2)+1)

#                 # add it to the channels
#                 self.x += noises
        
        
#     def plot_time_series(self, title): 
        
#         date_array = self.reference_time + np.array(self.t, dtype='timedelta64[m]')
    
#         duration = np.max(date_array) - np.min(date_array)
#         one_month = np.timedelta64(1, 'M').astype('timedelta64[m]')
        
#         formater = mdates.DateFormatter('%A')
#         locator = mdates.DayLocator()
#         if duration > one_month/2 and duration <= 2 * one_month:
#             locator = mdates.DayLocator(interval=2)
#         elif duration > 2 * one_month:
#             locator = mdates.WeekdayLocator()
#             formater = mdates.DateFormatter('%Y-%m-%d')
        
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(date_array, self.x.T)
#         ax.set(xlabel='Date', ylabel='Value', title=title)
        
#         ax.xaxis.set_major_formatter(formater)
#         ax.xaxis.set_major_locator(locator)
        
#         plt.xticks(rotation=45)
#         plt.grid(True)
#         plt.show()