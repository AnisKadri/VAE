import pytest
import pathlib
import sys
parent_dir = pathlib.Path().absolute()
print("{}\dataGen.py".format(parent_dir.parent))
sys.path.append("{}\dataGen.py".format(parent_dir.parent))

from dataGen import Gen


@pytest.fixture
def testGen():
   effects = {
      "Pulse": {
         "occurances": 0,
         "max_amplitude": 1.5,
         "interval": 4000,
         "start": 5000
      },
      "Trend": {
         "occurances": 0,
         "max_slope": 0.0002,
         "type": "linear",
         "start": 0
      },
      "Seasonality": {
         "occurances": 0,
         "frequency_per_week": (7, 14),  # min and max occurances per week
         "amplitude_range": (5, 20),
         "start": -5
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
   n_samples = 3
   periode = 1
   step = 5
   val = 1000
   nchannels = 2
   X = Gen(n_samples = n_samples, periode = periode, step = step, val = val, nchannels = nchannels, effects=effects)
   return X, n_samples, periode, step, val, nchannels, effects