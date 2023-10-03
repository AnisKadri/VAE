import pytest
from unittest.mock import MagicMock
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
# from dataGen import Gen

def test_init(testGen):
    X, n_samples, periode, step, val, nchannels, effects = testGen
    # X = Gen(n_samples=n_samples, periode=periode, step=step, val=val, nchannels=nchannels, effects=effects)
    data, params, e_params = X.parameters()
    min_per_day = 1440
    periode = periode * min_per_day
    n = periode // step

    X.add_effects = MagicMock()
    X.sample = MagicMock()

    assert X.mu.ndim == 2
    assert X.mu.shape[0] == nchannels * n_samples
    assert X.mu.shape[1] == n

    assert X.cov.ndim == 2
    assert X.cov.shape[0] == nchannels * n_samples
    assert X.cov.shape[1] == n

    # assert X.add_effects.called
    # assert X.sample.called

def test_datashape(testGen):
    X, n_samples, periode, step, val, nchannels, effects = testGen
    data, params, e_params = X.parameters()
    data_shape = data.shape
    min_per_day = 1440
    periode = periode * min_per_day
    assert data_shape[0] == n_samples
    assert data_shape[1] == nchannels
    assert data_shape[2] == periode // step

def test_add_effects_calls(testGen):
    X, n_samples, periode, step, val, nchannels, effects = testGen
    data, params, e_params = X.parameters()

    X.add_pulse = MagicMock()
    X.add_trend = MagicMock()
    X.add_seasonality = MagicMock()
    X.add_std_variation = MagicMock()
    X.add_channels_coupling = MagicMock()

    X.add_effects(effects)


    assert not X.add_pulse.called
    assert not X.add_trend.called
    assert not X.add_seasonality.called
    assert not X.add_std_variation.called
    assert not X.add_channels_coupling.called

@pytest.mark.parametrize("test_start, test_occ, test_interval, results", [(-1, 2, 0, 0), (0, 2, 0, 0), (0.5, 2, 0, 288 //2), (1, 2, 0, 288), (15, 2, 0, 288)])
def test_set_start_idxs(testGen, test_start, test_occ, test_interval, results):
    X, n_samples, periode, step, val, nchannels, effects = testGen
    data, params, e_params = X.parameters()

    start_idx = X.set_start_idxs(test_start, test_occ, test_interval)
    max_length = X.n - test_interval
    result_array = np.repeat(results, n_samples * test_occ)
    assert len(start_idx) == n_samples * test_occ
    assert (start_idx >= 0).all()
    assert (start_idx <= max_length).all()
    assert (start_idx == result_array).all()

@pytest.mark.parametrize("test_interval, test_treshhold, result", [(-1, 10, 0), (0, 20, 0), (10, 4, 10), (1000, 0, 288)])
def test_check_interval_length(testGen, test_interval, test_treshhold, result):
    X, n_samples, periode, step, val, nchannels, effects = testGen
    data, params, e_params = X.parameters()
    interval = X.check_interval_length(test_interval, test_treshhold)
    assert interval == result
@pytest.mark.parametrize("pulse",[({
        "occurances":1,
        "max_amplitude":1.5,
        "interval":4000,
        "start":-1
        }), ({
        "occurances":1,
        "max_amplitude":1.5,
        "interval":4000,
        "start":15
        }), ({
        "occurances":1,
        "max_amplitude":1.5,
        "interval":4000,
        "start":0
        }), ({
        "occurances":1,
        "max_amplitude":1.5,
        "interval":3,
        "start":0.3
        })])
def test_add_pulse(testGen, pulse):
    X, n_samples, periode, step, val, nchannels, effects = testGen
    data, params, e_params = X.parameters()

    mu_old = np.copy(X.mu)
    X.add_pulse(pulse)
    mu_new = X.mu

    channel = np.array(e_params["Pulse"]["channel"]).squeeze()
    indexes = np.array(e_params["Pulse"]["index"]).squeeze()
    amps = np.array(e_params["Pulse"]["amplitude"]).squeeze()


    affected_channels, starts, ends = get_start_end_indexs(mu_new, mu_old)

    for ch, start, end, amp in zip(affected_channels, starts, ends, amps):
        assert (mu_old[ch, start:end] != mu_new[ch, start:end]).all()

    assert (indexes == list(starts)).all()
    assert (mu_old != mu_new).any()
    assert (channel >= 0).all()
    assert (channel < nchannels).all()

def get_start_end_indexs(mu_new, mu_old):
    rows, cols = np.where(mu_old != mu_new)
    # print(rows, cols)
    unique_indices, counts = np.unique(rows, return_index=True)
    starts = cols[counts]
    ends = np.diff(np.append(counts, len(cols)))
    return unique_indices, starts, ends


def array_to_list(array_of_arrays):
    string_list = [str(item[0]) for item in array_of_arrays]
    return string_list

@pytest.mark.parametrize("trend",[({
        "occurances":2,
        "max_slope":1.5,
        "type":"linear",
        "start":0.5
        }), ({
        "occurances":2,
        "max_slope":1.5,
        "type":"quadratic",
        "start":0.5
        }), ({
        "occurances":2,
        "max_slope":1.5,
        "type":"mixed",
        "start":0.5
        })])
def test_add_trend(testGen, trend):
    X, n_samples, periode, step, val, nchannels, effects = testGen
    data, params, e_params = X.parameters()

    print(data.shape)
    mu_old = np.copy(X.mu)
    X.add_trend(trend)
    mu_new = X.mu

    channel = np.array(e_params["Trend"]["channel"]).squeeze()
    indexes = np.array(e_params["Trend"]["index"][-1]).squeeze()
    slopes = np.array(e_params["Trend"]["slope"]).squeeze()

    plt.plot(mu_old.T, "r")
    plt.plot(mu_new.T, "b")
    plt.show()


    affected_channels, starts, ends = get_start_end_indexs(mu_new, mu_old)
    print("\nsaved: ", e_params)
    print("Starts: ", starts)
    print("indexes: ", indexes)

    for ch, start, end, amp in zip(affected_channels, starts, ends, slopes):
        assert (mu_old[ch, start:] != mu_new[ch, start:]).all()

    assert (indexes == list(starts)).all()
    assert (mu_old != mu_new).any()
    assert (channel >= 0).all()
    assert (channel < nchannels).all()