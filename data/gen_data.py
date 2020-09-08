import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gen_sig(signal_length=100, gaussian_level=51, peak=None):
    sig = np.zeros(signal_length)
    if peak is not None:
        peaks = np.array([int(peak)])
    else:
        peaks = np.random.randint(0, signal_length, 1)
    sig[peaks] = 1
    window = signal.gaussian(gaussian_level, std=4)
    for i in range(peaks[0] - (gaussian_level // 2), peaks[0] + (gaussian_level // 2) + 1):
        if i >= 0 and i < signal_length:
            sig[i] = window[i - (peaks[0] - (gaussian_level // 2))]
        return sig, peaks

def gen_distance_peak_data(num_data=100000, channel=2, signal_length=100):
    """
    Args:
        num_data: (int) number of data you want to generate
        channel: number of channels per data, now can allow only 2
        signal_length: data length you want to generate
    
    Returns:
        X: (Array) with sized [num_data, channel, signal_length]
        Y: (Array) with sized [num_data, ]
    """
    X = np.zeros((num_data, channel, signal_length))
    Y = np.zeros((num_data, ))
    for i in range(num_data):
        peaks = []
        for j in range(channel):
            sig, peak = gen_sig(signal_length=signal_length)
            X[i, j] = sig
            peaks.append(peak)
        Y[i] = peaks[1] - peaks[0]
    return X, Y

def gen_distance_peak_data_choice(num_data, channel=2, signal_length=100):
    X = np.zeros((num_data, channel, signal_length))
    Y = np.zeros((num_data, ))
    Y_choice = [signal_length * 0.25, signal_length * 0.75, -signal_length * 0.25, -signal_length * 0.75]
    i = 0
    while i < num_data:
        peak_dis = np.random.choice(Y_choice)
        sig_1, peak_1 = gen_sig(signal_length=signal_length)
        next_peak = peak_1 + peak_dis
        if next_peak >= 0 and next_peak < signal_length:
            sig_2, peak_2 = gen_sig(signal_length=signal_length, peak=next_peak)
        else:
            continue
        X[i, 0] = sig_1
        X[i, 1] = sig_2
        Y[i] = peak_dis
        i += 1
    return X, Y