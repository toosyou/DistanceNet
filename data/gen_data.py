import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gen_sig(signal_length=100, gaussian_level=51):
    sig = np.zeros(signal_length)
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
        YL (Array) with sized [num_data, ]
    """
    X = np.zeros((num_data, channel, signal_length))
    Y = np.zeros((num_data, ))
    for i in range(num_data):
        peaks = []
        for j in range(channel):
            sig, peak = gen_sig()
            X[i, j] = sig
            peaks.append(peak)
        Y[i] = peaks[1] - peaks[0]
    return X, Y