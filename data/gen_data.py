import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gen_sig(signal_length=100, gaussian_level=51, peak=None):
    """
    Args:
        signal_length: (int) data length you want to generate
        gaussian_level: (int) peak smooth level
        peak: (int) generate specific peak position signal
    Returns:
        sig: (Array) with sized [signal_length, 2]
        peaks: (Array) with sized [1]
    """
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

class DataGenerator():
    """
    generate signals with labels. labels will be two gaussian peaks' distance, and there may be other shape waves generate.

    example usage:
        G = DataGenerator(num_data=1, channel=2, signal_length=1000, padding_length=1000)
        G.addPeakShape(["triangle", "square"])
        X, Y = G.generate(noisy_peak_num=4)
    """
    def __init__(self, num_data, channel, signal_length, padding_length=0):
        """
        Args:
            num_data: (int) number of data you want to generate
            signal_length: (int) signal length you want to generate
            channel: (int) number of channels you want to generate
        """
        self.num_data = num_data
        self.signal_length = signal_length
        self.channel = channel
        self.padding_length = padding_length
        self.peak_shape = []

    def addPeakShape(self, shape):
        """
        Args:
            shape: (list) denote the shape with string, current settings can be triangle and square.
        """
        for i in shape:
            if i in ["triangle", "square"]:
                self.peak_shape.append(i)

    def generate_single(self, peak=None, noisy_peak_num=0):
        """
        generate single signal.

        Args:
            peak: (int) specific peak location (wip)
            noisy_peak_num: (int) number of peaks to generate except gaussian peak
        
        Returns:
            sig: (Array) with sized [signal_length + 2*padding_length]
            peak: (int) gaussian peak location
        """
        rng = np.random.default_rng()
        num_peaks = noisy_peak_num + 1
        peaks_type = rng.choice(self.peak_shape, num_peaks)
        gaussian_peak = rng.integers(num_peaks)
        peaks_type[gaussian_peak] = "gaussian"
        avg_peak_range = self.signal_length // num_peaks
        peaks_position = [avg_peak_range*index + i for index, i in enumerate(rng.integers(0, avg_peak_range, num_peaks))]
        #print(peaks_type)
        #print(peaks_position)

        sig = np.zeros(self.signal_length + 2 * self.padding_length)
        for i in range(len(peaks_type)):
            width = rng.integers(80, 120) if peaks_type[i]=="gaussian" else rng.integers(40, 60)
            if peaks_type[i] == "gaussian":
                gaussian_peak_position = i
                window = self.generate_peak_window(peaks_type[i], width=width, strength=rng.random()*5+22)
                gaussian_window = window
                gaussian_width = width
            else:
                window = self.generate_peak_window(peaks_type[i], width=width, strength=rng.random()+0.5)
            #print(window)
            for index, j in enumerate(range(max(peaks_position[i]+self.padding_length-width, 0), min(peaks_position[i]+self.padding_length+width-1, self.signal_length+2*self.padding_length))):
                #print(index, j)
                sig[j] = max(window[index-min(peaks_position[i]+self.padding_length-width, 0)], sig[j])
        # replace all value within gaussian peaks with real gaussian peaks number
        for index, j in enumerate(range(max(peaks_position[gaussian_peak_position]+self.padding_length-gaussian_width, 0), min(peaks_position[gaussian_peak_position]+self.padding_length+gaussian_width-1, self.signal_length+2*self.padding_length))):
            sig[j] = gaussian_window[index-min(peaks_position[gaussian_peak_position]+self.padding_length-gaussian_width, 0)]

        return sig, peaks_position[gaussian_peak]


    def generate(self, noisy_peak_num=2):
        """
        generate whole dataset.

        Args:
            noisy_peak_num: (int) number of peaks to generate except gaussian peak
        
        Returns:
            X: (Array) with sized [#data, #channel, signal_length + 2*padding_length]
            Y: (Array) with sized [#data] (currently only support 2 channels)
        """
        X = np.zeros((self.num_data, self.channel, self.signal_length+2*self.padding_length))
        Y = np.zeros((self.num_data, ))
        for i in range(self.num_data):
            peaks = []
            for j in range(self.channel):
                sig, peak = self.generate_single(noisy_peak_num=noisy_peak_num)
                X[i, j] = sig
                peaks.append(peak)
            Y[i] = peaks[1] - peaks[0]
        return X, Y

    def generate_peak_window(self, peak_type, width, strength):
        """
        generate a specific type of window.

        Args:
        peak_type: (string) determine peak type, can be triangle, square, or gaussian
        width: (int) peak's half width
        strength: (int) peak strength. (in gaussian is std, others are amplitude)
        """
        if peak_type == "triangle":
            scope = strength / width
            half = np.arange(0, strength, step=scope)
            return np.concatenate((half,half[-2::-1]))
        elif peak_type == "square":
            return np.full((width*2-1), strength)
        elif peak_type == "gaussian":
            return signal.gaussian(width*2-1, std=strength)