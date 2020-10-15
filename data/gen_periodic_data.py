import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from gen_data import DataGenerator

from skimage.util.shape import view_as_windows as viewW

def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    p = np.full((a.shape[0],a.shape[1]-1),0)
    a_ext = np.concatenate((p,a,p),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), -r + (n-1),0]

class PeriodicGenerator():
    def __init__(self, num_data, channel, signal_length, noisy_peak_num, period):
        self.num_data = num_data
        self.channel = channel
        self.signal_length = signal_length
        self.noisy_peak_num = noisy_peak_num
        self.period = period
        self.DG = DataGenerator(self.num_data, self.channel, self.period-100, padding_length=50)
        self.DG.addPeakShape(["triangle", "square"])

    def generate_single(self):
        final_signal = []
        gaussian_peaks = []
        seed = np.random.randint(1000000, size=(self.num_data, self.channel))
        for i in range(self.signal_length // self.period):
            signal, gaussian_peak = self.DG.generate(noisy_peak_num=self.noisy_peak_num, seed=seed)
            signal = signal[0, np.argsort(gaussian_peak, axis=1), :]
            gaussian_peak = np.sort(gaussian_peak, axis=1) + i * self.period

            final_signal.append(signal)
            gaussian_peaks.append(gaussian_peak)
        final_signal.append(np.zeros((1, self.channel, self.signal_length % self.period)))
        final_signal = np.concatenate(final_signal, axis=2)
        gaussian_peaks = np.concatenate(gaussian_peaks, axis=0).transpose()

        return final_signal[0], gaussian_peaks
    
    def generate(self):
        X = np.zeros((self.num_data, self.channel, self.signal_length))
        Y = np.zeros((self.num_data, self.channel, self.signal_length//self.period))
        for i in range(self.num_data):
            X[i], Y[i] = self.generate_single()
        return X, Y
    
    def generate_fast(self, shift=False):
        final_signal = []
        gaussian_peaks = []
        seed = np.random.randint(1000000, size=(self.num_data, self.channel))
        if shift:
            shift_value = np.random.normal(0, 20, size=(self.num_data)).astype(int)
        for i in range(self.signal_length // self.period):
            signal, gaussian_peak = self.DG.generate(noisy_peak_num=self.noisy_peak_num, seed=seed) # (#data, #channel, (signal_length))
            if shift:
                signal[:, 1, :] = strided_indexing_roll(signal[:, 1, :], shift_value)
                gaussian_peak[:, 1] += shift_value
            indices = np.indices((self.num_data, self.channel))
            signal = signal[indices[0], np.argsort(gaussian_peak, axis=1), :]
            gaussian_peak = np.sort(gaussian_peak, axis=1) + i * self.period
            final_signal.append(signal)
            gaussian_peaks.append(gaussian_peak)
        final_signal.append(np.zeros((self.num_data, self.channel, self.signal_length % self.period)))
        final_signal = np.concatenate(final_signal, axis=2)
        gaussian_peaks = np.stack(gaussian_peaks, axis=-1)

        return final_signal, gaussian_peaks


if __name__ == "__main__":
    file = "data/periodic_100000_23.npz"
    if os.path.exists(file):
        print("data already generated.\nTo get the data, using index 'signals' and 'peaks'")
    else:
        PG = PeriodicGenerator(100000, 2, 5000, [2,3], 500)
        signals, peaks = PG.generate_fast(shift=False)
        np.savez(file, signals=signals, peaks=peaks)
        print(f"file {file} saved")