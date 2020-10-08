import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class PeriodicGenerator():
    def __init__(self, num_data, channel, signal_length, cycles):
        self.num_data = num_data
        self.channel = channel
        self.signal_length = signal_length
        self.cycles = cycles

    def generate_single(self, peak=None):
        rng = np.random.default_rng()
        