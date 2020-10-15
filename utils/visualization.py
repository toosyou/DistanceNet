import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch
from matplotlib import transforms

def draw(signal1, signal2, attention):
    """
    Args:
        signal1: (Array) with sized [data_length]
        signal2: (Array) with sized [data_length]
        attention: (Array) with sized [data_length, data_length]
    """
    signal_length = signal1.shape[0]
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(signal1)
    ax2.plot(signal2)

    for i in range(signal_length):
        print(i, end='')
        for j in range(signal_length):
            if attention[i][j] < 0.3:
                continue
            point1 = (i,signal1[i])
            point2 = (j, signal2[j])
            
            con = ConnectionPatch(xyA=point1, xyB=point2, coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color='red', alpha=(5 ** attention[i][j] - 1)/4/16)
            ax2.add_artist(con)
    
    return plt

def atten_heatmap(signal1, signal2, attention):
    fig, axes = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 10], 'height_ratios':[1, 10]}, figsize=(20, 20))
    axes[0,0].set_visible(False)

    axes[0, 1].plot(signal1)
    axes[0, 1].axis('off')
    axes[0, 1].margins(0.005)
    base = axes[1,0].transData
    rot = transforms.Affine2D().rotate_deg(90)
    axes[1, 0].plot(np.arange(attention.shape[1]), signal2, transform=rot+base)
    axes[1, 0].axis('off')
    axes[1, 0].margins(0.005)
    sns.heatmap(attention, ax=axes[1,1], cbar=False, xticklabels=False, yticklabels=False,cmap='bwr')
    plt.margins(0)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)