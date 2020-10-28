import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch
from matplotlib import transforms

def atten_pair(signal1, signal2, attention):
    """
    Args:
        signal1: (Array) with sized [data_length]
        signal2: (Array) with sized [data_length]
        attention: (Array) with sized [data_length, data_length]
    """
    plt.clf()
    signal_length = signal1.shape[0]
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(signal1)
    ax1.plot(signal2)
    ax2.plot(signal2)
    ax2.plot(signal1)

    for i in range(signal_length):
        for j in range(signal_length):
            if attention[i][j] < 0.2:
                continue
            point1 = (i, signal1[i])
            point2 = (j, signal2[j])
            point3 = (i, signal2[i])
            point4 = (j, signal1[j])
            
            pair = [[point1, point2],[point2, point3], [point3, point4], [point4, point1]]
            color = ['#ff0000', '#00ff00', '#0000ff', '#87cefa']

            for m in range(4):
                con = ConnectionPatch(xyA=pair[m][0], xyB=pair[m][1], coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color=color[m], alpha=(5 ** attention[i][j] - 1)/4)
                ax2.add_artist(con)
    
    return plt

def atten_heatmap(signal1, signal2, attention, figsize=(20, 20)):
    plt.clf()
    fig, axes = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 10], 'height_ratios':[1, 10]}, figsize=figsize)
    axes[0,0].set_visible(False)
    #axes[0,1].set_visible(False)
    #axes[1,0].set_visible(False)
    #axes[1,1].set_visible(False)

    axes[0, 1].plot(signal1, alpha=0.5)
    axes[0, 1].plot(signal2, alpha=0.5)
    axes[0, 1].axis('off')
    axes[0, 1].margins(0.005)    
    #axes[1, 2].plot(signal2)
    #axes[1, 2].axis('off')
    #axes[1, 2].margins(0.005)

    rot = transforms.Affine2D().rotate_deg(270)

    base = axes[1,0].transData
    axes[1, 0].plot(signal1, transform=rot+base, alpha=0.5)
    axes[1, 0].plot(signal2, transform=rot+base, alpha=0.5)
    axes[1, 0].axis('off')
    axes[1, 0].margins(0.005)
    #base = axes[2,1].transData
    #axes[2, 1].plot(np.arange(attention.shape[1]), signal1, transform=rot+base)
    #axes[2, 1].axis('off')
    #axes[2, 1].margins(0.005)

    sns.heatmap(attention, ax=axes[1,1], cbar=False, xticklabels=False, yticklabels=False,cmap='Greens', robust=False)
    plt.margins(0)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    return plt