# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:45:30 2024

@author: Wenxuan Zhao, Chris Yang

visualizes the flute_pipeline outputs
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sdt_reader as sdt
import tifffile as tiff
    
# plots irf values against data values
def plot_irfs_against_sdt(irf, shifted_irf, sdt_data):
    sdt_data = np.sum(sdt_data, 1)
    sdt_data = np.sum(sdt_data, 0)

    fig, ax = plt.subplots(ncols=2, figsize=(6, 2))
    ax[0].plot(irf / max(irf), label = "irf")
    ax[0].plot(sdt_data / max(sdt_data), label = "sdt curve")
    ax[0].legend()
    ax[0].set_title("Before Shift") 
    ax[1].plot(shifted_irf/ max(shifted_irf), label = "shifted_irf")
    ax[1].plot(sdt_data / max(sdt_data), label = "sdt curve")
    ax[1].legend()
    ax[1].set_title("After Shift") 
    # plt.title(title)
    return fig
    
# plot a phasor plot
#
# param: coords - iterable collection of (G,S) value in np array form
def plot_phasor(title, coords, names, show = False):
    # frame
    f = 0.080   # laser repetition rate in [GHz]
    # print(len(gs_coords))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 0.55])
    u = np.arange(0, 100, 0.01)
    ax.plot(1/(1+u**2), u/(1+u**2), c='k', picker=True)
    wt = 2*np.pi*f*np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],dtype=float)
    x = 1/(1+wt**2)
    y = wt/(1+wt**2)
    ax.scatter(x, y, s=50, c='k', marker="o", picker=True)
    # ln = ax.scatter(0, 0, s=50, c='r', marker="o", animated=True)  # animated=True tells matplotlib to only draw the artist when we explicitly request it
    
    # labels
    plt.title(title)
    lifetime_labels = ['0.5 ns', '1 ns', '2 ns', '3 ns', '4 ns', '5 ns']
    labels = len(lifetime_labels)
    
    label_coords = list(zip(list(x - 0.02), list(y + 0.03)))[:labels]
    
    for i in range(labels):
        ax.annotate(lifetime_labels[i], xy=label_coords[i], fontsize=10)
    
    ax.set_xlabel("g", fontsize=15, fontweight = "bold")
    ax.set_ylabel("s", fontsize=15, fontweight = "bold")
    ax.text(0.8, 0.5, str(f * 1000) + "MHz", fontsize=15, fontweight = "bold")
    
    # data points
    # if more than one class, set the transparency of the points to 0.5
    alpha = 0.5 if len(set(names)) > 1 else 1
    for i, subcoords in enumerate(coords):
        
        g = [gs[0] for gs in subcoords]
        s = [gs[1] for gs in subcoords]
        
        plt.scatter(g, s, s=3, label=names[i], alpha=alpha)
    
    # plot (if needed) and save
    plt.legend()
    plt.savefig("Outputs/Graphs/" + title + ".png")
    
    if show:
        plt.show()


# intensity graph of individual channel of sdt file
#
# param: file is path of sdt file
def visualize_sdt(file):
    test_data = sdt.read_sdt150(file)
    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    # visualize
    if (test_data.ndim == 3):
        ax.imshow(np.sum(test_data, axis = 2)) 
        ax.set_title("SDT")
       # plt.show()
         
    elif (test_data.ndim == 4):
        for i in range(test_data.shape[0]):
            
            if (np.count_nonzero(test_data[i]) == 0):
                continue
            else: 
                ax.imshow(np.sum(test_data[i], axis = 2))
                ax.set_title("SDT")
                test_data = test_data[i]
                break
    ax.axis('off')
    plt.tight_layout(pad=0)

    return fig, test_data
        
# intensity graph of single channel tiff file
#
# param: file is path of tiff file
def visualize_tiff(tif_path, title, cell_label=-1):
    with tiff.TiffFile(tif_path) as tif:
        tif_array = tif.asarray()
    fig, ax = plt.subplots(figsize=(4,4), dpi=300)
    if (tif_array.ndim == 3):
        tif_array = np.sum(tif_array, 2)
    
    # by default show all cell masks (if the tif is a mask), but user can specify a cell label to show only that cell mask
    if cell_label >= 0:
        tif_array = tif_array == cell_label

    ax.imshow(tif_array) 
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


