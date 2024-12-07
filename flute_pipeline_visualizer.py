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
    
    # visualize
    if (test_data.ndim == 3):
        plt.imshow(np.sum(test_data, axis = 2)) 
        plt.title("SDT")
        plt.show()
         
    elif (test_data.ndim == 4):
        for i in range(test_data.shape[0]):
            
            if (np.count_nonzero(test_data[i]) == 0):
                continue
            
            print(i)
            plt.imshow(np.sum(test_data[i], axis = 2))
            plt.title("SDT")
            plt.show()
             
        
# intensity graph of single channel tiff file
#
# param: file is path of tiff file
def visualize_tiff(file, title):
    with tiff.TiffFile(file) as tif:
        tif_array = tif.asarray()
    
    if (tif_array.ndim == 3):
        tif_array = np.swapaxes(tif_array, 0,2)
        tif_array = np.swapaxes(tif_array, 0,1)
        tif_array = np.sum(tif_array, 2)
        
    plt.imshow(tif_array) 
    plt.title(title)
    plt.show() 
    
    
# plots irf values against data values
def plot_irf_data(irf, data):
    plt.plot(irf / max(irf), label = "irf")
    plt.plot(data / max(data), label = "data")
    plt.legend()
    # plt.title(title)
    plt.show()


