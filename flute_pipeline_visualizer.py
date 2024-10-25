# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:45:30 2024

@author: Wenxuan Zhao, Chris Yang

visualizes the flute_pipeline outputs
"""

import tifffile as tiff
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sdt_reader as sdt

# plots irf values against data values
def plot_irf_data(irf, data, title):
    plt.plot(irf / max(irf), label = "irf")
    plt.plot(data / max(data), label = "data")
    plt.legend()
    plt.title(title)
    plt.show()
    
# plot a phasor plot
#
# param: gs_coords - iterable collection of (G,S) value in np array form
def plot_phasor(gs_coords):
    # frame
    f = 0.050   # laser repetition rate in [GHz]
    w = 2*np.pi*f
    bin_width = 100

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 0.55])
    u = np.arange(0, 100, 0.01)
    ax.plot(1/(1+u**2), u/(1+u**2), c='k', picker=True)
    wt = 2*np.pi*f*np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],dtype=float)
    ax.scatter(1/(1+wt**2), wt/(1+wt**2), s=50, c='k', marker="o", picker=True)
    # ln = ax.scatter(0, 0, s=50, c='r', marker="o", animated=True)  # animated=True tells matplotlib to only draw the artist when we explicitly request it
    # Lifetime labels
    ax.annotate('0.5 ns', xy=(0.985, 0.16), fontsize=6)
    ax.annotate('1 ns', xy=(0.92, 0.30), fontsize=6)
    ax.annotate('2 ns', xy=(0.72, 0.47), fontsize=6)
    ax.annotate('3 ns', xy=(0.52, 0.515), fontsize=6)
    ax.annotate('4 ns', xy=(0.372, 0.505), fontsize=6)
    ax.annotate('5 ns', xy=(0.27, 0.475), fontsize=6)
    
    # datat points
    g = [gs[0] for gs in gs_coords]
    s = [gs[1] for gs in gs_coords]
    
    plt.scatter(g, s)
    
    plt.show()


# # intensity graph of individual channel of sdt file
# #
# # param: file is path of sdt file
# def visualize_sdt(self, file):
#     test_data = sdt.read_sdt150(file)
    
#     # visualize
#     if (test_data.ndim == 3):
#         plt.imshow(np.sum(test_data, axis = 2)) 
#         plt.title("SDT")
#         plt.show()
         
#     elif (test_data.ndim == 4):
#         for i in range(test_data.shape[0]):
            
#             if (np.count_nonzero(test_data[i]) == 0):
#                 continue
            
#             plt.imshow(np.sum(test_data[i], axis = 2))
#             plt.title("SDT")
#             plt.show()
             
        
# # intensity graph of single channel tiff file
# #
# # param: file is path of tiff file
# def visualize_tiff(self, file, title):
#     with tiff.TiffFile(file) as tif:
#         tif_array = tif.asarray()
        
#         if (tif_array.ndim == 3):
#             tif_array = np.sum(tif_array, axis = self.time_axis)
            
#         plt.imshow(tif_array) 
#         plt.title(title)
#         plt.show() 


# # visualize cells and masked image of image
# def visualize_masked(self, masked_folder_path):
#     for image in Path(masked_folder_path).iterdir():
#         self.visualize_tiff(image, image.name)

        
# # visualize all sdt files
# def visualize_all_sdt(self):
#     sdt_paths = [path for path in Path("SDTs").iterdir()]
#     for path in sdt_paths:
#         self.visualize_sdt(path)

# # visualize all original tif
# def visualize_all_original_tiff(self):
#     tiff_paths = [path for path in Path("TIFFs/Original").iterdir()]
#     for path in tiff_paths:
#         self.visualize_tiff(path, "TIFF")

