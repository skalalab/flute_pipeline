# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:45:30 2024

@author: Chris Yang

visualizes the flute_pipeline outputs
"""

import tifffile as tiff
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sdt_reader as sdt

def plot_irf_data(irf, data, title):
    plt.plot(irf / max(irf), label = "irf")
    plt.plot(data / max(data), label = "data")
    plt.legend()
    plt.title(title)
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

