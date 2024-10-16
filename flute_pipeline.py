# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:50:00 2024

@author: Chris Yang

Takes FLIM data as .sdt files, converts the .sdt files to single channel .tif
files, masks the .tif files, then masks every single cell for the .tif files.
All masked images/cells are saved as .tif files.
"""

import tifffile as tiff
import numpy as np
from pathlib import Path
import os
import sdt_reader as sdt
import matplotlib.pyplot as plt

class Pipeline:

    def __init__(self, time):
        self.time_axis = time
        
        # create all folders needed
        single_channel_folder = "TIFFs/Original"
        
        if not os.path.exists(single_channel_folder):
            os.makedirs(single_channel_folder)
        
        masked_folder = "TIFFs/Masked"
        
        if not os.path.exists(masked_folder):
            os.mkdir(masked_folder)
                
        # TODO: form actual metadata
        # get .tif metadata for imagej
        self.imagej_meta = {"ImageJ": "1.54f", "images": 256, "frames": 256, 
                            "loop": False}


    # correct orientation and swap time axes of data
    #
    # param: data - 3d array to be correct
    def __swap_time_axis(self, data):
        data = np.swapaxes(data, self.time_axis, 2)
        
        if (self.time_axis == 0):
            data = np.swapaxes(data, 1, 2)
            
        return data
    
    # generate max correlation shifted IRF for .tif image
    #
    # param: file - txt file path
    # param: rows - number of rows 
    # param: cols - number of columns
    def generate_irf(self, file_name, image_channel, image_data):
        # get the irf values from .txt file
        if image_channel == 0:
            with open("IRFs/txt/Ch1_IRF_890.txt") as irf:
                irf_values = [int(line) for line in irf if line != "\n"]
        else:
            with open("IRFs/txt/Ch2_IRF_750.txt") as irf:
                irf_values = [int(line) for line in irf if line != "\n"]
                
        if len(irf_values) != image_data.shape[2]:
            raise Exception("Number of IRF values don't match time bins")
        
        # sum 3d image data into 256 time bins
        data_values = np.sum(image_data, 0)
        data_values = list(np.sum(data_values, 0))
        plt.plot(data_values)
        plt.show()
        print(len(data_values))
        
        plt.plot(irf_values)
        plt.show()
        print(len(irf_values))
        
        irf_array = np.empty((image_data.shape[0], image_data.shape[1], len(irf_values)), dtype=np.float32)
        
        # for row in range(irf_array.shape[0]):
        #     for col in range(irf_array.shape[1]):
        #         np.put(irf_array[row][col], range(len(irf_values)), irf_values)
                
        # tiff.imwrite("IRFs/tiff/" + file_name + "irf" + ".tif", self.__swap_time_axis(irf_array))
                    
                    
    # mask entire image. Also masks each individual cell. Saves all as
    # tiff files
    def mask_image(self, path):
        # create folder for all masked image and cells
        image_name = path.name[:-22]
        masked_folder_path = "TIFFs/Masked/" + image_name + "/"
        
        if not os.path.exists(masked_folder_path):
            os.mkdir(masked_folder_path)
        
        # get image from corresponding .sdt file
        sdt_path = Path("SDTs/" + image_name + ".sdt")
        sdt_data = sdt.read_sdt150(sdt_path)
             
        # remove empty channel if needed
        nonempty_channel = 0
        if (sdt_data.ndim == 4):
            for i in range(sdt_data.shape[0]):
                if (np.count_nonzero(sdt_data[i]) == 0):
                    continue
                
                sdt_data = sdt_data[i]
                nonempty_channel = i
                print(nonempty_channel)
                break
                    
        # save tif of original image
        tiff.imwrite("TIFFs/Original/" + image_name +".tif", 
                     self.__swap_time_axis(sdt_data), metadata=self.imagej_meta)
        
        # now mask the image
        with tiff.TiffFile(path) as mask_tif:
            mask = mask_tif.asarray()
            
        masked_image = np.copy(sdt_data)
                    
        # apply mask to tif!
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row][col] == 0:
                    masked_image[row][col][:] = 0
             
        # save and make irf of masked image
        file_name = image_name + "masked_image"
        
        self.generate_irf(file_name, nonempty_channel, masked_image)
        
        tiff.imwrite(masked_folder_path + file_name + ".tif", 
                     self.__swap_time_axis(masked_image), metadata=self.imagej_meta)
        
        # # split masked image into single cells
        # cell_values= np.unique(mask)
        # cell_values = cell_values[1:]

        
        # # create image for every single cell
        # for i in range(len(cell_values)):
        #     # mask each cell
        #     cell_image = np.copy(masked_image)
        #     for row in range(mask.shape[0]):
        #         for col in range(mask.shape[1]):
        #             if mask[row][col] != cell_values[i]:
        #                 cell_image[row][col][:] = 0
                    
        #     # save cell
        #     file_path = image_name + "cell_" + str(cell_values[i])
            
        #     tiff.imwrite(masked_folder_path + file_path + ".tif", 
        #                  self.__swap_time_axis(cell_image), metadata=self.imagej_meta)
            
            
            





