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
from scipy import signal
import flute_pipeline_visualizer as visualizer


# This class is the pipeline
class Pipeline:

    # creates all folders needed to run program, also sets time_axis for
    # the pipeline
    def __init__(self, time):
        # invalid time axis
        if (time < 0 or time > 2):
            raise Exception("Time must be from 0-2")
        
        self.time_axis = time
        
        # create all folders needed
        sdt = "SDTs"
        if not os.path.exists(sdt):
            os.mkdir(sdt)
            
        mask = "Masks"
        if not os.path.exists(mask):
            os.mkdir(mask)
        
        tiff_original = "TIFFs/Original"
        if not os.path.exists(tiff_original):
            os.makedirs(tiff_original)
        
        tiff_masked = "TIFFs/Masked"
        if not os.path.exists(tiff_masked):
            os.mkdir(tiff_masked)
            
        irf_txt = "IRFs/txt"
        if not os.path.exists(irf_txt):
            os.makedirs(irf_txt)
            
        irf_tiff = "IRFs/tiff"
        if not os.path.exists(irf_tiff):
            os.mkdir(irf_tiff)

    # create metadata for an image of sdt data
    #
    # param: time_bins - number of time bins
    # return: metadata - dict
    def __generate_metadata(self, time_bins):
        metadata = {"ImageJ": "1.54f", "images": time_bins, "frames": time_bins, 
                            "loop": False}
        
        return metadata 

    # swap time axes of data
    #
    # param: data - 3d array to swap
    # return: time swapped array
    def __swap_time_axis(self, data):
        data = np.swapaxes(data, self.time_axis, 2)
        
        if (self.time_axis == 0):
            data = np.swapaxes(data, 1, 2)
            
        return data
    
    # zero padding shift
    #
    # param: shift - amount to shift
    # param: array - array to be shifted
    # return: shifted array
    def __shift(self, array, shift):
        if (shift > 0):
            shifted = np.concatenate([np.zeros(shift), array[:-shift]])
        elif (shift < 0):
            shifted = np.concatenate([array[shift:], np.zeros(-shift)])
        else:
            shifted = array
            
        return shifted
    
    # shift irf by findning where peaks align with data
    #
    # param: irf - array to shift
    # param: data - array to be compare irf with
    # param: return_array - return array or shift value
    # return: shift if specifeid
    # return: shifted irf if specified
    def __correlate_max_peak(self, irf, data, return_array = False):
        # get max indexes
        irf_max_index = np.where(irf == max(irf))[0][0]
        data_max_index = np.where(data == max(data))[0][0]
        
        print(irf_max_index)
        print(data_max_index)
        
        # shift
        shift = data_max_index - irf_max_index
        
        if return_array is True:
            return self.__shift(irf, shift)
        
        return shift
            
    
    # generate max correlation shifted IRF for .tif image
    #
    # param: file_name - txt file path
    # param: image_channel - the channel of .sdt that holds the nonempty image 
    # param: image_data - the data of the .sdt
    def __generate_irf(self, file_name, image_channel, image_data):
        # get the irf values from .txt file
        if image_channel == 0:
            with open("IRFs/txt/Ch1_IRF_890.txt") as irf:
                irf_values = [int(line) for line in irf if line != "\n"]
        else:
            with open("IRFs/txt/Ch2_IRF_750.txt") as irf:
                irf_values = [int(line) for line in irf if line != "\n"]
                
        if len(irf_values) != image_data.shape[2]:
            raise Exception("Number of IRF values don't match time bins")
        
        time_bins = len(irf_values)
        
        # sum 3d image data into just time bins
        data_values = np.sum(image_data, 1)
        data_values = np.sum(data_values, 0)
                
        # Interpolate 9 values bewteen each time bin (linear) of values
        x_scale_factor = 10
        scaled_bins = ((time_bins - 1) * x_scale_factor) + 1 # +1 to account for range()
        
        interp_irf_values = np.interp(range(scaled_bins), range(0, scaled_bins, x_scale_factor), irf_values)
        interp_data_values = np.interp(range(scaled_bins), range(0, scaled_bins, x_scale_factor), data_values)

        # Cross correlate them
        corr_result = signal.correlate(interp_irf_values, interp_data_values)
      
        # shift found by taking middle index of cross correlation array
        # and subtracing index of peak correlation
        shift = (scaled_bins - 1) - np.where(corr_result == max(corr_result))[0][0]
        
        # shift and unscale irf values
        shifted_irf_values = self.__shift(interp_irf_values, shift)
        peak_shift = self.__correlate_max_peak(interp_irf_values, interp_data_values, return_array = True)
 
        visualizer.plot_irf_data(interp_irf_values, interp_data_values, "before")
        visualizer.plot_irf_data(shifted_irf_values, interp_data_values, "correlate: after")
        visualizer.plot_irf_data(peak_shift, interp_data_values, "peak: after")
    
        final_irf_values = [shifted_irf_values[i] for i in range(0, scaled_bins, x_scale_factor)]
        
        # create shifted irf tiff        
        irf_array = np.empty((image_data.shape[0], image_data.shape[1], time_bins), dtype=np.float32)
        
        for row in range(irf_array.shape[0]):
            for col in range(irf_array.shape[1]):
                np.put(irf_array[row][col], range(time_bins), final_irf_values)
                
        tiff.imwrite("IRFs/tiff/" + file_name + "irf" + ".tif", self.__swap_time_axis(irf_array))
                    
                    
    # mask entire image. Also masks each individual cell. Saves all as
    # tiff files
    #
    # param: path - the mask .tiff file
    def mask_image(self, path):
        # create folder for all masked image and cells
        image_name = "_".join(path.name[:path.name.index(".tif")].split("_")[:-2])
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
                break
            
        # generate metadata
        metadata = self.__generate_metadata(sdt_data.shape[2])
                    
        # save tif of original image
        tiff.imwrite("TIFFs/Original/" + image_name +".tif", 
                     self.__swap_time_axis(sdt_data), metadata=metadata)
        
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
        
        self.__generate_irf(file_name, nonempty_channel, masked_image)
        
        tiff.imwrite(masked_folder_path + file_name + ".tif", 
                     self.__swap_time_axis(masked_image), metadata=metadata)
        
        # split masked image into single cells
        cell_values= np.unique(mask)
        cell_values = cell_values[1:]

        
        # create image for every single cell
        for i in range(len(cell_values)):
            # mask each cell
            cell_image = np.copy(masked_image)
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row][col] != cell_values[i]:
                        cell_image[row][col][:] = 0
                    
            # save cell
            file_path = image_name + "cell_" + str(cell_values[i])
            
            tiff.imwrite(masked_folder_path + file_path + ".tif", 
                          self.__swap_time_axis(cell_image), metadata=metadata)
            
            
            





