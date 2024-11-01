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
import sdt_reader
from scipy import signal
import flute_pipeline_visualizer as visualizer

# This class is the pipeline
class Pipeline:
    
    # creates all folders needed to run program
    def __init__(self):        
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
    # param: time_bins - number of time bins, greater than 0
    # return: metadata - dict
    def __generate_metadata(self, time_bins):
        if time_bins < 1:
            raise Exception("time_bins must be above 0")
        
        metadata = {"ImageJ": "1.54f", "images": time_bins, "frames": time_bins, 
                            "loop": False}
        
        return metadata 

    # swap time axes of data to 0 
    #
    # param: data - (rows, cols, time) array
    # return: time swapped view of array
    def __swap_time_axis(self, data):
        if data.ndim > 3:
            raise Exception("data must be 3D")
        
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 1, 2)
            
        return data
    
    # zero padding shift
    #
    # param: shift - amount to shift
    # param: array - array to be shifted
    # return: shifted array
    def __shift(self, array, shift):
        data_type = array.dtype
        
        if (shift > 0):
            shifted = np.concatenate([np.zeros(shift), array[:-shift]])
        elif (shift < 0):
            shifted = np.concatenate([array[-shift:], np.zeros(-shift)])
        else:
            shifted = np.copy(array)
            
        return shifted.astype(data_type)
          
    
    # shift IRF to max correlation with data. Then, make .tif of the IRF
    #
    # param: irf_path - path of irf to shift
    # param: image_name - name of image of data
    # param: image_data - the data of the .sdt
    # return: the shifted irf values
    def __generate_irf(self, irf_path, image_name, image_data):
        # get the irf values from .txt file 
        with open(irf_path) as irf:
            irf_values = [int(line) for line in irf if line.strip()]
           
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
        print(shift)
        
        # shift and unscale irf values
        shifted_irf_values = self.__shift(interp_irf_values, shift)
        final_irf_values = [shifted_irf_values[i] for i in range(0, scaled_bins, x_scale_factor)]
            
        # create shifted irf tiff        
        irf_array = np.empty(image_data.shape, dtype=np.float32)
        
        for row in range(irf_array.shape[0]):
            for col in range(irf_array.shape[1]):
                np.put(irf_array[row][col], range(time_bins), final_irf_values)
                
        tiff.imwrite("IRFs/tiff/" + image_name + "irf" + ".tif", self.__swap_time_axis(irf_array))
        
        # return the shifted irf values
        return np.array(final_irf_values, np.float32)
                    
                    
    # mask entire sdt image and split into cells. Saves all as .tif files
    # Creates a shifted irf .tif for the sdt image also
    #
    # param: sdt - the path for the sdt to be masked
    # return: {"cells": [cell_data], "IRF_decay", [shifted_irf_values]}
    def mask_image(self, sdt):
        # create folders for all masked image and cells
        image_name = sdt.name[:sdt.name.find(".sdt")]
        
        image_folder_path = "TIFFs/Masked/" + image_name + "/Image/"
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)
            
        cell_folder_path = "TIFFs/Masked/" + image_name + "/Cell/"
        if not os.path.exists(cell_folder_path):
            os.mkdir(cell_folder_path)
        
        # get image datafrom sdt
        sdt_data = sdt_reader.read_sdt150(sdt)
             
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
                    
        # make shifted irf tif
        if nonempty_channel == 0:
            irf_path = "IRFs/txt/Ch1_IRF_890.txt"
        else:
            irf_path = "IRFs/txt/Ch2_IRF_750.txt"
        
        IRF_decay = self.__generate_irf(irf_path, image_name, sdt_data)
        
        # save tif of original image 
        tiff.imwrite("TIFFs/Original/" + image_name +".tif", 
                     self.__swap_time_axis(sdt_data), metadata=metadata)
        
        # get the mask of the sdt
        for mask in Path("Masks").iterdir():
            if image_name in mask.name or image_name[:image_name.find("_summed")] in mask.name:
                mask_path = mask
                break
        
        # now mask the image
        with tiff.TiffFile(mask_path) as mask_tif:
            mask = mask_tif.asarray()
            
        masked_image = np.copy(sdt_data)
                    
        # apply mask to tif!
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row][col] == 0:
                    masked_image[row][col][:] = 0
             
        # save masked image
        file_name = image_name + "masked_image"
        
        tiff.imwrite(image_folder_path + file_name + ".tif", 
                     self.__swap_time_axis(masked_image), metadata=metadata)
        
        # split masked image into single cells
        cell_values= np.unique(mask)
        cell_values = cell_values[1:]

        # create image for every single cell
        cell_images = list()
        for i in range(len(cell_values)):
            # mask each cell
            cell_image = np.copy(masked_image)
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row][col] != cell_values[i]:
                        cell_image[row][col][:] = 0
                    
            # save cell
            file_path = image_name + "cell_" + str(cell_values[i])
            cell_images.append(cell_image)
            tiff.imwrite(cell_folder_path + file_path + ".tif", 
                          self.__swap_time_axis(cell_image), metadata=metadata)
            
            
        # return cell_images and IRF_decay as tuple
        return {"cells": cell_images, "IRF_decay": IRF_decay}      
              
    # calculate the (G,S) coordinates of pixel
    #
    # param: cell-hist - summed time bins of cell
    # param: IRF_decay - shifted IRF values
    # return: gs coordinate as np array
    def __get_GS(self, cell_hist, IRF_decay):
        f = 0.080   # laser repetition rate in [GHz]
        w = 2*np.pi*f
        time_axis = np.arange(0, 1/f, 1/f/256)
        G_IRF = np.dot(np.transpose(IRF_decay) , np.cos(w*time_axis)) / np.sum(IRF_decay)
        S_IRF = np.dot(np.transpose(IRF_decay) , np.sin(w*time_axis)) / np.sum(IRF_decay)
        cos_coeff = np.cos(w*time_axis)
        sin_coeff = np.sin(w*time_axis)
        
        # corrected coefficients
        corrected_cos_coeff = (G_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (S_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff
        corrected_sin_coeff = (-S_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (G_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff
        
        # cell_hist is the summed histograms of a cell: it can also be a pixel
        cell_hist_sum = np.sum(cell_hist)
        G_decay = np.dot(cell_hist, corrected_cos_coeff) / cell_hist_sum
        S_decay = np.dot(cell_hist, corrected_sin_coeff) / cell_hist_sum
        
        return np.array([G_decay, S_decay])


    # plots cell level phasor of one or more images
    #
    # param: images - list of of {"cells": cells, "IRF_decay":, IRF_decay} dicts
    def plot_cell_phasor(self, images):
        # get cell (G,S) for each cell of image
        coords = list()
        for image in images:
            for cell in image["cells"]:
                # get cell_hist
                cell_hist = np.sum(cell, 0)
                cell_hist = np.sum(cell_hist, 0)
            
                coords.append(self.__get_GS(cell_hist, image["IRF_decay"]))
            
        # plot
        visualizer.plot_phasor(coords)    


    # testing purposese only
    def plot_pixel_phasor(self, image, IRF_decay):
        coords = list()
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if np.count_nonzero(image[row,col]) != 0:
                    coords.append(self.__get_GS(image[row,col], IRF_decay))
                
        visualizer.plot_phasor(coords)
            
        
    # # testing purposes only
    # def __correlate_max_peak(self, irf, data, return_array = False):
    #     # get max indexes
    #     irf_max_index = np.where(irf == max(irf))[0][0]
    #     data_max_index = np.where(data == max(data))[0][0]
        
    #     # shift
    #     shift = data_max_index - irf_max_index
        
    #     if return_array is True:
    #         return self.__shift(irf, shift)
        
    #     return shift

    # # plots cell level phasor
    # #
    # # param: cells - iterable collection of 3D cell array
    # def plot_cell_phasor(self, cells, IRF_decay):
    #     # get cell (G,S) for each cell
    #     coords = list()
    #     for cell in cells:
    #         # get cell_hist
    #         cell_hist = np.sum(cell, 0)
    #         cell_hist = np.sum(cell_hist, 0)
            
    #         coords.append(self.__get_GS(cell_hist, IRF_decay))
            
    #     # plot
    #     visualizer.plot_phasor(coords)  

