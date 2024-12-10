# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:50:00 2024

@author: Chris Yang and Wenxuan Zhao

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
    
    def __init__(self):
        # craft some directories
        output_graph = "Outputs/Graphs"
        if not os.path.exists(output_graph):
            os.makedirs(output_graph)
    
    
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
          
    # write csv of irf values
    #
    # param: irf_values - list of irf values to write
    # parm: file_path - file path/name
    def __irf_csv(self, irf_values, file_path):
        with open(file_path, "w") as csv:
            csv.write("value,\n")
            for value in irf_values:
                csv.write(str(value) + ",\n")
    
    
    # shift IRF to max correlation with data. Then, make .tif of the IRF
    #
    # param: irf_path - path of irf to shift
    # param: image_name - name of image of data
    # param: image_data - the data of the .sdt
    # return: the shifted irf values
    def __generate_irf(self, irf_path, image_name, image_data, output_path, channel, summed = False, ui_mode = False):
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
        
        # shift and unscale irf values
        shifted_irf_values = self.__shift(interp_irf_values, shift)
        final_irf_values = [shifted_irf_values[i] for i in range(0, scaled_bins, x_scale_factor)]
            
        # create shifted irf array       
        irf_array = np.empty(image_data.shape, dtype=np.float32)
        
        for row in range(irf_array.shape[0]):
            for col in range(irf_array.shape[1]):
                np.put(irf_array[row][col], range(time_bins), final_irf_values)
                
        # save irf as tif 
        # visualizer.plot_irf_data(final_irf_values, data_values)
        if not ui_mode:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            if not summed:
                tiff.imwrite(output_path + image_name + "_channel" + str(channel) + "_irf.tif", self.__swap_time_axis(irf_array))
            if summed:
                tiff.imwrite(output_path + image_name + "_channel" + str(channel) + "_summed_irf.tif", self.__swap_time_axis(irf_array))
            
            # save irf values as csv
            if not summed:
                self.__irf_csv(final_irf_values, output_path + image_name + "_channel" + str(channel) + "_irf.csv")
            if summed:
                self.__irf_csv(final_irf_values, output_path + image_name + "_channel" + str(channel) + "_summed_irf.csv")
                
        
        # return the shifted irf values
        return np.array(final_irf_values, np.float32)
                    
                    
    # mask entire sdt image and split into cells. Saves all as .tif files
    # Creates a shifted irf .tif for the sdt image also
    # Finally creates a summed roi tiff and makes shifted irf for that too
    #
    # param: sdt - the path for the sdt to be masked
    # param: irf - path of txt file of irf
    # param: masks_path - path of where all masks are located
    # return: {"cells": [cell_data], "IRF_decay", [shifted_irf_values]}
    def mask_image(self, sdt, irf, masks_path, channel, ui_mode = False):
        # get image datafrom sdt
        sdt_data = sdt_reader.read_sdt150(sdt)
             
        # find nonempty channel if needed
        if channel == -1:
            for i in range(sdt_data.shape[0]):
                if (np.count_nonzero(sdt_data[i]) == 0):
                    continue
                
                sdt_data = sdt_data[i]
                channel = i
                break
        else:
            sdt_data = sdt_data[channel]
            
        # create folder for output
        image_name = sdt.name[:sdt.name.find(".sdt")]
        
        output_path = "Outputs/" + image_name + "/channel" + str(channel) + "/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)    
        
        # generate metadata
        metadata = self.__generate_metadata(sdt_data.shape[2])
                    
        # make shifted irf tif
        IRF_decay = self.__generate_irf(irf, image_name, sdt_data, output_path, channel)
        
        # get the mask of the sdt
        if os.path.isdir(masks_path): 
            for mask in masks_path.iterdir():
                if image_name in mask.name and ".tif" in mask.name:
                    mask_path = mask
                    break
        else:
            mask_path = masks_path
        
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
        if not ui_mode:
            tiff.imwrite(output_path + image_name + "_channel" + str(channel) + "_masked_image.tif", 
                         self.__swap_time_axis(masked_image), metadata=metadata)
        
        # get cell values
        cell_values= np.unique(mask)
        cell_values = cell_values[1:]

        # single out each cell
        cell_images = list()
        cell_sums = list()
        for i in range(len(cell_values)):
            # mask each cell
            cell_image = np.copy(masked_image)
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row][col] != cell_values[i]:
                        cell_image[row][col][:] = 0
                    
            cell_images.append(cell_image)    
            
            # sum each cell
            cell_hist = np.sum(cell_image, 0)
            cell_hist = np.sum(cell_hist, 0)
            cell_sums.append(cell_hist)
           
        # make summed roi
        if not ui_mode:
            summed_image = np.copy(masked_image)
            for i in range(len(cell_values)):
                for row in range(mask.shape[0]):
                    for col in range(mask.shape[1]):
                        if mask[row][col] == cell_values[i]:
                            np.put(summed_image[row][col], range(summed_image.shape[2]), cell_sums[i])
            
            tiff.imwrite(output_path + image_name + "_channel" + str(channel) + "_summed_image.tif",
                         self.__swap_time_axis(summed_image), metadata = metadata)
            
            # make summed irf
            summed_IRF_decay = self.__generate_irf(irf, image_name, summed_image, output_path, channel, summed=True)

        # return cell_images, cell value, IRF_decay, and output path as tuple
        return {"name": image_name, "channel": channel, "cells": cell_images, "values": cell_values, "IRF_decay": IRF_decay, "Summed_decay": summed_IRF_decay}      
        
    
    # calculate the (G,S) coordinates of pixel
    #
    # param: cell-hist - summed time bins of cell
    # param: IRF_decay - shifted IRF values
    # return: gs coordinate as np array
    def __get_GS(self, cell_hist, IRF_decay):
        f = 0.080   # laser repetition rate in [GHz]
        w = 2*np.pi*f
        # time_axis = np.arange(0, 1/f, 1/f/256)
        time_axis = np.arange(0, 10, 10/256)
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
    # param: images - list of of {"name", image_name, cells": cells, "values", cell_values, "IRF_decay":, IRF_decay} dicts
    def plot_cell_phasor(self, images, title, show = False, csv = True):
        # get cell (G,S) for each cell of image
        coords = list()
        names = list()
        
        for image in images:
            image_name = image["name"] + "_channel" + str(image["channel"])
            names.append(image_name)
            
            if csv:
                print(image_name + " opened")
                data = open("Outputs/" + image["name"] + "/channel" + str(image["channel"]) + "/" + image_name + "_data.csv", "w")
                data.write("Cell Number, G coordinate, S coordinate,\n")
            
            subcoords = list()
            i = 0
            for cell in image["cells"]:                    
                # get cell_hist
                cell_hist = np.sum(cell, 0)
                cell_hist = np.sum(cell_hist, 0)
            
                # get gs
                gs = self.__get_GS(cell_hist, image["IRF_decay"])
                subcoords.append(gs)
                
                # write gs to .txt file
                if csv:
                    data.write(str(image["values"][i]) + ", " + str(gs[0]) + ", " + str(gs[1]) + ",\n")
                
                i += 1
                
            coords.append(subcoords)
            
            if csv:
                print(image_name + " closed")
                data.close()
            
        # plot
        visualizer.plot_phasor(title, coords, names, show) 
