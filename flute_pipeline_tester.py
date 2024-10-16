# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:21:10 2024

@author: Chris Yang

methods for testing the flute_pipeline program :)
"""

import tifffile as tiff
import numpy as np
from pathlib import Path
import sdt_reader as sdt
import flute_pipeline_visualizer as visualizer
import flute_pipeline as pipeline

# get 3d array from file
def extract_array(file):
    if ".sdt" in file:
        array = sdt.read_sdt150(file)
             
        # remove empty channel if needed
        if (array.ndim == 4):
            for i in range(array.shape[0]):
                if (np.count_nonzero(array[i]) == 0):
                    continue
                
                array = array[i]
                break
            
    elif ".tif" in file:
        with tiff.TiffFile(file) as tif:
            array = tif.asarray()
            
    return array

# see if two 3d arrays of files are equal
def file_array_equal(file1, file2):
    file1_array = extract_array(file1)
    file2_array = extract_array(file2)
    
    return np.array_equal(file1_array, file2_array)

# check if irf tif have correct shape
def check_irf_shape(rows, cols, values):
    irf_tifs = [path for path in Path("IRFs").iterdir() if ".tif" in path.name]
    
    # check expected content
    for path in irf_tifs:
        with tiff.TiffFile(path) as tif:
            irf_array = tif.asarray()
                            
            if irf_array.shape != (rows,cols,values):
                return False
            
    # all good
    return True


#=======================================================================================
time_axis = 0

pipeline = pipeline.Pipeline(time_axis)
        