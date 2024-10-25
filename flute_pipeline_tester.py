# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:21:10 2024

@author: Chris Yang

test the flute_pipeline program :)
"""

import tifffile as tiff
import numpy as np
from pathlib import Path
import sdt_reader as sdt
import flute_pipeline_visualizer as visualizer
import flute_pipeline as pipeline
import os

# tests constructor of pipeline
#
# return: false if bad
def test_init():
    # all folders created
    testline = pipeline.Pipeline()
    
    if not os.path.exists("SDTs"):
        return False
    
    if not os.path.exists("Masks"):
        return False
    
    if not os.path.exists("IRFs/tiff"):
        return False
    
    if not os.path.exists("IRFs/txt"):
        return False
    
    if not os.path.exists("TIFFs/Masked"):
        return False
    
    if not os.path.exists("TIFFs/Original"):
        return False
    
    # all good
    return True


# tests generate_metadata()
#
# return: false if bad
def test_generate_metadata():
    testline = pipeline.Pipeline()
    
    # time_bins = 0    
    exception_raised = False
    try:
        testline._Pipeline__generate_metadata(0)
    except:
        exception_raised = True
        
    if not exception_raised:
        return False
    
    # time_bins = 1
    actual_value = testline._Pipeline__generate_metadata(1)
    
    expected_value  = {"ImageJ": "1.54f", "images": 1, "frames": 1, 
                        "loop": False}
    
    if actual_value != expected_value:
        return False
    
    # time_bins = 256
    actual_value = testline._Pipeline__generate_metadata(256)
    
    expected_value  = {"ImageJ": "1.54f", "images": 256, "frames": 256, 
                        "loop": False}
    
    if actual_value != expected_value:
        return False
        
    # all good
    return True

# tests swap_time_axis()
#
# return: false if bad
def test_swap_time_axis():
    # data dimension above 3
    testline = pipeline.Pipeline()
    
    test_data = np.arange(16)
    test_data = np.reshape(test_data, (2, 2, 2, 2))
       
    exception_raised = False
    try:
        testline._Pipeline__swap_time_axis(test_data)
    except:
        exception_raised = True
        
    if not exception_raised:
        return False
    
    # valid 3d array
    testline = pipeline.Pipeline()
    
    test_data = np.arange(8)
    test_data = np.reshape(test_data, (2, 2, 2))
    
    actual = testline._Pipeline__swap_time_axis(test_data)
        
    expected = np.arange(8)
    expected = np.reshape(expected, (2, 2, 2))
    expected = np.swapaxes(expected, 0, 2)
    expected = np.swapaxes(expected, 1, 2)
    
    if not np.array_equal(expected, actual):
        return False
    
    # all good
    return True
    

# test the shift() function
#
# return: False if bad
def test_shift():
    # all good
    return False

# output "pass" or "fail" depending on if function passed or failed
#
# param: function - function
# return: "pass" or "fail"
def pass_fail(function):
    return ("Passed" if function() else "Failed")
    

#=======================================================================================

print("test_init(): " + pass_fail(test_init))    
print("test_generate_metadata(): " + pass_fail(test_generate_metadata))
print("test_swap_time_axis(): " + pass_fail(test_swap_time_axis))



















