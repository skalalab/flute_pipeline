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
    # setup array and pipeline
    testline = pipeline.Pipeline()
    test_array = np.array([1, 2, 5, 2, 1])
    
    # shift by 0
    expected = np.array([1, 2, 5, 2, 1])
    actual = testline._Pipeline__shift(test_array, 0)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by -1
    expected = np.array([2, 5, 2, 1, 0])
    actual = testline._Pipeline__shift(test_array, -1)

    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by 1
    expected = np.array([0, 1, 2, 5, 2])
    actual = testline._Pipeline__shift(test_array, 1)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by 3
    expected = np.array([0, 0, 0, 1, 2])
    actual = testline._Pipeline__shift(test_array, 3)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by -3
    expected = np.array([2, 1, 0, 0, 0])
    actual = testline._Pipeline__shift(test_array, -3)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by 5
    expected = np.array([0, 0, 0, 0, 0])
    actual = testline._Pipeline__shift(test_array, 5)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by -5
    expected = np.array([0, 0, 0, 0, 0])
    actual = testline._Pipeline__shift(test_array, -5)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by 6
    expected = np.array([0, 0, 0, 0, 0, 0])
    actual = testline._Pipeline__shift(test_array, 6)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # shift by -6
    expected = np.array([0, 0, 0, 0, 0, 0])
    actual = testline._Pipeline__shift(test_array, -6)
    
    if (actual.__array_interface__["data"][0] == expected.__array_interface__["data"][0]):
        return False
    
    if not np.array_equal(expected, actual):
        return False
    
    # test int array returns int array
    result = testline._Pipeline__shift(test_array, 2)
    
    if result.dtype != np.int32:
        return False
    
    # test float array reurns float
    test_array = np.array([1.52, 2.42, 504.2, 24442.55])
    result = testline._Pipeline__shift(test_array, 2)
    
    if result.dtype != np.float64:
        return False
    
    # test bool array returns bool
    test_array = np.array([False, True, False, False])
    result = testline._Pipeline__shift(test_array, 2)
    
    if result.dtype != np.bool_:
        return False
    
    # all good
    return True

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
print("test_shift(): " + pass_fail(test_shift))



















