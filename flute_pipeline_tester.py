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

# # tests constructor of pipeline
# #
# # return: false if bad
# def test_init():
#     # all folders created
#     testline = pipeline.Pipeline()
    
#     if not os.path.exists("Outputs"):
#         return False
    
#     # all good
#     return True


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


# test generate_irf() function
# 
# return: False if bad
def test_generate_irf():
    testline = pipeline.Pipeline()
    
    # number of irf values less than time bins
    data = np.empty((4,4,25))
    
    exception = False
    try:
        testline._Pipeline__generate_irf("IRFs/testing/no_shift.txt", "invalid", data)
    except:
       exception = True
       
    if not exception:
        return False
    
    # number of irf values more than time bins
    data = np.empty((4,4,23))
    
    exception = False
    try:
        testline._Pipeline__generate_irf("IRFs/testing/no_shift.txt", "invalid", data)
    except:
       exception = True
       
    if not exception:
        return False
    
    # text has whitespace
    values = [0, 5, 10, 9, 8, 5]
    data = np.array([[values, values, values, values], [values, values, values, values], [values, values, values, values], [values, values, values, values]])
    
    try:
        testline._Pipeline__generate_irf("IRFs/testing/blanks.txt", "whitespace", data)
    except Exception as e:
        return False
        
    # positive shift
    values = [0, 0, 0, 0, 0, 1, 1, 2, 4, 8, 16, 8, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    data = np.array([[values, values, values, values], [values, values, values, values], [values, values, values, values], [values, values, values, values]])
    
    actual_values = testline._Pipeline__generate_irf("IRFs/testing/positive_shift.txt", "positive", data)
    
    # check tif
    with tiff.TiffFile("Outputs/positive/positiveirf.tif") as tif:
        actual_tif = tif.asarray()
        
    if actual_tif.dtype != np.float32:
        return False
    
    if actual_tif.shape != (24, 4, 4):
        return False
    
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            for t in range(data.shape[2]):
                if actual_tif[t, row, col] != data[row, col, t]:
                    return False
    
    actual_tif = np.swapaxes(actual_tif, 0, 2)
    actual_tif = np.swapaxes(actual_tif, 0, 1)
    
    if not np.array_equal(actual_tif, data):
        return False
        
    # check return values
    if list(actual_values) != values:
        return False
    
    if actual_values.dtype != np.float32:
        return False
    
    # negative shift
    values = [0, 0, 0, 0, 0, 1, 1, 2, 4, 8, 16, 8, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    data = np.array([[values, values, values, values], [values, values, values, values], [values, values, values, values], [values, values, values, values]])
    
    actual_values = testline._Pipeline__generate_irf("IRFs/testing/negative_shift.txt", "negative", data)
    
    # check tif
    with tiff.TiffFile("Outputs/negative/negativeirf.tif") as tif:
        actual_tif = tif.asarray()
        
    if actual_tif.dtype != np.float32:
        return False
    
    if actual_tif.shape != (24, 4, 4):
        return False
    
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            for t in range(data.shape[2]):
                if actual_tif[t, row, col] != data[row, col, t]:
                    return False
    
    actual_tif = np.swapaxes(actual_tif, 0, 2)
    actual_tif = np.swapaxes(actual_tif, 0, 1)
    
    if not np.array_equal(actual_tif, data):
        return False
        
    # check return values
    if list(actual_values) != values:
        return False
    
    if actual_values.dtype != np.float32:
        return False
    
    # no shift
    values = [0, 0, 0, 0, 0, 1, 1, 2, 4, 8, 16, 8, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    data = np.array([[values, values, values, values], [values, values, values, values], [values, values, values, values], [values, values, values, values]])
    
    actual_values = testline._Pipeline__generate_irf("IRFs/testing/no_shift.txt", "no", data)
    
    # check tif
    with tiff.TiffFile("Outputs/no/noirf.tif") as tif:
        actual_tif = tif.asarray()
        
    if actual_tif.dtype != np.float32:
        return False
    
    if actual_tif.shape != (24, 4, 4):
        return False
    
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            for t in range(data.shape[2]):
                if actual_tif[t, row, col] != data[row, col, t]:
                    return False
    
    actual_tif = np.swapaxes(actual_tif, 0, 2)
    actual_tif = np.swapaxes(actual_tif, 0, 1)
    
    if not np.array_equal(actual_tif, data):
        return False
        
    # check return values
    if list(actual_values) != values:
        return False
    
    if actual_values.dtype != np.float32:
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

# do the tests
# print("test_init(): " + pass_fail(test_init))    
print("test_generate_metadata(): " + pass_fail(test_generate_metadata))
print("test_swap_time_axis(): " + pass_fail(test_swap_time_axis))
print("test_shift(): " + pass_fail(test_shift))
print("test_generate_irf(): " + pass_fail(test_generate_irf))



















