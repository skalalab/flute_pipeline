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
from pipeline import Pipeline
import os

# tests generate_metadata()
#
# return: false if bad
def test_generate_metadata():
    testline = Pipeline()
    
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
    testline = Pipeline()
    
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
    testline = Pipeline()
    
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
    testline = Pipeline()
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
    testline = Pipeline()
    
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
    if not os.path.exists("Outputs/positive/positiveirf.tif"):
        return False
    
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
    if not os.path.exists("Outputs/negative/negativeirf.tif"):
        return False
    
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
    if not os.path.exists("Outputs/no/noirf.tif"):
        return False
    
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


# tests mask_image()
#
# return: false if bad
def test_mask_image():
    testline = Pipeline()
    
    # first image
    results = testline.mask_image(Path("dHL60_Control_DMSO_02_n-024.sdt"), "Ch2_IRF_750.txt", Path("./"))
    
    with tiff.TiffFile("dHL60_Control_DMSO_02_n-024_photons_cellpose.tif") as mask_tif:
        mask = mask_tif.asarray()
    
    sdt_data = sdt.read_sdt150(Path("dHL60_Control_DMSO_02_n-024.sdt"))[1]
    
    # test returned irf decay
    if len(results["IRF_decay"]) != 256:
        return False
    
    if results["IRF_decay"].dtype != np.float32:
        return False
    
    # test returned cell values
    if len(results["cells"]) != 71:
        return False
    
    for cell in results["cells"]:
        for row in range(cell.shape[0]):
            for col in range(cell.shape[1]):
                if np.count_nonzero(cell[row][col]) != 0:
                    if mask[row][col] == 0:
                        return False
                    
                    if not np.array_equal(cell[row][col], sdt_data[row][col]):
                        return False
                    
                    
    # test saved masked image
    masked_path = "Outputs/dHL60_Control_DMSO_02_n-024/dHL60_Control_DMSO_02_n-024masked_image.tif"
    
    if not os.path.exists(masked_path):
        return False
    
    with tiff.TiffFile(masked_path) as image_tif:    
        test_image = image_tif.asarray()
        
    test_image = np.swapaxes(test_image, 0, 2)
    test_image = np.swapaxes(test_image, 0, 1)
        
    for row in range(test_image.shape[0]):
        for col in range(test_image.shape[1]):
            if np.count_nonzero(test_image[row][col]) != 0:
                if mask[row][col] == 0:
                    return False
                
                if not np.array_equal(test_image[row][col], sdt_data[row][col]):
                    return False
            else:
                if mask[row][col] != 0:
                    return False
                
    # test saved irf
    irf_path = "Outputs/dHL60_Control_DMSO_02_n-024/dHL60_Control_DMSO_02_n-024irf.tif"
    
    if not os.path.exists(irf_path):
        return False
    
    with tiff.TiffFile(irf_path) as irf_tif:    
        test_irf = irf_tif.asarray()
    
    test_irf = np.swapaxes(test_irf, 0, 2)
    test_irf = np.swapaxes(test_irf, 0, 1)    
    
    for row in range(test_irf.shape[0]):
        for col in range(test_image.shape[1]):
            if not np.array_equal(test_irf[row][col], results["IRF_decay"]):
                return False
    
    # all good
    return True

# tests plot_cell_phasor
#
# return: false if bad
def test_plot_cell_phasor():
    # setup
    pipeline = Pipeline()

    images = list()
    
    image = pipeline.mask_image(Path("dHL60_Control_DMSO_02_n-024.sdt"), "Ch2_IRF_750.txt", Path("./"))
    images.append(image) 

    image = pipeline.mask_image(Path("dHL60_Control_na_01_n-010.sdt"), "Ch2_IRF_750.txt", Path("./"))
    images.append(image) 
        
    # test first plot
    test_coords = pipeline.plot_cell_phasor([images[0]])
    
    if len(test_coords) != 71:
        return False
    
    # test second plot
    test_coords = pipeline.plot_cell_phasor([images[1]])
    
    if len(test_coords) != 85:
        return False
    
    # test plotting both
    test_coords = pipeline.plot_cell_phasor(images)
    
    if len(test_coords) != 156:
        return False
    
    return True
    

# output "pass" or "fail" depending on if function passed or failed
#
# param: function - function
# return: "pass" or "fail"
def pass_fail(function):
    return ("Passed" if function() else "Failed")
    

#=======================================================================================

# do the tests
print("test_generate_metadata(): " + pass_fail(test_generate_metadata))
print("test_swap_time_axis(): " + pass_fail(test_swap_time_axis))
print("test_shift(): " + pass_fail(test_shift))
print("test_generate_irf(): " + pass_fail(test_generate_irf))
print("test_mask_image(): " + pass_fail(test_mask_image))
# print("test_plot_cell_phasor(): " + pass_fail(test_plot_cell_phasor))



















