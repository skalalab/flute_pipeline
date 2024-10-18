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

# test constructor
#
# return: false if bad
def test_init():
    # time is -1
    exceptionThrown = False
    try:
        testline = pipeline.Pipeline(-1)
    except:
        exceptionThrown = True
    
    if not exceptionThrown:
        return False
    
    # time is 3
    exceptionThrown = False
    try:
        testline = pipeline.Pipeline(3)
    except:
        exceptionThrown = True
    
    if not exceptionThrown:
        return False
    
    # time is 0
    try:
        testline = pipeline.Pipeline(0)
    except:
        return False
    
    if testline.time_axis != 0:
        return False
    
    # time is 1
    try:
        testline = pipeline.Pipeline(1)
    except:
        return False
    
    if testline.time_axis != 1:
        return False
    
    # time is 2
    try:
        testline = pipeline.Pipeline(2)
    except:
        return False
    
    if testline.time_axis != 2:
        return False
    
    # test all folders are made (time is 0)
    testline = pipeline.Pipeline(0)
    
    
    # all good
    return True

#=======================================================================================

print("test_init(): " + str(test_init()))    