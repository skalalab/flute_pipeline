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




#=======================================================================================
time_axis = 0

pipeline = pipeline.Pipeline(time_axis)
        