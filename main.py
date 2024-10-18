# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:04:44 2024

@author: Chris Yang

run flute pipeline
"""

import flute_pipeline as pipeline
import flute_pipeline_visualizer as visualizer
from pathlib import Path


# Run
time_axis = 0

pipeline = pipeline.Pipeline(time_axis)

mask_paths = [path for path in Path("Masks").iterdir()]
for path in mask_paths:
    pipeline.mask_image(path)
    

visualizer = visualizer.Visualizer(time_axis)
