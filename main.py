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
pipeline = pipeline.Pipeline()

mask_paths = [path for path in Path("Masks").iterdir()]

pipeline.mask_image(mask_paths[0])  