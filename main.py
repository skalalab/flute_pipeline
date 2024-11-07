# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:04:44 2024

@author: Chris Yang

run flute pipeline
"""

import flute_pipeline as pipeline
from pathlib import Path
import sdt_reader as sdt
import tifffile as tiff
import numpy as np


pipeline = pipeline.Pipeline()

# Run pipeline
sdt_paths = [path for path in Path("SDTs").iterdir()]

images = list()
for path in sdt_paths:
    image = pipeline.mask_image(path)
    images.append(image) 
    pipeline.plot_cell_phasor([image])
    
pipeline.plot_cell_phasor(images)      

