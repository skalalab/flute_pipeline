# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:04:44 2024

@author: Chris Yang

run flute pipeline
"""

import flute_pipeline as pipeline
from pathlib import Path


# Run
pipeline = pipeline.Pipeline(0)

mask_paths = [path for path in Path("Masks").iterdir()]
for path in mask_paths:
    pipeline.mask_image(path)


