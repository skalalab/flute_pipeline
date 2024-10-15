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
    
irf_txt_paths = [path for path in Path("IRFs").iterdir() if ".txt" in path.name]
for path in irf_txt_paths:
    pipeline.generate_irfs(path, 256, 256)


