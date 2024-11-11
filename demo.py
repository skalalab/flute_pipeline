# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:19:00 2024

Demo

@author: chris
"""

from flute_pipeline import Pipeline
from pathlib import Path

# Run pipeline
pipeline = Pipeline()

images = list()

image = pipeline.mask_image(Path("SDTs/dHL60_Control_DMSO_02_n-024.sdt"), "IRFs/txt/Ch2_IRF_750.txt")
images.append(image) 
pipeline.plot_cell_phasor([image])

image = pipeline.mask_image(Path("SDTs/dHL60_Control_na_01_n-010.sdt"), "IRFs/txt/Ch2_IRF_750.txt")
images.append(image) 
pipeline.plot_cell_phasor([image])
    
pipeline.plot_cell_phasor(images)   