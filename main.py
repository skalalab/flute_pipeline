# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:04:44 2024

@author: Chris Yang

run flute pipeline
"""

import flute_pipeline as pipeline
import flute_pipeline_visualizer as visualizer
from pathlib import Path
import sdt_reader as sdt
import tifffile as tiff


# Run
pipeline = pipeline.Pipeline()

# sdt_data = sdt.read_sdt150("SDTs/dHL60_Control_DMSO_02_n-024_summed.sdt")
# sdt_data = sdt_data[1]
# tiff.imwrite("test.tif", pipeline._Pipeline__swap_time_axis(sdt_data))

sdt_paths = [path for path in Path("SDTs").iterdir()]

images = list()
for path in sdt_paths:
    image = pipeline.mask_image(path)
    images.append(image) 
    pipeline.plot_cell_phasor([image])
    
pipeline.plot_cell_phasor(images)      

