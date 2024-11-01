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

# do some shit
summed_sdts = [path for path in Path("SDTs").iterdir() if "summed" in path.name]

for path in summed_sdts:
    sdt_data = sdt.read_sdt150(path)
    
    if (sdt_data.ndim == 4):
        for i in range(sdt_data.shape[0]):
            if (np.count_nonzero(sdt_data[i]) == 0):
                continue
            
            sdt_data = sdt_data[i]
            break
    
    tiff.imwrite("TIFFs/summed/" + path.name[:path.name.find(".sdt")] + ".tif", pipeline._Pipeline__swap_time_axis(sdt_data))


# Run pipeline
sdt_paths = [path for path in Path("SDTs").iterdir()]

images = list()
for path in sdt_paths:
    image = pipeline.mask_image(path)
    images.append(image) 
    pipeline.plot_cell_phasor([image])
    
pipeline.plot_cell_phasor(images)      

