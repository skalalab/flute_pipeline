# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:19:00 2024

Demo

@author: chris
"""

from pipeline import Pipeline
from pathlib import Path
from sum_sdts import sum_roi_decays


# run pipeline
pipeline = Pipeline()

# sum sdts 
sum_roi_decays.sum_sdts(Path("./"), Path("./"))

# mask and plot
sdt_paths = [path for path in Path("./").iterdir() if ".sdt" in path.name]

images = list()

for path in sdt_paths:
    if "summed" not in path.name:
        image = pipeline.mask_image(path, "Ch2_IRF_750.txt", Path("./"))
        images.append(image) 
        pipeline.plot_cell_phasor([image], image["name"], show=True)
    else:
        pipeline.process_summed(path, "Ch2_IRF_750.txt")
    
pipeline.plot_cell_phasor(images, "summary", show=True)   