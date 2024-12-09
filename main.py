# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:04:44 2024

@author: Chris Yang

run flute pipeline
"""

from pipeline import Pipeline
from pathlib import Path
import argparse

# get inputs
parser = argparse.ArgumentParser()
parser.add_argument("sdt_directory", help="directory containing the sdts.")
parser.add_argument("mask_directory", help="directory containing the masks.")
parser.add_argument("irf", help="The IRF file path")
parser.add_argument("--c", help = "Channels of data to process. The first channel is channel 0. separate channel numbers by commas, no spaces. Ex: 1,2,3. By default will mask first nonempty", default="-1")
args = parser.parse_args()

# run pipeline
pipeline = Pipeline()

# parse channels input
channels = [int(channel) for channel in args.c.split(",")]

# mask and plot
sdt_paths = [path for path in Path(args.sdt_directory).iterdir() if ".sdt" in path.name and "_summed" not in path.name]

images = list()

for path in sdt_paths:
    for channel in channels:
        image = pipeline.mask_image(path, args.irf, Path(args.mask_directory), channel)
        images.append(image) 
        pipeline.plot_cell_phasor([image], image["name"] + "_channel" + str(image["channel"]))
    
pipeline.plot_cell_phasor(images, "summary", csv=False) 

print("success...")  
    