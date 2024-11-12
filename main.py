# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:04:44 2024

@author: Chris Yang

run flute pipeline
"""

from flute_pipeline import Pipeline
from pathlib import Path
import argparse


# get inputs
parser = argparse.ArgumentParser()
parser.add_argument("sdt_directory", help="directory containing the sdts.")
parser.add_argument("mask_directory", help="directory containing the masks.")
parser.add_argument("irf", help="The IRF file path")
args = parser.parse_args()

# run
pipeline = Pipeline()

sdt_paths = [path for path in Path(args.sdt_directory).iterdir() if ".sdt" in path.name]


images = list()

for path in sdt_paths:
    image = pipeline.mask_image(path, args.irf, args.mask_directory)
    images.append(image) 
    pipeline.plot_cell_phasor([image])
    
pipeline.plot_cell_phasor(images)   
    