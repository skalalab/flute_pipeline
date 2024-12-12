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
parser.add_argument("--i", "--irfs", nargs="+", help="File paths for each IRF", required=True)
parser.add_argument("--c", "--channels", nargs="+", help = "Channels of data to process. First channel is 0. By default will mask first nonempty", type=int, default=-1)
parser.add_argument("--co", "--column", help = "which column of file irf value is in. Default is 0", type=int, default=0)
args = parser.parse_args()

# run pipeline
pipeline = Pipeline()

# mask and plot
sdt_paths = [path for path in Path(args.sdt_directory).iterdir() if ".sdt" in path.name and "_summed" not in path.name]

images = list()

for path in sdt_paths:
    for i in range(len(args.i)):
        image = pipeline.mask_image(path, args.i[i], Path(args.mask_directory), args.c[i], args.co)
        images.append(image) 
        pipeline.plot_cell_phasor([image], image["name"] + "_channel" + str(image["channel"]))
    
pipeline.plot_cell_phasor(images, "summary", csv=False) 

print("success...")  
    