# FLUTE Pipeline
## Purpose
TBD...
## Basic Overview
1. FLIM image data in the form of *.sdt* files are inputted into our pipeline.
2. The *.sdt* files are split and converted into single channel *.tif* files. Only the non-empty files are saved.
3. User provided masks are applied onto corresponding *.tif* files and the new image is saved.
4. Each individual cell in the masked image is saved as its own *.tif* file
5. ...
