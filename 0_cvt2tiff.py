# Convert single channel bayer pattern RAW files to 3 channel RGB tiff files using DCRAW

import os,shutil,subprocess

SOURCE = "galaxy"
EXT = "dng" # raw file extension. use dng for galaxy, nef for nikon, arw for sony.

places = [f for f in os.listdir(SOURCE) if os.path.isdir(os.path.join(SOURCE,f))==True]

for place in places:
    path = os.path.join(SOURCE, place)
    files = [f for f in os.listdir(path) if f.endswith(EXT)]

    for file in files:
        full_path = os.path.join(path,file)
        cmd = ["dcraw", "-h", "-D", "-4", "-T", full_path]
        print(cmd)
        subprocess.call(cmd)