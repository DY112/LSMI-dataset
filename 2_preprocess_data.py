"""
This code
1) splits train/validation/test places,
2) generates MCC chart area mask (mask.png),
3) subtracts default black level for each camera,
3) then crops & resizes the LSMI dataset
"""

import cv2,os,shutil,json
import rawpy
import numpy as np
from tqdm import tqdm
from utils import *

SIZE = 256              # size of output image
CAMERA = "galaxy"       # LSMI subset camera
DST_ROOT = CAMERA + "_" + str(SIZE)
ZERO_MASK = -1          # Zero mask value for black pixels

RAW = CAMERA+".dng"
TEMPLETE = rawpy.imread(RAW)
if CAMERA == "sony":
    BLACK_LEVEL = 128
else:
    BLACK_LEVEL = min(TEMPLETE.black_level_per_channel)
SATURATION = TEMPLETE.white_level


with open(os.path.join(CAMERA,"meta.json"), 'r') as meta_json:
    meta_data = json.load(meta_json)
with open(os.path.join(CAMERA,"split.json"), 'r') as split_json:
    split_data = json.load(split_json)

for key, places in split_data.items():
    split = key.split("_")[-1]
    print("Processing "+key)

    for place in tqdm(places):
        files = [f for f in os.listdir(os.path.join(CAMERA, place)) if f.endswith("tiff")]
        dst_path = os.path.join(DST_ROOT,split)
        if os.path.isdir(dst_path) == False:
            os.makedirs(dst_path)

        for file in files:
            # if "two_illum" in key and "_12.tiff" not in file:
            #     continue
            if "three_illum" in key and "_123.tiff" not in file:
                continue

            fname = os.path.splitext(file)[0]
            illum_count = fname.split("_")[1]

            # open tiff image & subtract black level
            img_org = cv2.cvtColor(cv2.imread(os.path.join(CAMERA,place,file), cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB).astype('float32')
            img_org = np.clip(img_org - BLACK_LEVEL, 0, SATURATION)
            
            # make pixel-level illumination map
            if len(illum_count) == 1:
                mixmap = np.ones_like(img_org[:,:,0:1],dtype=np.float)
            else:
                mixmap = np.load(os.path.join(CAMERA,place,fname+".npy"))
            illum_chroma = [[0,0,0],[0,0,0],[0,0,0]]
            for i in illum_count:
                illum_chroma[int(i)-1] = meta_data[place]["Light"+i]
            illum_map = mix_chroma(mixmap,illum_chroma,illum_count)
            x_zero,y_zero = np.where(mixmap[:,:,0]==ZERO_MASK)
            illum_map[x_zero,y_zero,:] = [1.,1.,1.]

            # white balance original image
            img_wb = img_org / illum_map

            # apply MCC mask to original image, GT image (training set)
            if split == "train":
                mask_org = np.ones_like(img_org[:,:,0:1], dtype='float32')
                mcc1 = (np.float32(meta_data[place]["MCCCoord"]["mcc1"]) / 2).astype(np.int)
                mcc2 = (np.float32(meta_data[place]["MCCCoord"]["mcc2"]) / 2).astype(np.int)
                mcc3 = (np.float32(meta_data[place]["MCCCoord"]["mcc3"]) / 2).astype(np.int)
                mcc_list = [mcc1.tolist(),mcc2.tolist(),mcc3.tolist()]
                for mcc in mcc_list:
                    contour = np.array([[mcc[0]],[mcc[1]],[mcc[2]],[mcc[3]]]).astype(np.int)
                    cv2.drawContours(mask_org, [contour], 0, (0), -1)
                img_org = img_org * mask_org
                img_wb = img_wb * mask_org

            # Crop original image, GT image, MCC mask, mixmap
            height, width, _ = img_org.shape
            w_start = int(width/2) - int(height/2)
            w_end = w_start + height
            img_crop = img_org[:,w_start:w_end,:]
            img_wb_crop = img_wb[:,w_start:w_end,:]
            mixmap_crop = mixmap[:,w_start:w_end,:]
            # prevent negative mask value interpolation if ZERO_MASK is negative value
            mixmap_crop = np.where(mixmap_crop==ZERO_MASK,0,mixmap_crop)
            
            # resize & save
            img_resize = cv2.resize(img_crop, dsize=(SIZE,SIZE), interpolation=cv2.INTER_LINEAR).astype('uint16')
            img_wb_resize = cv2.resize(img_wb_crop, dsize=(SIZE,SIZE), interpolation=cv2.INTER_LINEAR).astype('uint16')
            mixmap_resize = cv2.resize(mixmap_crop, dsize=(SIZE,SIZE), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(dst_path,file), cv2.cvtColor(img_resize,cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dst_path,fname+"_gt.tiff"), cv2.cvtColor(img_wb_resize,cv2.COLOR_RGB2BGR))
            if split == "train":
                mask_crop = mask_org[:,w_start:w_end,:]
                mask_resize = cv2.resize(mask_crop, dsize=(SIZE,SIZE), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(dst_path,place+"_mask.png"), mask_resize)
            if len(illum_count) != 1:
                np.save(os.path.join(dst_path,fname), mixmap_resize)

        # delete original MCC coordinates in meta json
        meta_data[place].pop("MCCCoord")

# shutil.copy(os.path.join(CAMERA,"meta.json"),DST_ROOT)
with open(os.path.join(DST_ROOT,"meta.json"), 'w') as out_file:
    json.dump(meta_data, out_file, indent=4)
shutil.copy(os.path.join(CAMERA,"split.json"),DST_ROOT)