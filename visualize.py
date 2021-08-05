import rawpy
import cv2,os
import numpy as np
import multiprocessing

def func(arg):
    print(arg)
    split = arg[0]
    file = arg[1]
    camera = arg[2]
    raw = rawpy.imread(camera + ".dng")

    img = cv2.imread(os.path.join(split, file), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bayer = bayerize(img, camera)
    img_bayer += min(raw.black_level_per_channel)
    height, width, _ = img.shape
    if "gt" in file:
        img_render = render(raw, img_bayer, height, width, [1,1,1,1])
    else:
        img_render = render(raw, img_bayer, height, width, raw.daylight_whitebalance)

    cv2.imwrite(os.path.join("vis",os.path.splitext(file)[0]+'.png'), cv2.cvtColor(img_render, cv2.COLOR_RGB2BGR))

def bayerize(img_rgb, camera):
    h,w,c = img_rgb.shape

    bayer_pattern = np.zeros((h*2,w*2))

    if camera == "galaxy":
        bayer_pattern[0::2,1::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,2] # B
    elif camera == "sony" or camera == "nikon":
        bayer_pattern[0::2,0::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,2] # B

    return bayer_pattern

def render(raw, bayer, height, width, wb_mat):
    raw_mat = raw.raw_image
    for h in range(height*2):
        for w in range(width*2):
            raw_mat[h,w] = bayer[h,w]

    rgb = raw.postprocess(user_wb=wb_mat, highlight_mode=rawpy.HighlightMode(0), half_size=True, no_auto_bright=True)
    rgb_croped = rgb[0:height,0:width,:]
    
    return rgb_croped

if __name__ == '__main__':
    split = ['galaxy_256/test','GALAXY_orgset/test']
    camera = 'galaxy'

    arg_list = []

    for s in split:
        files = [f for f in os.listdir(s) if not f.startswith(".") and f.endswith('.tiff')]# and "gt" not in f]

        for file in files:
            arg = []
            arg.append(s)
            arg.append(file)
            arg.append(camera)
            arg_list.append(arg)

    pool = multiprocessing.Pool(processes=16)
    pool.map(func, arg_list)
