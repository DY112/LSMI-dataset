import os,json,cv2,rawpy,math
import numpy as np
from tqdm import tqdm

CAMERA = "galaxy"
RAW = CAMERA+".dng"
VISUALIZE = False
ZERO_MASK = -1              # masking value for unresolved pixel where G = 0 in all image pairs
SAVE_SUBTRACTED_IMG = False # option for saving subtracted image (ex. _2, _3)

RAW_EXT = os.path.splitext(RAW)[1]
TEMPLETE = rawpy.imread(RAW)
if CAMERA == 'sony':
    BLACK_LEVEL = 128
    BLACK_LEVEL_RAW = 512
    SATURATION = 4095
else:
    BLACK_LEVEL = min(TEMPLETE.black_level_per_channel)
    BLACK_LEVEL_RAW = BLACK_LEVEL
    SATURATION = TEMPLETE.white_level
RAW_PATTERN = TEMPLETE.raw_pattern.astype('int8')

"""
cellchart contains 24 color patch coordinates (x, y)

0   1   2   3   4   5
6   7   8   9   10  11
12  13  14  15  16  17
18  19  20  21  22  23

Each color patch coordinates start from upper left, clockwise order
"""
CELLCHART = np.float32([[
    # Row 1
    [0.25, 0.25],   [2.75, 0.25],   [2.75, 2.75],   [0.25, 2.75],
    [3.00, 0.25],   [5.50, 0.25],   [5.50, 2.75],   [3.00, 2.75], 
    [5.75, 0.25],   [8.25, 0.25],   [8.25, 2.75],   [5.75, 2.75],
    [8.50, 0.25],   [11.00, 0.25],  [11.00, 2.75],  [8.50, 2.75],
    [11.25, 0.25],  [13.75, 0.25],  [13.75, 2.75],  [11.25, 2.75],
    [14.00, 0.25],  [16.50, 0.25],  [16.50, 2.75],  [14.00, 2.75],

    # Row 2  
    [0.25, 3.00],   [2.75, 3.00],   [2.75, 5.50],   [0.25, 5.50],
    [3.00, 3.00],   [5.50, 3.00],   [5.50, 5.50],   [3.00, 5.50],
    [5.75, 3.00],   [8.25, 3.00],   [8.25, 5.50],   [5.75, 5.50],
    [8.50, 3.00],   [11.00, 3.00],  [11.00, 5.50],  [8.50, 5.50],
    [11.25, 3.00],  [13.75, 3.00],  [13.75, 5.50],  [11.25, 5.50],
    [14.00, 3.00],  [16.50, 3.00],  [16.50, 5.50],  [14.00, 5.50],

    # Row 3
    [0.25, 5.75],   [2.75, 5.75],   [2.75, 8.25],   [0.25, 8.25],
    [3.00, 5.75],   [5.50, 5.75],   [5.50, 8.25],   [3.00, 8.25],
    [5.75, 5.75],   [8.25, 5.75],   [8.25, 8.25],   [5.75, 8.25],
    [8.50, 5.75],   [11.00, 5.75],  [11.00, 8.25],  [8.50, 8.25],
    [11.25, 5.75],  [13.75, 5.75],  [13.75, 8.25],  [11.25, 8.25],
    [14.00, 5.75],  [16.50, 5.75],  [16.50, 8.25],  [14.00, 8.25],

    # Row 4
    [0.25, 8.50],   [2.75, 8.50],   [2.75, 11.00],  [0.25, 11.00],
    [3.00, 8.50],   [5.50, 8.50],   [5.50, 11.00],  [3.00, 11.00],
    [5.75, 8.50],   [8.25, 8.50],   [8.25, 11.00],  [5.75, 11.00],
    [8.50, 8.50],   [11.00, 8.50],  [11.00, 11.00], [8.50, 11.00],
    [11.25, 8.50],  [13.75, 8.50],  [13.75, 11.00], [11.25, 11.00],
    [14.00, 8.50],  [16.50, 8.50],  [16.50, 11.00], [14.00, 11.00]
]])
MCCBOX = np.float32([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])

def angular_distance(l1, l2):
    unit_l1 = l1 / np.linalg.norm(l1)
    unit_l2 = l2 / np.linalg.norm(l2)
    dot_product = np.dot(unit_l1, unit_l2)
    radian = np.arccos(dot_product)  # radian
    degree = math.degrees(radian)    # degree

    return degree

def make_grid(img1, img1_wb1, img12, img12_wb12, img12_wb1, img12_wb2, img2_wb2, rb_map):
    # convert RGB to BGR
    img1_wb1 = cv2.cvtColor(img1_wb1, cv2.COLOR_RGB2BGR)
    img2_wb2 = cv2.cvtColor(img2_wb2, cv2.COLOR_RGB2BGR)
    img12_wb12 = cv2.cvtColor(img12_wb12, cv2.COLOR_RGB2BGR)
    img12_wb1 = cv2.cvtColor(img12_wb1, cv2.COLOR_RGB2BGR)
    img12_wb2 = cv2.cvtColor(img12_wb2, cv2.COLOR_RGB2BGR)
    rb_map = cv2.cvtColor(rb_map, cv2.COLOR_RGB2BGR)
    
    # resize the side of the images 1/4 the length of the side
    img1 = cv2.resize(img1, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img1_wb1 = cv2.resize(img1_wb1, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12 = cv2.resize(img12, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12_wb12 = cv2.resize(img12_wb12, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12_wb1 = cv2.resize(img12_wb1, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12_wb2 = cv2.resize(img12_wb2, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img2_wb2 = cv2.resize(img2_wb2, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    rb_map = cv2.resize(rb_map, dsize=(1000, 750), interpolation=cv2.INTER_AREA)

    # make text image
    img1 = add_label(img1, "img1")
    img1_wb1 = add_label(img1_wb1, "img1_wb1")
    img12 = add_label(img12, "img12")
    img12_wb12 = add_label(img12_wb12, "img12_wb12")
    img12_wb1 = add_label(img12_wb1, "img12_wb1")
    img12_wb2 = add_label(img12_wb2, "img12_wb2")
    img2_wb2 = add_label(img2_wb2, "img2_wb2")
    rb_map = add_label(rb_map, "rb_map")
    
    # concatenate all
    col1 = np.hstack((img1, img1_wb1))
    col2 = np.hstack((img12, img12_wb12))
    col3 = np.hstack((img12_wb1, img12_wb2))
    col4 = np.hstack((img2_wb2, rb_map))
    return cv2.vconcat([col1, col2, col3, col4])

def add_label(img, name):
    """
    img                 : image matrix
    name                : string
    """
    text = np.zeros((100, img.shape[1], 3), np.uint8) + 255
    cv2.putText(text, name, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((img, text))
    
def get_rb_map(coefficient_map):
    h,w = coefficient_map.shape

    rb_map = np.zeros((h,w,3), dtype=np.uint8)
    rb_map[:,:,0] = coefficient_map * 255
    rb_map[:,:,2] = (1 - coefficient_map) * 255
    
    return rb_map

def apply_wb_raw(raw, illumination_map):
    """
    raw                 : rawpy.RawPy class
    illumination_map    : half size illumination map in RGB channel order
    """
    h,w,_ = illumination_map.shape
    rh,rw = raw.raw_image.shape
    margin_h = int((rh-h*2)/2)
    margin_w = int((rw-w*2)/2)

    raw_matrix = np.clip(raw.raw_image.copy().astype('int16') - BLACK_LEVEL_RAW,0,SATURATION)
    raw_multiplier = np.ones_like(raw.raw_image,dtype=np.float32)

    r_multiplier = illumination_map[:,:,1] / illumination_map[:,:,0]
    b_multiplier = illumination_map[:,:,1] / illumination_map[:,:,2]

    wb_matrix = np.tile(-RAW_PATTERN, (h,w)).astype(np.float32)
    wb_matrix[wb_matrix==0] = r_multiplier.reshape(-1)
    wb_matrix[wb_matrix==-1] = 1
    wb_matrix[wb_matrix==-2] = b_multiplier.reshape(-1)
    wb_matrix[wb_matrix==-3] = 1

    raw_multiplier[margin_h:margin_h+h*2,margin_w:margin_w+w*2] = wb_matrix
    raw_matrix_wb = raw_matrix * raw_multiplier + BLACK_LEVEL_RAW

    for i in range(rh):
        for j in range(rw):
            raw.raw_image[i,j] = raw_matrix_wb[i,j]

def get_patch_chroma(chroma_map, method, normalize='green'):
    """
    chroma_map  : (3, 6, 3) shape array (MCC, graypatch, chroma)
    
    method      : mean or max
                  mean - average three brightest patches from three MCCs excluding saturation
                  max  - only one brightest patch excluding saturation

    normalize   : 'green' or 'sum'
                  green - normalize chroma vector to G=1
                  sum   - normalize chroma vector to sum = 1

    returns     : maxChartIdx, maxPatchIdx, normalized rg-chroma
                  (-1, -1, chroma) for method = "mean"
    """
    assert chroma_map.shape == (3,6,3)
    
    maxChartIdx = -1
    maxPatchIdx = -1
    retChroma = np.array([0, 0, 0])

    for chartIdx in range(3):
        for patchIdx in range(6):
            chroma = chroma_map[chartIdx, patchIdx, :]
            if SATURATION in chroma:
                continue

            elif method == "max" and chroma[1] > retChroma[1]:
                maxChartIdx = chartIdx
                maxPatchIdx = patchIdx
                retChroma = chroma

            elif method == "mean":
                retChroma += chroma
                break
    # print(retChroma)
    rgChroma = retChroma / np.sum(retChroma)
    g1_chroma = retChroma / retChroma[1]

    # print(rgChroma, g1_chroma)

    if normalize == 'sum':
        return maxChartIdx, maxPatchIdx, list(rgChroma)
    elif normalize == 'green':
        return maxChartIdx, maxPatchIdx, list(g1_chroma)

def get_coefficient_map(img_1_wb, img_2_wb, zero_mask=-1):
    """
    zero_mask   : masking value for pixel where both G = 0
    returns     : img_1's illuminant coefficient r
    """

    denominator = img_1_wb[:,:,1] + img_2_wb[:,:,1]

    # compute coefficient. fill zero_mask value for invalid denominator (if G value from both image = 0)
    coefficient = img_1_wb[:,:,1] / np.clip(denominator, 0.0001, SATURATION)
    coefficient = np.where(denominator==0, zero_mask, coefficient)

    return coefficient

def get_illuminant_chroma(img, mcc_list):
    """
    img         : BGR image
    mcc_list    : MCC chart coordinate list

    returns     : numpy array with shape (3,6,3)
                  (MCC chart, patch, RGB channel sum)
    """
    chroma = np.zeros((3,6,3), dtype=int)

    for mcc_idx in range(len(mcc_list)):
        mcc = mcc_list[mcc_idx]
        src = MCCBOX
        dst = mcc

        # Get perspective transform matrix, apply transform to cellchart
        M = cv2.getPerspectiveTransform(src, dst)
        cellchart = cv2.perspectiveTransform(CELLCHART, M)
        cellchart = np.reshape(cellchart, (24,4,2))

        # Reduce the box size by 50%
        for i in range(24):
            centerPoint = np.sum(cellchart[i], axis=0) / 4
            for j in range(4):
                cellchart[i][j] = (cellchart[i][j] + centerPoint) / 2

        # generate mask for gray patches (18~23) & record chromaticity
        for i in range(18,24):
            mask = np.zeros_like(img)
            cell = np.array([[cellchart[i,0]], [cellchart[i,1]], [cellchart[i,2]], [cellchart[i,3]]]).astype(int)
            cv2.drawContours(mask, [cell], 0, (1,1,1), -1) # fill inside the contour
            maskedImage = img*mask

            # RGB channelwise sum & flip (BGR to RGB)
            sumRGB = np.flip(np.sum(maskedImage, axis=(0,1)))
            chroma[mcc_idx, i-18, :] = sumRGB

            if True in (maskedImage >= (SATURATION - BLACK_LEVEL)):
                chroma[mcc_idx, i-18, :] = SATURATION

    return chroma

def get_illumination_map(place, placeInfo):
    # directory configuration
    # print("\n",place)
    src_path = os.path.join(CAMERA, place) + "/"
    vis_path = os.path.join(CAMERA+"_visualize")

    if os.path.isdir(vis_path) == False and VISUALIZE:
        os.makedirs(vis_path)

    # read place annotation data from json data
    numOfLights = placeInfo["NumOfLights"]
    mccScale = 2 #placeInfo["MCCScale"]
    mcc1 = placeInfo["MCCCoord"]["mcc1"]
    mcc2 = placeInfo["MCCCoord"]["mcc2"]
    mcc3 = placeInfo["MCCCoord"]["mcc3"]
    mcc_list = np.float32([mcc1, mcc2, mcc3]) / mccScale

    # make illumination map of 2 images (1,12)
    if placeInfo["NumOfLights"] == 2:
        singleimage = place + "_1"
        multiimage = place + "_12"

        # prevent uint16 type subtraction underflow
        img_1 = cv2.imread(src_path + singleimage + ".tiff", cv2.IMREAD_UNCHANGED).astype("int16")
        img_12 = cv2.imread(src_path + multiimage + ".tiff", cv2.IMREAD_UNCHANGED).astype("int16")
        
        img_2 = np.clip(img_12 - img_1, 0, SATURATION - BLACK_LEVEL)
        img_1 = np.clip(img_1 - BLACK_LEVEL, 0, SATURATION - BLACK_LEVEL)

        if SAVE_SUBTRACTED_IMG:
            cv2.imwrite(src_path + place + "_2.tiff", img_2.astype("uint16"))

        # calculate MCC gray cellchart RGB value (shape 3,6,3)
        chroma_1 = get_illuminant_chroma(img_1, mcc_list)
        chroma_2 = get_illuminant_chroma(img_2, mcc_list)

        # get maximum chart,patch,chromaticity without saturation
        # json value is directly updated with calculated chromaticity
        maxChart1, maxPatch1, placeInfo["Light1"] = get_patch_chroma(chroma_1, method="max", normalize='green')
        maxChart2, maxPatch2, placeInfo["Light2"] = get_patch_chroma(chroma_2, method="max", normalize='green')

        light_1 = placeInfo["Light1"]
        light_2 = placeInfo["Light2"]

        # calculate angular distance between light 1 & 2
        placeInfo["AD"] = angular_distance(placeInfo["Light1"], placeInfo["Light2"])

        # calculate brightness (sum of G pixels) difference of light2-affected MCC
        before = np.sum(chroma_1[maxChart2, maxPatch2, 1])
        diff = np.sum(chroma_2[maxChart2, maxPatch2, 1])
        ratio = (diff / before) * 100.
        placeInfo["BrightnessDiff"] = ratio

        """
        From here, calculate each pixel's light combination coefficient map
        and save them as numpy array (.npy)
        """
        # generate coefficient (mixture) map from G channel
        coefficient_1 = get_coefficient_map(img_1, img_2, ZERO_MASK)
        coefficient_2 = np.where(coefficient_1==ZERO_MASK,ZERO_MASK,1.0 - coefficient_1)

        # save coefficient map
        coefficient_map = np.stack((coefficient_1, coefficient_2), axis=-1)
        np.save(src_path + multiimage, coefficient_map)

        # calculate coefficient statistics
        masked_coefficient_map = coefficient_1[coefficient_1>-1]
        var_1 = np.var(masked_coefficient_map)
        var_2 = np.var(1-masked_coefficient_map)
        std_1 = np.std(masked_coefficient_map)
        std_2 = np.std(1-masked_coefficient_map)
        placeInfo["CoeffVariance"] = (var_1 + var_2) / 2
        placeInfo["CoeffSTD"] = (std_1 + std_2) / 2

        """
        ##########################################################################
        #  JPG Visualization - using RawPy                                       #
        #  If you use RawPy, lots of postprocess operations are performed.       #
        #  (auto brightness, 8bit sRGB colorspace transform, etc...)             #
        #  You can control them, by using arguments of postprocess() function.   #
        ##########################################################################
        """
        if VISUALIZE:
            # open two raw images (1,12)
            raw_1 = rawpy.imread(src_path + singleimage + RAW_EXT)
            raw_12 = rawpy.imread(src_path + multiimage + RAW_EXT)
            
            # subtract two raw images (12 - 1)
            raw_2 = rawpy.imread(src_path + multiimage + RAW_EXT)
            raw_12_matrix = raw_2.raw_image.copy().astype('int16')
            raw_1_matrix = raw_1.raw_image.copy().astype('int16')
            raw_2_matrix = np.clip(raw_12_matrix - raw_1_matrix, 0, SATURATION) + BLACK_LEVEL
            height, width = raw_2.sizes.raw_height, raw_2.sizes.raw_width
            for h in range(height):
                for w in range(width):
                    raw_2.raw_image[h,w] = raw_2_matrix[h,w]
            
            # compute mixed illumination map and apply WB
            illumination_map_12 = np.stack((coefficient_1,)*3, axis=2) * [[light_1]] \
                                + np.stack((coefficient_2,)*3, axis=2) * [[light_2]]
            z, y, x = np.where(coefficient_map == -1)
            for i in range(len(x)):
                illumination_map_12[z[i], y[i], x[i]] = 1/3
            
            apply_wb_raw(raw_12, illumination_map_12)
            img12_wb12 = raw_12.postprocess(user_black=BLACK_LEVEL, user_wb=[1,1,1,1], no_auto_bright=True, half_size=True)

            raw_12 = rawpy.imread(src_path + multiimage + RAW_EXT)
            rgb_12_awb = raw_12.postprocess(use_auto_wb=True, no_auto_bright=True, half_size=True)
            rgb_12_daylight = raw_12.postprocess(user_wb=raw_12.daylight_whitebalance, no_auto_bright=True, half_size=True)
            rgb_12_camera = raw_12.postprocess(user_wb=raw_12.camera_whitebalance, no_auto_bright=True, half_size=True)

            cv2.imwrite(os.path.join(vis_path, place+"_awb.png"), cv2.cvtColor(rgb_12_awb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_path, place+"_daylight.png"), cv2.cvtColor(rgb_12_daylight, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_path, place+"_camera.png"), cv2.cvtColor(rgb_12_camera, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_path, place+"_wb12.png"), cv2.cvtColor(img12_wb12, cv2.COLOR_RGB2BGR))

            # rb-map image
            rb_map = get_rb_map(coefficient_1)
            cv2.imwrite(os.path.join(vis_path, place+"_rbmap.jpg"), cv2.cvtColor(rb_map, cv2.COLOR_RGB2BGR))


    # make illumination map (1,12,13,123 pair)
    elif placeInfo["NumOfLights"] == 3:
        img_1_name = place + "_1"
        img_12_name = place + "_12"
        img_13_name = place + "_13"
        img_123_name = place + "_123"

        img_1 = cv2.imread(src_path + img_1_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        img_12 = cv2.imread(src_path + img_12_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        img_13 = cv2.imread(src_path + img_13_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        img_123 = cv2.imread(src_path + img_123_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        
        # prevent uint16 type subtraction underflow
        img_1_int16 = img_1.astype("int16")
        img_12_int16 = img_12.astype("int16")
        img_13_int16 = img_13.astype("int16")
        img_123_int16 = img_123.astype("int16")

        # pixel level image subtraction
        img_2 = np.clip(img_12_int16 - img_1_int16, 0, SATURATION - BLACK_LEVEL)
        img_3 = np.clip(img_13_int16 - img_1_int16, 0, SATURATION - BLACK_LEVEL)
        img_23 = np.clip(img_123_int16 - img_1_int16, 0, SATURATION - BLACK_LEVEL)

        if SAVE_SUBTRACTED_IMG:
            cv2.imwrite(src_path + place + "_2.tiff", img_2.astype("uint16"))
            cv2.imwrite(src_path + place + "_3.tiff", img_3.astype("uint16"))
            cv2.imwrite(src_path + place + "_23.tiff", img_23.astype("uint16"))

        # calculate MCC gray cellchart RGB value (shape 3,6,3)
        chroma_1 = get_illuminant_chroma(img_1, mcc_list)
        chroma_2 = get_illuminant_chroma(img_2, mcc_list)
        chroma_3 = get_illuminant_chroma(img_3, mcc_list)

        # get maximum chart,patch,chromaticity without saturation
        # json value is directly updated with calculated chromaticity
        maxChart1, maxPatch1, placeInfo["Light1"] = get_patch_chroma(chroma_1, method="max", normalize='green')
        maxChart2, maxPatch2, placeInfo["Light2"] = get_patch_chroma(chroma_2, method="max", normalize='green')
        maxChart3, maxPatch3, placeInfo["Light3"] = get_patch_chroma(chroma_3, method="max", normalize='green')

        light_1 = placeInfo["Light1"]
        light_2 = placeInfo["Light2"]
        light_3 = placeInfo["Light3"]

        # calculate angular distance between lights
        placeInfo["AD12"] = angular_distance(placeInfo["Light1"], placeInfo["Light2"])
        placeInfo["AD23"] = angular_distance(placeInfo["Light2"], placeInfo["Light3"])
        placeInfo["AD31"] = angular_distance(placeInfo["Light3"], placeInfo["Light1"])

        """
        From here, calculate each pixel's light combination coefficient map
        and save them as 2-channel numpy array (.npy)
        """
        # generate coefficient map from G channel
        # cannot use get_coefficient_map function in 3 lights case
        denominator_13 = img_1[:,:,1] + img_3[:,:,1]
        denominator_12 = img_1[:,:,1] + img_2[:,:,1]
        denominator_123 = img_1[:,:,1] + img_2[:,:,1] + img_3[:,:,1]

        # compute coefficient. -1 for invalid denominator_123 (if G value from both image = 0)
        coefficient_1 = img_1[:,:,1] / np.clip(denominator_12, 0.0001, SATURATION)
        coefficient_1 = np.where(denominator_12==0, ZERO_MASK, coefficient_1)
        coefficient_2 = img_2[:,:,1] / np.clip(denominator_12, 0.0001, SATURATION)
        coefficient_2 = np.where(denominator_12==0, ZERO_MASK, coefficient_2)
        coefficient_1 = coefficient_1.clip(0, 1)
        coefficient_2 = coefficient_2.clip(0, 1)
        coefficient_map_12 = np.stack((coefficient_1, coefficient_2), axis=-1)
        np.save(src_path + img_12_name, coefficient_map_12)

        coefficient_1 = img_1[:,:,1] / np.clip(denominator_13, 0.0001, SATURATION)
        coefficient_1 = np.where(denominator_13==0, ZERO_MASK, coefficient_1)
        coefficient_3 = img_3[:,:,1] / np.clip(denominator_13, 0.0001, SATURATION)
        coefficient_3 = np.where(denominator_13==0, ZERO_MASK, coefficient_3)
        coefficient_1 = coefficient_1.clip(0, 1)
        coefficient_3 = coefficient_3.clip(0, 1)
        coefficient_map_13 = np.stack((coefficient_1, coefficient_3), axis=-1)
        np.save(src_path + img_13_name, coefficient_map_13)

        coefficient_2 = img_2[:,:,1] / np.clip(denominator_123, 0.0001, SATURATION)
        coefficient_2 = np.where(denominator_123==0, ZERO_MASK, coefficient_2)
        coefficient_3 = img_3[:,:,1] / np.clip(denominator_123, 0.0001, SATURATION)
        coefficient_3 = np.where(denominator_123==0, ZERO_MASK, coefficient_3)
        coefficient_2 = coefficient_2.clip(0, 1)
        coefficient_3 = coefficient_3.clip(0, 1)
        coefficient_1 = np.where(denominator_123==0, ZERO_MASK, 1 - coefficient_2 - coefficient_3)
        coefficient_1 = coefficient_1.clip(0, 1)
        coefficient_map = np.stack((coefficient_1, coefficient_2, coefficient_3), axis=-1)
        np.save(src_path + img_123_name, coefficient_map)

        # save coefficient statistics
        masked_coefficient_1 = coefficient_1[coefficient_1>-1]
        masked_coefficient_2 = coefficient_2[coefficient_2>-1]
        masked_coefficient_3 = coefficient_3[coefficient_3>-1]

        var_1 = np.var(masked_coefficient_1)
        var_2 = np.var(masked_coefficient_2)
        var_3 = np.var(masked_coefficient_3)
        std_1 = np.std(masked_coefficient_1)
        std_2 = np.std(masked_coefficient_2)
        std_3 = np.std(masked_coefficient_3)

        placeInfo["CoeffVariance"] = (var_1 + var_2 + var_3) / 3
        placeInfo["CoeffSTD"] = (std_1 + std_2 + std_3) / 3

        """
        ##########################################################################
        #  JPG Visualization - using RawPy                                       #
        #  If you use RawPy, lots of postprocess operations are performed.       #
        #  (auto brightness, 8bit sRGB colorspace transform, etc...)             #
        #  You can control them, by using arguments of postprocess() function.   #
        ##########################################################################
        """
        if VISUALIZE:
            # open two raw images (1,12, 13, 123)
            raw_1 = rawpy.imread(src_path + img_1_name + RAW_EXT)
            raw_12 = rawpy.imread(src_path + img_12_name + RAW_EXT)
            raw_13 = rawpy.imread(src_path + img_13_name + RAW_EXT)
            raw_123 = rawpy.imread(src_path + img_123_name + RAW_EXT)
            
            # subtract two raw images (12 - 1)
            raw_2 = rawpy.imread(src_path + img_12_name + RAW_EXT)
            raw_2_matrix = raw_2.raw_image
            raw_1_matrix = raw_1.raw_image
            height, width = raw_2.sizes.raw_height, raw_2.sizes.raw_width
            for h in range(height):
                for w in range(width):
                    if raw_2_matrix[h,w] < raw_1_matrix[h,w]:
                        raw_2_matrix[h,w] = 0
                    else:
                        raw_2_matrix[h,w] = raw_2_matrix[h,w] - raw_1_matrix[h,w]

            # subtract two raw images (123 - 12)
            raw_3 = rawpy.imread(src_path + img_123_name + RAW_EXT)
            raw_3_matrix = raw_3.raw_image
            raw_12_matrix = raw_12.raw_image
            height, width = raw_3.sizes.raw_height, raw_3.sizes.raw_width
            for h in range(height):
                for w in range(width):
                    if raw_3_matrix[h,w] < raw_12_matrix[h,w]:
                        raw_3_matrix[h,w] = 0
                    else:
                        raw_3_matrix[h,w] = raw_3_matrix[h,w] - raw_12_matrix[h,w]
            
            # compute mixed illumination map and apply WB
            illumination_map_123 = np.stack((coefficient_map[:,:,0],)*3, axis=2) * [[light_1]] \
                                + np.stack((coefficient_map[:,:,1],)*3, axis=2) * [[light_2]] \
                                + np.stack((coefficient_map[:,:,2],)*3, axis=2) * [[light_3]]
            z, y, x = np.where(coefficient_map == -1)
            for i in range(len(x)):
                illumination_map_123[z[i], y[i], x[i]] = 1/3

            apply_wb_raw(raw_123, illumination_map_123)

            # apply white balance & decode raw file to 3 channel image
            # img1_wb1 = raw_1.postprocess(user_wb=[light_1[1]/light_1[0], 1.0, light_1[1]/light_1[2], 1.0], no_auto_bright=True, half_size=True)
            # img2_wb2 = raw_2.postprocess(user_wb=[light_2[1]/light_2[0], 1.0, light_2[1]/light_2[2], 1.0], no_auto_bright=True, half_size=True)
            # img12_wb12 = raw_12.postprocess(user_wb=[1,1,1,1], no_auto_bright=True, half_size=True)

            img123_wb123 = raw_123.postprocess(user_black=BLACK_LEVEL, user_wb=[1,1,1,1], no_auto_bright=True, half_size=True)
            img123_wb123 = cv2.cvtColor(img123_wb123, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(vis_path, place+"_IMG123_WB123.jpg"), img123_wb123)

            # rb-map image
            rgb_map = np.zeros_like(coefficient_map, dtype=np.uint8)
            rgb_map[:,:,0] = coefficient_map[:,:,0] * 255
            rgb_map[:,:,1] = coefficient_map[:,:,1] * 255
            rgb_map[:,:,2] = coefficient_map[:,:,2] * 255
            cv2.imwrite(os.path.join(vis_path, place+"_coefficient_map.png"), cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR))

            # calculate coefficient variance
            placeInfo["CoeffVariance"] = np.mean(np.var(coefficient_map, axis=(0,1)))

    # return Json data contains Light_RGB
    return placeInfo

if __name__ == "__main__":
    print("Generating",CAMERA,"mixture map...")
    
    # open json annotation file
    with open(os.path.join(CAMERA, "meta.json"), 'r') as json_file:
        jsonData = json.load(json_file)

    # initialize coefficient std array
    coeff_std = []
    coeff_var = []

    # get directory list
    places = sorted([f for f in os.listdir(CAMERA) if os.path.isdir(os.path.join(CAMERA,f))])

    for place in tqdm(places):
        # read json annotation
        placeInfo = jsonData[place]

        # get illumination map & update json (light RGB)
        jsonData[place] = get_illumination_map(place, placeInfo)
        
        coeff_var.append(jsonData[place]["CoeffVariance"])
        coeff_std.append(jsonData[place]["CoeffSTD"])

    print("Coeff Variance :", np.mean(coeff_var))
    print("Coeff STD :", np.mean(coeff_std))

    # save json annotation
    with open(os.path.join(CAMERA, "meta.json"), 'w') as out_file:
        json.dump(jsonData, out_file, indent=4)
