import cv2,torch,os
from metrics import *

OLD = "../Multillum_Unet/GALAXY_orgset/train/"
NEW = "galaxy_256/train/"

for file in [f for f in os.listdir(NEW) if f.endswith('.tiff')]:
    if 'gt' in file:
        place,illum_count,_ = file.split('_')
        oldfile = place+'_'+illum_count+'_0_gt.tiff'
    else:
        place,illum_count = os.path.splitext(file)[0].split('_')
        oldfile = place+'_'+illum_count+'_0.tiff'

    img_old = torch.tensor(cv2.imread(OLD+oldfile,cv2.IMREAD_UNCHANGED).astype('float32')).permute(2,0,1).reshape((-1,3,256,256))
    img_new = torch.tensor(cv2.imread(NEW+file,cv2.IMREAD_UNCHANGED).astype('float32')).permute(2,0,1).reshape((-1,3,256,256))

    mae = get_MAE(img_old,img_new,'rgb',camera='galaxy',mask=None)

    print(file,oldfile,":",mae)
    input()