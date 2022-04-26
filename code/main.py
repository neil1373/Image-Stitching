import argparse
import numpy as np
import os
import cv2
from warp import *
from feature import *
from match import *
from stitch import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str, help = 'Input directory')
    parser.add_argument('--output', type = str, help = 'Output directory')
    args = parser.parse_args()

    # Read inputs
    src_images = []
    image_dir = args.input
    output_dir = args.output
    imgs_name = sorted(os.listdir(image_dir))

    for name in imgs_name:
        if name.endswith(('.png', '.jpg', '.PNG', '.JPG', '.JPEG')):
            image = cv2.imread(os.path.join(image_dir, name))
            if image.shape[0] > 600:
                image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
            src_images.append(image)

    focals = []
    with open(os.path.join(image_dir, 'pano.txt')) as f:
        all = f.readlines()
        for i in range(len(all)):  
            if(i!=0 and all[i-1]=='\n' and all[i+1]=='\n'):
                focals.append(float(all[i]))

    print('==Cylinder Warpping==')
    warp_imgs = cylinder_warpping(src_images,focals)

    print('==Feature Detection==')
    #r_thres = 0.003 (if too many features, adjust it down)
    fea, R = feature_detect(warp_imgs, 0.04, 0.003)

    print('==Feature descriptor==')
    descript=[]
    position=[]
    for i in range(len(warp_imgs)):
        # kernel最低＝5,往下結果會變糟糕
        a, b = descriptor(R[i] ,fea[i],7)
        descript.append(a)
        position.append(b)

    drift = 0
    for i in range(len(warp_imgs) - 1):
        print("==Feature match==")
        matches = matching(warp_imgs[i], warp_imgs[i + 1], descript[i], descript[i + 1], position[i], position[i+1])
        print("==Feature match using RANSAC==")
        shift = RANSAC(matches, 3)
        print(shift)

        if i == 0:
            tgt_img, shift_h = stitching(warp_imgs[i], warp_imgs[i + 1], shift, 0)
        
        else:
            tgt_img, shift_h = stitching(tgt_img, warp_imgs[i + 1], shift, shift_h)
        
        drift += shift[0] * np.sign(shift[1])

    print("[Info]\tdrift =", drift)
    if (not os.path.exists(output_dir)): 
        os.mkdir(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'pano_img.png'), tgt_img)
    tgt_img = global_warping(tgt_img, drift)    
    cv2.imwrite(os.path.join(output_dir, 'pano_img_warp.png'), tgt_img)
