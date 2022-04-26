import numpy as np
import os
import cv2
from sys import path

#######################
## feature detection ##
#######################
def Harris(gray, k):
    #assume kernel size = 9
    Ix = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    Iy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    Iyy = Iy**2
    Ixx = Ix**2
    Ixy = Ix*Iy
    #Compute the sums of the products of derivatives at each pixel
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)

    detM = Sxx * Syy - (Sxy**2)
    traceM = Sxx + Syy

    R = detM - k*(traceM**2)
    return R

def feature_detect(images, k =0.04, r_thres=0.01):
    save = 'detected/'
    if(not os.path.exists(save)): 
        os.mkdir(save)
    corners = []
    Rs = []
    for i in range(len(images)):
        gray_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        R = Harris(gray_img, k)
        Rs.append(R)
        #find corner
        maxR = R.max()
        h, w, _ = images[i].shape
        corner = np.zeros_like(R,dtype=np.uint8)
        # Reduce corner
        # start from 10 and end with -10 to let point don't locate in edge
        for a in range(10,h-10):
            for b in range(10,w-10):
                # window = 5
                if R[a,b] > maxR*r_thres and R[a,b] == np.max(R[(a-2):(a+3),(b-2):(b+3)]):
                    corner[a,b] = 255
        img = images[i].copy()
        img[corner>0.5*corner.max()] =[0,0,255]
        cv2.imwrite(save+'detect'+str(i)+'.png',img)
        corners.append(corner)    

    return corners, Rs

########################
## feature descriptor ##
########################
def descriptor(R, corner_response, kernel=3):
    h, w = corner_response.shape
    positions = []
    descriptions = []

    for a in range(h):
        for b in range(w):
            # window = kernel
            if corner_response[a][b] == 255:
                if((a-1)<0 or (a+kernel-1)>(w-1) or (b-1)<0 or(b+kernel-1)>(w-1)): continue
                positions += [[a, b]]
                tmp = R[(a-1):(a+kernel-1),(b-1):(b+kernel-1)]
                descriptions.append(tmp.flatten())

    return descriptions, positions