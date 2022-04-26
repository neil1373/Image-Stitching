import numpy as np
import os
import cv2

def cylinder_warpping(images, focals):
    # save = 'warping/'
    # if(not os.path.exists(save)): 
    #   os.mkdir(save)
    
    height, width, _ = images[0].shape
    res = []
    w_origin = width//2
    h_origin = height//2
    WW, HH = np.meshgrid(np.arange(width), np.arange(height))

    for i in range(len(images)):
        img = images[i]
        s = focals[i]
        x = s * np.arctan((WW-w_origin) / focals[i])
        y = s * (HH-h_origin) / np.sqrt((WW-w_origin)**2 + s*s) 
        tmp = np.zeros([height, width, 3],dtype=np.uint8)
        for h in range(height):
            for w in range(width):
                tmp[int(h_origin + y[h][w]), int(w_origin + x[h][w]), :] = img[h, w, :]       
        idx = np.argwhere(np.all(tmp[..., :] == [0,0,0], axis=0))
        tmp = np.delete(tmp, idx, axis=1)
        #cv2.imwrite(save+'warp'+str(i)+'.png',tmp)
        res.append(tmp)
    return res