from scipy.spatial import distance
import numpy as np
import os
import cv2
from sys import path

######################
## feature matching ##
######################
def matching(img1, img2, des1, des2, pos1, pos2):
    distances = distance.cdist(des1, des2)
    sorted_index = np.argsort(distances, axis=1)

    num_feature = sorted_index.shape[0]
    match_pairs = []
    for i in range(num_feature):
        row = sorted_index[i]
        # shortest/seconf_shortest < threshold and height of two feature should not be tow far apart
        if distances[i, row[0]] / distances[i, row[1]] < 0.8 and abs(pos1[i][0]-pos2[row[0]][0])<20:
            # img1:des1[i] correspond to img2:des2[row[0]] 
            match_pair = [pos1[i], pos2[row[0]]]
            match_pairs += [match_pair]

    return match_pairs


####################
## image matching ##
####################
def RANSAC(match, threshold = 3):
    match = np.array(match)
    res = []
    maximum = 0
    k = len(match) if(len(match)<1000) else 1000

    for i in range(k):
        sample_idx = np.random.randint(0, len(match)-1)
        sample = match[sample_idx]
        shift = sample[1] - sample[0]
        difference = match[:,0] - match[:,1] + shift  
        tmp = np.sqrt(np.sum(difference ** 2, axis = 1)) 
        inlier = np.sum(tmp< threshold)
        if maximum < inlier:
            maximum = inlier
            res = shift
    return res