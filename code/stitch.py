from scipy.spatial import distance
import numpy as np
import os
import cv2
from sys import path

def stitching(tgt_img, src_img, shift, shift_h):
	width = tgt_img.shape[1] + abs(shift[1])

	if shift_h + shift[0] < 0:
		height = tgt_img.shape[0] + abs(shift[0] + shift_h)
	elif shift[0] > 0:
		height = tgt_img.shape[0] + shift[0]
	else:
		height = tgt_img.shape[0]

	channels = tgt_img.shape[2]

	img_1 = np.zeros((height, width, channels))	# left image
	img_2 = np.zeros((height, width, channels))	# right image

	if shift[1] < 0:
		if shift[0] + shift_h > 0:
			img_2[:src_img.shape[0], -src_img.shape[1]:, :] = src_img
			img_1[-tgt_img.shape[0]:, :tgt_img.shape[1], :] = tgt_img
		else:
			img_2[-src_img.shape[0]:, -src_img.shape[1]:, :] = src_img
			img_1[:tgt_img.shape[0], :tgt_img.shape[1], :] = tgt_img
	else:
		if shift[0] + shift_h > 0:
			# img_1[:tgt_img.shape[0], :tgt_img.shape[1], :] = tgt_img
			# img_2[shift[0] + shift_h : shift[0] + shift_h + src_img.shape[0], -src_img.shape[1]:, :] = src_img
			img_1[:src_img.shape[0], :src_img.shape[1], :] = src_img
			img_2[-tgt_img.shape[0]:, -tgt_img.shape[1]:, :] = tgt_img
		else:
			img_1[-src_img.shape[0]:, :src_img.shape[1], :] = src_img
			img_2[:tgt_img.shape[0], -tgt_img.shape[1]:, :] = tgt_img

	merge_img = np.zeros_like(img_1)

	cv2.imwrite('test_stitch/left.png', img_1)
	cv2.imwrite('test_stitch/right.png', img_2)
	

	if shift[1] < 0:
		stitch_left = tgt_img.shape[1] + abs(shift[1]) - src_img.shape[1]
		stitch_right = tgt_img.shape[1]
		print(stitch_left, stitch_right)
	else:
		stitch_left = abs(shift[1])
		stitch_right = src_img.shape[1]

	merge_img[:, :stitch_left, :] = img_1[:, :stitch_left, :]
	merge_img[:, stitch_right:, :] = img_2[:, stitch_right:, :]

	# linear transformation in overlapping area
	for i in range(stitch_left, stitch_right):
		merge_img[:, i, :] = (img_1[:, i, :] * (stitch_right - i) + img_2[:, i, :]*(i - stitch_left)) / (stitch_right - stitch_left)

	return merge_img, 0 if shift[0] + shift_h < 0 else shift[0] + shift_h

def global_warping(img, drift):
	new_height = img.shape[0] - abs(drift)
	new_img = np.zeros((new_height, img.shape[1], img.shape[2]))
	
	# do global warping
	avg = drift / img.shape[1]
	if avg > 0:
		for col in range(img.shape[1]):
			new_img[:, col, :] = img[int(col * avg) : int(col * avg) + new_height, col,:]
	else:
		for col in range(img.shape[1]):
			new_img[:, col, :] = img[abs(drift) + int(col * avg) : abs(drift) + int(col * avg) + new_height, col, :] 
	return new_img