import cv2
import numpy as np

img_left = cv2.imread('test/left/frame10.jpg', 0)
img_right = cv2.imread('test/right/frame10.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img_left, img_right)
disparity = cv2.convertScaleAbs(disparity)
cv2.imwrite('out.png', disparity)
cv2.imshow('gray', disparity)
cv2.waitKey(0)
