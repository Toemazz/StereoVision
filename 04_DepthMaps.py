import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('SampleImages/DepthMaps/tsukuba_l.png', 0)
imgR = cv2.imread('SampleImages/DepthMaps/tsukuba_r.png', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
