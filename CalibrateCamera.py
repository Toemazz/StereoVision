import cv2
import glob
import numpy as np

# Define image format and chessboard dimensions
image_path_format = 'images/calibration/leftsample/*.png'
save_path = 'data/CameraInfoLeft.npz'
chessboard_size = (7, 7)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_pts = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
obj_pts[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points (3D) and image points (2D) from all the images.
obj_points = []
img_points = []

# Get list of image paths
images = glob.glob(image_path_format)

for img_name in images:
    img = cv2.imread(img_name)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img_g, (chessboard_size[0], chessboard_size[1]), None)

    # If found, add object points, image points (after refining them)
    if ret:
        obj_points.append(obj_pts)

        # Refine the corner location
        corners2 = cv2.cornerSubPix(img_g, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibrate camera using both 2D and 3D points
ret, cam_matrix, distortion_coeff, rotation_vecs, translation_vecs = cv2.calibrateCamera(obj_points,
                                                                                         img_points,
                                                                                         img_g.shape[::-1],
                                                                                         None,
                                                                                         None)

# Save camera calibration information for later
np.savez(save_path,
         cam_matrix=cam_matrix,
         distortion_coeff=distortion_coeff,
         rotation_vecs=rotation_vecs,
         translation_vecs=translation_vecs)

# Calculate the re-projection error
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rotation_vecs[i], translation_vecs[i],
                                       cam_matrix, distortion_coeff)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error

print('Total Error: {:.4f}%'.format((100*mean_error) / len(obj_points)))


img = cv2.imread('images/calibration/leftsample/frame1.png')
h,  w = img.shape[:2]

new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, distortion_coeff, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, cam_matrix, distortion_coeff, None, new_cam_matrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('undistorted_left.png', dst)
