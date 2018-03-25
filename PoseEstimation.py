import cv2
import numpy as np
import glob


def draw(img, corners, img_pts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)  # X-axis
    img = cv2.line(img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)  # Y-axis
    img = cv2.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)  # Z-axis
    return img


# Define image format and chessboard dimensions
image_path_format = 'chessboard/leftsample/*.jpg'
load_path = 'LeftCameraInfo.npz'
dims = (7, 7)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_pts = np.zeros((dims[1]*dims[0], 3), np.float32)
obj_pts[:, :2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)

# Load camera calibration information
with np.load(load_path) as temp:
    cam_matrix = temp['cam_matrix']
    distortion_coeff = temp['distortion_coeff']

# Set up xyz axis positions
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

# Get list of image paths
images = glob.glob(image_path_format)

for img_name in images:
    img = cv2.imread(img_name)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img_g, (dims[0], dims[1]), None)

    # If found, add object points, image points (after refining them)
    if ret:
        # Refine the corner location
        corners2 = cv2.cornerSubPix(img_g, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        _, rotation_vecs, translation_vecs, inliers = cv2.solvePnPRansac(obj_pts, corners2, cam_matrix, distortion_coeff)

        # Project 3D points to image plane
        img_pts, jac = cv2.projectPoints(axis, rotation_vecs, translation_vecs, cam_matrix, distortion_coeff)

        img = draw(img, corners2, img_pts)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()
