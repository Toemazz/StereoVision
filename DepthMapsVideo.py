import cv2
from tqdm import tqdm

# Set up video capture
left_video = cv2.VideoCapture('LeftActual.mp4')
right_video = cv2.VideoCapture('RightActual.mp4')

# Get information about the videos
n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)),
               int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))
fps = int(left_video.get(cv2.CAP_PROP_FPS))

for _ in tqdm(range(n_frames)):
    # Grab the frames from their respective video streams
    ok, left = left_video.read()
    _, right = right_video.read()

    if ok:
        # Convert from 'BGR' to 'RGB'
        # left, right = left[..., ::-1], right[..., ::-1]
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(gray_left, gray_right)

        # Disparity needs to be converted to match the format of the camera video feeds
        disparity = cv2.convertScaleAbs(disparity)
        cv2.imshow('Output', disparity)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
