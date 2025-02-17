import numpy as np
import cv2 as cv
import argparse
import os

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation using multiple image frames.')
parser.add_argument('images', type=str, nargs='+', help='list of paths to image files')
args = parser.parse_args()

# Params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for Lucas-Kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for visualization
color = np.random.randint(0, 255, (100, 3))

# Read the first frame
first_image = args.images[0]
old_frame = cv.imread(first_image)
if old_frame is None:
    raise ValueError(f"Could not load image: {first_image}")

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing the optical flow
mask = np.zeros_like(old_frame)

# Process subsequent frames
for image_path in args.images[1:]:
    frame = cv.imread(image_path)
    if frame is None:
        print(f"Skipping invalid image: {image_path}")
        continue
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
        img = cv.add(frame, mask)
        cv.imshow('Optical Flow', img)
        cv.waitKey(0)

    # Wait for 'Esc' key to exit
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
