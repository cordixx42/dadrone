import numpy as np
import cv2 as cv
import argparse
import os
import re 

# Parse command line arguments
parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation using multiple image frames.')
parser.add_argument('folder', type=str, help='path to the folder containing the image files')
args = parser.parse_args()

# Params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for visualization
color = np.random.randint(0, 255, (100, 3))

# Get a sorted list of image files in the folder
image_files = [f for f in os.listdir(args.folder) if f.endswith('.jpg')]
# image_files.sort()  # Sort the images by name (adjust sorting if filenames don't ensure proper order)
image_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))

if len(image_files) == 0:
    raise ValueError("No .jpg files found in the specified folder.")

# Read the first frame
first_image = os.path.join(args.folder, image_files[0])
old_frame = cv.imread(first_image)
if old_frame is None:
    raise ValueError(f"Could not load image: {first_image}")

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing the optical flow
mask = np.zeros_like(old_frame)

# Process subsequent frames
for image_filename in image_files[1:]:
    image_path = os.path.join(args.folder, image_filename)
    print(image_filename)
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
