import numpy as np
import cv2 as cv
import argparse
import os
import re


def mask_image(frame):
    # Define the red channel threshold and the green/blue thresholds
    red_threshold = 200
   
    # Create a mask for pixels where the red channel is high and the green/blue channels are lower
    mask = (frame[:, :, 2] > red_threshold) 
    
    # Convert all non-"very red" pixels to white (255, 255, 255)
    frame[~mask] = [255, 255, 255]
    return frame

# Parse command line arguments
parser = argparse.ArgumentParser(description='This sample demonstrates Dense Optical Flow calculation using multiple image frames.')
parser.add_argument('folder', type=str, help='path to the folder containing the image files')
args = parser.parse_args()

# Get a sorted list of image files in the folder
image_files = [f for f in os.listdir(args.folder) if f.endswith('.jpg')]
# Sort the images by numeric order (if the filenames are numbers)
image_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))

if len(image_files) == 0:
    raise ValueError("No .jpg files found in the specified folder.")

# Read the first frame
first_image = os.path.join(args.folder, image_files[0])
old_frame = cv.imread(first_image)
if old_frame is None:
    raise ValueError(f"Could not load image: {first_image}")
old_frame = mask_image(old_frame)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

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
    frame = mask_image(frame)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualize the flow by drawing arrows for each pixel
    hsv = np.zeros_like(frame)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents the flow direction
    hsv[..., 1] = 255  # Saturation is set to the maximum
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # Value represents the flow magnitude
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Draw motion vectors (arrows) on the image
    step = 10  # Step size to reduce the density of arrows
    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            # Get the flow vectors at each pixel
            flow_at_point = flow[y, x]
            fx, fy = flow_at_point[0], flow_at_point[1]
            # Scale the flow vectors for better visualization
            magnitude = np.sqrt(fx**2 + fy**2)
            if magnitude > 1:  # Only draw significant vectors
                cv.arrowedLine(frame, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.03)

    # Combine the original frame with the flow visualization
    img = cv.addWeighted(frame, 0.7, rgb, 0.3, 0)

    # Show the final image with arrows and flow
    cv.imshow('Dense Optical Flow with Vectors', img)
    cv.waitKey(0)
    
    # Wait for 'Esc' key to exit
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Update previous frame and points
    old_gray = frame_gray.copy()

cv.destroyAllWindows()
