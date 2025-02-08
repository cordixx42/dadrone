import cv2
import numpy as np

# Load the image
image = cv2.imread('../thermal_videos/flow_frames/frame20.jpg', cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding or edge detection to find contours
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours using cv2.findContours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Contours', image)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
