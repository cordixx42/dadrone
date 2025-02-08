import cv2
import numpy as np
import os

# Function to load thermal images (JPG frames)
def load_images(image_folder):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                images.append(img)
    return images

# Function to compute optical flow
def compute_optical_flow(prev_gray, next_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Function to analyze the flow and estimate direction
def analyze_flow(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_angle = np.mean(angle)
    
    # Convert angle to degrees
    avg_angle_deg = np.rad2deg(avg_angle) % 360
    
    if 45 <= avg_angle_deg < 135:
        direction = "Upward"
    elif 135 <= avg_angle_deg < 225:
        direction = "Left"
    elif 225 <= avg_angle_deg < 315:
        direction = "Downward"
    else:
        direction = "Right"
    
    return direction, avg_angle_deg

# Main function to process the images
def detect_flow_direction(image_folder):
    images = load_images(image_folder)
    if len(images) < 2:
        print("Need at least two images to calculate flow.")
        return
    
    prev_gray = images[0]
    
    for i in range(1, len(images)):
        next_gray = images[i]
        
        # Compute optical flow
        flow = compute_optical_flow(prev_gray, next_gray)
        
        # Analyze flow
        direction, angle = analyze_flow(flow)
        
        # Display the direction and angle
        print(f"Frame {i}: Flow direction: {direction} (Average angle: {angle:.2f} degrees)")
        
        prev_gray = next_gray

# Set the folder containing the thermal JPG images
image_folder = "path_to_your_thermal_images"

# Run the detection
detect_flow_direction(image_folder)
