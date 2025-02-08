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

# Function to analyze the flow and return the dominant flow vector
def analyze_flow(flow):
    # Calculate the flow's horizontal (u) and vertical (v) components
    u = flow[..., 0]
    v = flow[..., 1]
    
    # Calculate the average flow vector in both directions (x and y components)
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    
    # Calculate the magnitude and direction (angle) of the dominant vector
    magnitude = np.sqrt(avg_u**2 + avg_v**2)
    angle = np.arctan2(avg_v, avg_u)  # Angle of the dominant flow vector
    
    # Convert angle to degrees
    angle_deg = np.rad2deg(angle) % 360
    
    return (avg_u, avg_v), magnitude, angle_deg

# Main function to process the images
def detect_flow_vector(image_folder):
    images = load_images(image_folder)
    if len(images) < 2:
        print("Need at least two images to calculate flow.")
        return
    
    prev_gray = images[0]
    
    for i in range(1, len(images)):
        next_gray = images[i]
        
        # Compute optical flow
        flow = compute_optical_flow(prev_gray, next_gray)
        
        # Analyze flow and get the dominant flow vector
        flow_vector, magnitude, angle = analyze_flow(flow)
        
        # Display the dominant flow vector, its magnitude, and direction
        print(f"Frame {i}: Flow vector: ({flow_vector[0]:.2f}, {flow_vector[1]:.2f})")
        print(f"Frame {i}: Magnitude of flow: {magnitude:.2f}")
        print(f"Frame {i}: Direction (angle): {angle:.2f} degrees")
        
        prev_gray = next_gray

# Set the folder containing the thermal JPG images
image_folder = "../thermal_videos/flow_frames"

# Run the detection
detect_flow_vector(image_folder)
