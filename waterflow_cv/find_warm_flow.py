import cv2
import numpy as np

def show_modified_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Define the red channel threshold and the green/blue thresholds
    red_threshold = 200
   
    # Create a mask for pixels where the red channel is high and the green/blue channels are lower
    mask = (image[:, :, 2] > red_threshold) 
    #& (image[:, :, 1] < green_blue_threshold) & (image[:, :, 0] < green_blue_threshold)
    
    # Convert all non-"very red" pixels to white (255, 255, 255)
    image[~mask] = [255, 255, 255]
    
    # Show the image in a window
    cv2.imshow('Modified Image', image)
    
    # Wait for any key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
show_modified_image('../thermal_videos/flow_frames/frame4.jpg')
