import cv2
import numpy as np

def mask_image(frame):
    # Define the red channel threshold and the green/blue thresholds
    red_threshold = 200
   
    # Create a mask for pixels where the red channel is high and the green/blue channels are lower
    mask = (frame[:, :, 2] > red_threshold) 
    #& (image[:, :, 1] < green_blue_threshold) & (image[:, :, 0] < green_blue_threshold)
    
    # Convert all non-"very red" pixels to white (255, 255, 255)
    frame[~mask] = [255, 255, 255]
    return frame

# Replace 'thermo_video.mp4' with your video file path
cap = cv2.VideoCapture("../thermal_videos/flow_cropped.mp4")


# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error reading video")
    cap.release()
    exit()

#frame1 = mask_image(frame1)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
height, width = prvs.shape

# Scaling factor to lengthen the arrows
arrow_scale = 5
arrow_directions = []  # List to store the flow vector directions

red_threshold = 200

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    
    #frame2 = mask_image(frame2)
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, 
                                        None, 
                                        pyr_scale=0.5, 
                                        levels=3, 
                                        winsize=15, 
                                        iterations=3, 
                                        poly_n=5, 
                                        poly_sigma=1.2, 
                                        flags=0)

    vis = frame2.copy()
    step = 16  

    for y in range(0, height, step):
        for x in range(0, width, step):
            if frame2[y, x, 2] < red_threshold:
                continue
            fx, fy = flow[y, x]
            # Multiply the flow vector by arrow_scale to make it longer
            cv2.arrowedLine(vis, (x, y), 
                            (int(x + arrow_scale * fx), int(y + arrow_scale * fy)), 
                            (0, 255, 0), 1, tipLength=1.0)
            
            arrow_directions.append((fx, fy))

    if arrow_directions:
        avg_fx = np.mean([fx for fx, fy in arrow_directions])
        avg_fy = np.mean([fy for fx, fy in arrow_directions])
        print("Average arrow direction: x:", avg_fx * arrow_scale, " y: ", avg_fy * arrow_scale)
    else:
        print("No flow data was computed.")

    
    cv2.imshow("Fluid Flow Vectors", vis)
    
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
    
    prvs = next_frame

cap.release()
cv2.destroyAllWindows()
