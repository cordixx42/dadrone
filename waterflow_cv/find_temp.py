import cv2
import numpy as np

# Set real-world conversion factors (adjust based on experiment setup)
PIXEL_TO_METER = 0.001  # Example: each pixel represents 1 mm (adjust as needed)
FPS = 30  # Frames per second (modify according to video properties)

def compute_temperature_dissipation(prev_frame, current_frame):
    """Compute the average change in red intensity between frames to estimate heat dissipation."""
    red_channel_prev = prev_frame[:, :, 2]  # Extract red channel
    red_channel_curr = current_frame[:, :, 2]

    # Compute intensity difference
    intensity_change = np.abs(red_channel_curr.astype(int) - red_channel_prev.astype(int))
    avg_dissipation = np.mean(intensity_change)  # Average change in red intensity
    return avg_dissipation

# Load video
cap = cv2.VideoCapture("../thermal_videos/flow_cropped.mp4")
ret, frame1 = cap.read()
if not ret:
    print("Error reading video")
    cap.release()
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
height, width = prvs.shape

arrow_scale = 5
red_threshold = 200

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    vis = frame2.copy()
    step = 16  
    velocities = []

    for y in range(0, height, step):
        for x in range(0, width, step):
            if frame2[y, x, 2] < red_threshold:  # Ignore non-hot regions
                continue
            fx, fy = flow[y, x]

            # Convert displacement to velocity (real-world units)
            velocity_x = (fx * PIXEL_TO_METER * FPS)
            velocity_y = (fy * PIXEL_TO_METER * FPS)
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
            velocities.append(velocity_magnitude)

            # Draw flow arrows
            cv2.arrowedLine(vis, (x, y), 
                            (int(x + arrow_scale * fx), int(y + arrow_scale * fy)), 
                            (0, 255, 0), 1, tipLength=1.0)

    avg_velocity = np.mean(velocities) if velocities else 0
    avg_dissipation = compute_temperature_dissipation(frame1, frame2)

    print(f"Avg Velocity: {avg_velocity:.2f} m/s | Temp Dissipation: {avg_dissipation:.2f}")

    cv2.imshow("Fluid Flow & Temperature", vis)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
    
    prvs = next_frame
    frame1 = frame2  # Update frame for temperature dissipation analysis

cap.release()
cv2.destroyAllWindows()
