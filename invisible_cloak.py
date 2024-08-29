import cv2
import numpy as np

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Load the background image
background = cv2.imread('./image.jpg')

# Check if background image is loaded successfully
if background is None:
    print("Error: Could not load background image.")
    exit()

while cap.isOpened():
    # Read a frame from the camera
    ret, current_frame = cap.read()
    
    if ret:
        # Resize current_frame to match background dimensions
        current_frame = cv2.resize(current_frame, (background.shape[1], background.shape[0]))
        
        # Convert the current frame to HSV color space
        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        
        # Define the range for detecting the red color in HSV
        l_red = np.array([0, 120, 70])
        u_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, l_red, u_red)
        
        l_red = np.array([170, 120, 70])
        u_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_frame, l_red, u_red)
        
        # Combine the two masks to get the final red mask
        red_mask = mask1 + mask2
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=10)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # Ensure red_mask is uint8
        red_mask = red_mask.astype(np.uint8)
        
        # Apply the mask
        part1 = cv2.bitwise_and(background, background, mask=red_mask)
        
        red_free = cv2.bitwise_not(red_mask)
        
        part2 = cv2.bitwise_and(current_frame, current_frame, mask = red_free)
        
        # Show the red mask in a window titled "red cloak"
        cv2.imshow("red cloak", part1+part2)
        
        # Save the background image and exit the loop when 'q' is pressed
        if cv2.waitKey(5) == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()