road lane dynamic
import cv2
import numpy as np

def road_lane_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough transform to detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is not None:
        # Draw detected lines on a copy of the original image
        line_img = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0,0,255), 2)

        # Combine the original image with the detected lines
        result = cv2.addWeighted(image, 0.8, line_img, 1, 0)

        return result
    else:
        return image  # Return the original image if no lines are detected

# Read the video input (you can replace 'input_video.mp4' with your video file)
cap = cv2.VideoCapture('C:/Users/lenovo/OneDrive/Desktop/Dbridge.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply lane detection to the current frame
    processed_frame = road_lane_detection(frame)

    # Display the result
    cv2.imshow('Lane Detection', processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
 
 
 
