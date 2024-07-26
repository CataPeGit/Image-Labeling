import cv2

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

# Read a frame from the camera
ret, frame = cap.read()

# Check if frame was read successfully
if ret:
    # Save the frame to a file
    cv2.imwrite('image.jpg', frame)
    print("Image captured and saved as image.jpg")
else:
    print("Error: Failed to capture image.")

# Release the camera
cap.release()
