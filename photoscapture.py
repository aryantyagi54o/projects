import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize variable to track hand state
prev_hand_raised = False

# Main loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for selfie-view display
    frame = cv2.flip(frame, 1)

    # Display output
    cv2.imshow('Hand Gesture Recognition', frame)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to segment objects from the background
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if hand is raised
    hand_raised = False
    for contour in contours:
        # Calculate area of contour
        area = cv2.contourArea(contour)

        # If the contour area is above a certain threshold, consider it as a hand
        if area > 2000:  # Adjust this threshold according to your environment
            hand_raised = True
            break

    # Check for hand state change (from not raised to raised)
    if hand_raised and not prev_hand_raised:
        # Capture image
        cv2.imwrite('hand_raised_photo.jpg', frame)
        print("Photo captured!")

    # Update previous hand state
    prev_hand_raised = hand_raised

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
