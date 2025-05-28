import cv2
import numpy as np
import mediapipe as mp
import dlib

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variables for hand gesture recognition
prev_thumbs_up_state = False
prev_pinch_state = False
prev_contour_center = None

while True:
    # Read frame from webcam
    success, image = cap.read()
    if not success:
        break

    # Flip the frame horizontally for selfie-view display
    image = cv2.flip(image, 1)

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hands
    results_hands = hands.process(image_rgb)

    # Draw hand landmarks and detect thumbs-up gesture
    thumbs_up_detected = False
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Draw landmarks
            for lm in hand_landmarks.landmark:
                # Draw a small circle at each landmark
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

            # Detect thumbs-up gesture (thumb pointing up)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_y = thumb_tip.y * image.shape[0]
            if thumb_y < hand_landmarks.landmark[5].y * image.shape[0]:  # Comparing with the y-coordinate of the base of the thumb
                thumbs_up_detected = True

    # Perform action based on thumbs-up state change
    if thumbs_up_detected and not prev_thumbs_up_state:
        print("Thumbs-up gesture detected!")
        # Display "Hello" text
        cv2.putText(image, "Hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    prev_thumbs_up_state = thumbs_up_detected

    # Detect hand gesture (pinch)
    pinch_state = False
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Detect pinch gesture (thumb and index finger close together)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            thumb_x, thumb_y = int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])
            index_x, index_y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
            pinch_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
            if pinch_distance < 50:
                pinch_state = True
                break

    # Perform action based on pinch state change
    if pinch_state and not prev_pinch_state:
        print("Pinch gesture detected!")
    
    prev_pinch_state = pinch_state

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using Dlib's face detector
    faces = detector(gray)

    # Draw bounding boxes around the detected faces and estimate age
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Empirically determined age-to-face-width ratio (e.g., 0.2)
        age_to_face_width_ratio = 0.2

        # Calculate estimated age based on face width
        estimated_age = int((w * age_to_face_width_ratio))

        # Display estimated age
        cv2.putText(image, f'Estimated Age: {estimated_age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Apply background subtraction to detect head movement
    fg_mask = bg_subtractor.apply(image)
    _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Apply Canny edge detection for object tracking
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    
    # Perform object tracking
    if largest_contour is not None:
        # Get the center of the largest contour
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            contour_center = (cx, cy)
            
            # Track the movement of the contour
            if prev_contour_center is not None:
                cv2.line(image, prev_contour_center, contour_center, (0, 255, 0), 2)
            
            prev_contour_center = contour_center

    # Display output
    cv2.imshow('Merged Functionality', image)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
