import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for hand gesture recognition
prev_pinch_state = False

# Object tracking variables
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
    
    # Draw hand landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Draw landmarks
            for lm in hand_landmarks.landmark:
                # Draw a small circle at each landmark
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            
            # Detect pinch gesture (thumb and index finger close together)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            thumb_x, thumb_y = int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])
            index_x, index_y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
            pinch_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
            if pinch_distance < 50:
                pinch_state = True
            else:
                pinch_state = False
            
            # Perform action based on pinch state change
            if pinch_state and not prev_pinch_state:
                print("Pinch gesture detected!")
            
            prev_pinch_state = pinch_state
    
    # Apply Canny edge detection
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
    cv2.imshow('Hand Gesture Recognition and Object Tracking', image)
    
    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
