import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for hand gesture recognition
prev_thumbs_up_state = False

# Main loop
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

    # Display output
    cv2.imshow('Hand Gesture Recognition', image)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
