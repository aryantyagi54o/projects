import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

    # Check if hand is raised
    hand_raised = False
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Get the highest point of the hand (top of the palm)
            highest_point = min(hand_landmarks.landmark, key=lambda lm: lm.y)
            highest_point_y = highest_point.y * image.shape[0]

            # Check if the highest point is above a certain threshold (e.g., 0.2)
            if highest_point_y < 0.2 * image.shape[0]:
                hand_raised = True
                break

    # Display "Hello" text if hand is raised
    if hand_raised:
        cv2.putText(image, "Hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Hand Gesture Recognition', image)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
