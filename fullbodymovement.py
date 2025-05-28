import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands, Face Detection, and Pose Detection models
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection()
pose = mp_pose.Pose()

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
    
    # Detect faces
    results_faces = face_detection.process(image_rgb)
    
    # Detect poses
    results_pose = pose.process(image_rgb)
    
    # Draw hand landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2))
            # Count fingers
            # Implement finger counting logic here
    
    # Draw nose and ears
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                   mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2))
        # Add text indicating human body
        cv2.putText(image, "This is a human body", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Hand, Face, Nose, Ears, and Contour Detection', image)
    
    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
