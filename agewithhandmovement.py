import cv2
import mediapipe as mp
import dlib  # Import dlib for face detection and age estimation

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands, Face Detection, and Pose Detection models
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection()
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

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

    # Detect faces
    results_faces = face_detection.process(image_rgb)

    # Detect poses
    results_pose = pose.process(image_rgb)

    # Draw hand landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw face bounding boxes
    if results_faces.detections:
        for detection in results_faces.detections:
            bbox_c = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                   int(bbox_c.width * iw), int(bbox_c.height * ih)
            cv2.rectangle(image, bbox, (255, 0, 255), 2)

    # Draw nose and ears
    if results_pose.pose_landmarks:
        nose_landmark = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        ear_landmark_left = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        ear_landmark_right = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        h, w, c = image.shape
        x_nose, y_nose = int(nose_landmark.x * w), int(nose_landmark.y * h)
        x_ear_left, y_ear_left = int(ear_landmark_left.x * w), int(ear_landmark_left.y * h)
        x_ear_right, y_ear_right = int(ear_landmark_right.x * w), int(ear_landmark_right.y * h)
        cv2.circle(image, (x_nose, y_nose), 5, (255, 0, 0), -1)
        cv2.circle(image, (x_ear_left, y_ear_left), 5, (255, 0, 0), -1)
        cv2.circle(image, (x_ear_right, y_ear_right), 5, (255, 0, 0), -1)
        # Add text indicating human body
        cv2.putText(image, "This is a human body", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Detect faces using Dlib's face detector and estimate age
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Empirically determined age-to-face-width ratio (e.g., 0.2)
        age_to_face_width_ratio = 0.2

        # Calculate estimated age based on face width
        estimated_age = int((w * age_to_face_width_ratio))

        # Display estimated age
        cv2.putText(image, f'Estimated Age: {estimated_age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Combined Detection', image)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
