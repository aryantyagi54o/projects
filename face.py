import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands and Face Detection models
hands = mp_hands.Hands()
face_detection = mp_face_detection.FaceDetection()

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
    
    # Display output
    cv2.imshow('Hand and Face Detection', image)
    
    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()