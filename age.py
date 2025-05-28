import cv2
import dlib

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Open webcam
cap = cv2.VideoCapture(0)

# Main loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Dlib's face detector
    faces = detector(gray)

    # Draw bounding boxes around the detected faces and estimate age
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        
        # Empirically determined age-to-face-width ratio (e.g., 0.2)
        age_to_face_width_ratio = 0.2

        # Calculate estimated age based on face width
        estimated_age = int((w * age_to_face_width_ratio))

        # Display estimated age
        cv2.putText(frame, f'Estimated Age: {estimated_age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Age Estimation', frame)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
