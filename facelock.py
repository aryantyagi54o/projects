import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

# Check if the classifier loaded successfully
if face_cascade.empty():
    print("Error: Unable to load the face cascade classifier.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize variables
locked = False
unlock_counter = 0
unlock_threshold = 30  # Number of frames to detect screen unlock

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If faces are detected, screen is considered unlocked
    if len(faces) > 0:
        unlock_counter = 0
        locked = False
    else:
        unlock_counter += 1
        if unlock_counter >= unlock_threshold:
            locked = True

    # Display lock/unlock status on the frame
    if locked:
        cv2.putText(frame, "Screen Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Screen Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Screen Lock Detection', frame)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
