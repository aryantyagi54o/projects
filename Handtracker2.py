import cv2
import mediapipe as mp
import time


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

prev_time = 0  

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    
    image = cv2.flip(image, 1)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    pose_results = pose.process(image_rgb)
    hand_results = hands.process(image_rgb)

    
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   
    cv2.imshow('Pose and Hand Tracking', image)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
