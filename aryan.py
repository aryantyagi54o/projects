import sys
import cv2
import numpy as np
import sqlite3
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit,
    QMessageBox, QFileDialog, QTextEdit, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class GaitRecognitionApp(QWidget):
    def __init__(self):  # ✅ Fixed __init__ constructor
        super().__init__()
        self.initUI()
        self.conn = sqlite3.connect("gait_data.db")
        self.create_table()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.is_authenticating = False

    def initUI(self):
        self.setWindowTitle("Gait Recognition System")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("background-color: #e6f7ff;")
        
        self.label = QLabel(self)
        self.label.setText("Live Camera Feed")
        self.label.setFont(QFont("Arial", 16, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #004d99; background-color: #cceeff; padding: 10px; border-radius: 5px;")
        
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter your name")
        self.name_input.setFont(QFont("Arial", 12))
        self.name_input.setStyleSheet("padding: 8px; border: 2px solid #004d99; border-radius: 5px;")
        
        self.upload_btn = QPushButton("Upload Video for Enrollment", self)
        self.upload_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.upload_btn.setStyleSheet("background-color: #ff9933; color: white; padding: 10px; border-radius: 5px;")
        self.upload_btn.clicked.connect(self.upload_video)

        self.auth_btn = QPushButton("Authenticate", self)
        self.auth_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.auth_btn.setStyleSheet("background-color: #ff5050; color: white; padding: 10px; border-radius: 5px;")
        self.auth_btn.clicked.connect(self.start_live_auth)
        
        self.live_btn = QPushButton("Start Live Camera", self)
        self.live_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.live_btn.setStyleSheet("background-color: #3399ff; color: white; padding: 10px; border-radius: 5px;")
        self.live_btn.clicked.connect(self.start_live_camera)

        self.capture_btn = QPushButton("Capture Gait Feature", self)
        self.capture_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.capture_btn.setStyleSheet("background-color: #9933ff; color: white; padding: 10px; border-radius: 5px;")
        self.capture_btn.clicked.connect(self.capture_gait_feature)
        
        self.extract_btn = QPushButton("View All Stored Data", self)
        self.extract_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.extract_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.extract_btn.clicked.connect(self.extract_all_data)
        
        self.result_label = QTextEdit(self)
        self.result_label.setReadOnly(True)
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setStyleSheet("padding: 10px; background-color: #fff; border: 1px solid #004d99; border-radius: 5px;")
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.auth_btn)
        button_layout.addWidget(self.live_btn)
        button_layout.addWidget(self.capture_btn)
        button_layout.addWidget(self.extract_btn)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.name_input)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.result_label)
        self.setLayout(self.layout)
    
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                gait_features TEXT
            )
        """)
        self.conn.commit()
    
    def upload_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options
        )
        if file_path:
            self.process_video(file_path)
    
    def process_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open video file.")
            return
        
        features = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = (
                    [landmark.x for landmark in results.pose_landmarks.landmark] +
                    [landmark.y for landmark in results.pose_landmarks.landmark]
                )
                features.append(landmarks)
        cap.release()
        
        if features:
            gait_feature = np.mean(features, axis=0).tolist()
            name = self.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Please enter a name before enrolling.")
                return
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO users (name, gait_features) VALUES (?, ?)", (name, str(gait_feature)))
            self.conn.commit()
            self.result_label.setText(f"Gait Data Successfully Enrolled for {name}!\nExtracted Feature:\n{gait_feature}")
        else:
            self.result_label.setText("No gait features detected in the video.")
    
    def extract_all_data(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users")
        records = cursor.fetchall()
        
        if records:
            result_text = "Stored Gait Data:\n"
            for row in records:
                user_id, name, gait_features = row
                result_text += f"ID: {user_id}, Name: {name}, Gait Features: {gait_features[:50]}...\n"
            self.result_label.setText(result_text)
        else:
            self.result_label.setText("No data found in the database.")
    
    def start_live_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open camera.")
            return
        self.timer.start(30)
    
    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if self.is_authenticating:
                    self.authenticate_user(results.pose_landmarks)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))
    
    def capture_gait_feature(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter your name before capturing gait feature.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open camera for capturing.")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.warning(self, "Error", "Failed to capture frame from camera.")
            return
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            gait_feature = (
                [landmark.x for landmark in results.pose_landmarks.landmark] +
                [landmark.y for landmark in results.pose_landmarks.landmark]
            )
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO users (name, gait_features) VALUES (?, ?)", (name, str(gait_feature)))
            self.conn.commit()
            self.result_label.setText(f"Gait feature captured for {name}!\nFeature:\n{gait_feature}")
        else:
            self.result_label.setText("No gait features detected. Please try again.")
    
    def authenticate_user(self, landmarks):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, gait_features FROM users")
        records = cursor.fetchall()
        
        if not records:
            self.result_label.setText("No users found in the database.")
            return
        
        live_features = (
            [landmark.x for landmark in landmarks.landmark] +
            [landmark.y for landmark in landmarks.landmark]
        )
        
        for name, stored_features in records:
            stored_features = eval(stored_features)
            similarity = np.linalg.norm(np.array(live_features) - np.array(stored_features))
            if similarity < 0.1:  # Threshold
                self.result_label.setText(f"Authenticated as {name}!")
                return
        
        self.result_label.setText("Authentication failed. No match found.")
    
    def start_live_auth(self):
        self.is_authenticating = True
        self.result_label.setText("Authenticating user...")
    
    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.conn.close()
        event.accept()

# ✅ Entry point - fixed __name__ typo
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GaitRecognitionApp()
    window.show()
    sys.exit(app.exec_())
