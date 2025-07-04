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
        self.authentication_name = None

    def initUI(self):
        self.setWindowTitle("Gait Recognition System")
        self.setGeometry(100, 100, 800, 750)  # Increased height to accommodate delete button
        self.setStyleSheet("background-color: #e6f7ff;")

        self.label = QLabel(self)
        self.label.setText("Live Camera Feed")
        self.label.setFont(QFont("Arial", 16, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #004d99; background-color: #cceeff; padding: 10px; border-radius: 5px;")

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter name")
        self.name_input.setFont(QFont("Arial", 12))
        self.name_input.setStyleSheet("padding: 8px; border: 2px solid #004d99; border-radius: 5px;")

        self.upload_btn = QPushButton("Upload Video for Enrollment", self)
        self.upload_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.upload_btn.setStyleSheet("background-color: #ff9933; color: white; padding: 10px; border-radius: 5px;")
        self.upload_btn.clicked.connect(self.upload_video)

        self.auth_btn = QPushButton("Start Authentication", self)
        self.auth_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.auth_btn.setStyleSheet("background-color: #ff5050; color: white; padding: 10px; border-radius: 5px;")
        self.auth_btn.clicked.connect(self.start_live_auth)

        self.live_btn = QPushButton("Start Live Camera", self)
        self.live_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.live_btn.setStyleSheet("background-color: #3399ff; color: white; padding: 10px; border-radius: 5px;")
        self.live_btn.clicked.connect(self.start_live_camera)

        self.capture_btn = QPushButton("Capture Gait Feature for Enrollment", self)
        self.capture_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.capture_btn.setStyleSheet("background-color: #9933ff; color: white; padding: 10px; border-radius: 5px;")
        self.capture_btn.clicked.connect(self.capture_gait_feature)

        self.extract_btn = QPushButton("View All Stored Data", self)
        self.extract_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.extract_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.extract_btn.clicked.connect(self.extract_all_data)

        self.delete_btn = QPushButton("Delete Record", self)
        self.delete_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.delete_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px; border-radius: 5px;")
        self.delete_btn.clicked.connect(self.delete_record)

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

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.name_input)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.delete_btn)
        main_layout.addWidget(self.result_label)
        self.setLayout(main_layout)

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
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
            try:
                cursor.execute("INSERT INTO users (name, gait_features) VALUES (?, ?)", (name, str(gait_feature)))
                self.conn.commit()
                self.result_label.setText(f"Gait Data Successfully Enrolled for {name}!\nExtracted Feature:\n{gait_feature[:50]}...")
            except sqlite3.IntegrityError:
                QMessageBox.warning(self, "Error", f"User with name '{name}' already exists. Please use a different name.")
                self.conn.rollback()
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

    def delete_record(self):
        name_to_delete = self.name_input.text().strip()
        if not name_to_delete:
            QMessageBox.warning(self, "Error", "Please enter the name of the record to delete.")
            return

        reply = QMessageBox.question(
            self, "Confirmation", f"Are you sure you want to delete the record for '{name_to_delete}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM users WHERE name = ?", (name_to_delete,))
            self.conn.commit()
            if cursor.rowcount > 0:
                self.result_label.setText(f"Record for '{name_to_delete}' deleted successfully.")
            else:
                self.result_label.setText(f"No record found for '{name_to_delete}'.")
        # Clear the name input field after attempting deletion
        self.name_input.clear()

    def start_live_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open camera.")
            return
        self.timer.start(30)
        self.is_authenticating = False
        self.authentication_name = None
        self.result_label.setText("Live camera feed started.")

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
            QMessageBox.warning(self, "Error", "Please enter your name before capturing gait feature for enrollment.")
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
            try:
                cursor.execute("INSERT INTO users (name, gait_features) VALUES (?, ?)", (name, str(gait_feature)))
                self.conn.commit()
                self.result_label.setText(f"Gait feature captured for {name}!\nFeature:\n{gait_feature[:50]}...")
            except sqlite3.IntegrityError:
                QMessageBox.warning(self, "Error", f"User with name '{name}' already exists. Please use a different name.")
                self.conn.rollback()
        else:
            self.result_label.setText("No gait features detected. Please try again.")

    def euclidean_distance(self, feature1, feature2):
        """Calculates the Euclidean distance between two feature vectors."""
        return np.linalg.norm(np.array(feature1) - np.array(feature2))

    def authenticate_user(self, live_landmarks):
        """Authenticates the user based on live gait features."""
        if not live_landmarks:
            self.result_label.setText("No gait features detected in the current frame.")
            return

        live_features = (
            [landmark.x for landmark in live_landmarks.landmark] +
            [landmark.y for landmark in live_landmarks.landmark]
        )

        auth_name = self.name_input.text().strip()
        if not auth_name:
            self.result_label.setText("Please enter the name to authenticate against.")
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT gait_features FROM users WHERE name = ?", (auth_name,))
        result = cursor.fetchone()

        if result:
            stored_features_str = result[0]
            try:
                stored_features = eval(stored_features_str)
                distance = self.euclidean_distance(live_features, stored_features)
                threshold = 0.1  # Adjust this threshold as needed

                if distance < threshold:
                    self.result_label.setText(f"Authenticated as: {auth_name} (Similarity: {distance:.4f})")
                else:
                    self.result_label.setText(f"Authentication failed for {auth_name}. Similarity: {distance:.4f} (Threshold: {threshold:.4f})")
            except (SyntaxError, TypeError):
                self.result_label.setText(f"Error processing stored features for {auth_name}.")
        else:
            self.result_label.setText(f"User '{auth_name}' not found in the database.")

    def start_live_auth(self):
        """Starts the live authentication process."""
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Please start the live camera first.")
            return
        self.is_authenticating = True
        self.result_label.setText("Starting live authentication...")

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