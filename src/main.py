import cv2 as cv
import numpy as np
import mediapipe as mp
from tkinter import Tk   
from tkinter.filedialog import askopenfilename


Tk().withdraw()
videoPath = askopenfilename()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1, 
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) 
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist
    
    # Vectors BA and BC
    ba = a - b
    bc = c - b
    
    # Compute the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

repCount = 0
stage = None

capture = cv.VideoCapture(videoPath)

if not capture.isOpened():
    print("Error: Could not open video")
    exit()

while capture.isOpened():
    ret, frame = capture.read()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = pose.process(frame)
    frame.flags.writeable = True
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark
        landmark_array = []

        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        ]
        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        ]
        right_wrist = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        ]

        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        cv.putText(frame, f"Elbow Angle: {int(angle)}", 
                    (50, 50), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        if angle < 90:
            stage = "down"
        elif angle > 150 and stage == "down":
            stage = "up"
            repCount += 1
        cv.putText(frame, f"Reps: {repCount}", 
                (50, 100), 
                cv.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

        for lm in landmarks:
            landmark_array.extend([lm.x, lm.y, lm.z, lm.visibility])
        feature_vector = np.array(landmark_array, dtype=np.float32)
    else:
        landmarks = None

    
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if not ret:
        break

capture.release()
cv.destroyAllWindows()
