import cv2 as cv
import numpy as np
# Import and apply the patch before importing mediapipe
import mediapipe as mp
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
import pickle

# Function to get user-friendly error messages - MOVED TO TOP
def get_error_message(error_type):
    messages = {
        "not_high": "Bar not pulled high enough",
        "not_low": "Not starting from full extension",
        "excessive_lean": "Excessive back lean",
        "excessive_elbow_flare": "Elbows flaring too much",
        "elbows_too_far": "Elbows too far forward/back"
    }
    return messages.get(error_type, error_type)

# Open a file dialog to select a video file
Tk().withdraw()
videoPath = askopenfilename()

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load trained models and scaler
def load_models():
    models = {}
    # List of possible error types
    error_types = [
        "not_high", "not_low", "excessive_lean", 
        "excessive_elbow_flare", "elbows_too_far"
    ]
    
    # Load scaler
    try:
        # First try to load from the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(script_dir, 'feature_scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            # Try current working directory
            scaler = joblib.load('feature_scaler.joblib')
        print("Loaded feature scaler successfully")
    except Exception as e:
        print(f"Warning: Scaler not found. Will create a new one. Error: {e}")
        scaler = StandardScaler()
    
    # Load each model
    for error_type in error_types:
        # Try loading neural network model
        try:
            # Try different possible locations
            model_paths = [
                f"{error_type}_model.h5",  # Current directory
                os.path.join(script_dir, f"{error_type}_model.h5"),  # Script directory
                os.path.join(os.path.dirname(script_dir), f"{error_type}_model.h5")  # Parent directory
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    models[error_type] = ('nn', load_model(path))
                    print(f"Loaded neural network model for {error_type} from {path}")
                    break
            else:
                # If no model found, try logistic regression
                logreg_paths = [
                    f"{error_type}_logreg.pkl",
                    os.path.join(script_dir, f"{error_type}_logreg.pkl"),
                    os.path.join(os.path.dirname(script_dir), f"{error_type}_logreg.pkl")
                ]
                
                for path in logreg_paths:
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            models[error_type] = ('logreg', pickle.load(f))
                        print(f"Loaded logistic regression model for {error_type} from {path}")
                        break
                else:
                    print(f"Warning: No model found for {error_type}. Prediction will not be available.")
                    
        except Exception as e:
            print(f"Error loading model for {error_type}: {e}")
    
    return models, scaler

# Load models
models, scaler = load_models()

# Function to calculate the angle (in degrees) between three points.
# Here, a, b, and c should be lists or arrays of [x, y].
def calculate_angle(a, b, c):
    a = np.array(a)  # Example: shoulder
    b = np.array(b)  # Vertex: elbow
    c = np.array(c)  # Example: wrist
    
    # Calculate vectors BA and BC
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

# Function to calculate an angle relative to the vertical direction.
# Given two 2D points (hip and shoulder), this returns the angle between the
# vector (shoulder - hip) and the vertical (0, 1) vector.
def calculate_trunk_angle(hip, shoulder):
    # Vector from hip to shoulder
    v = np.array(shoulder) - np.array(hip)
    vertical = np.array([0, 1])
    dot = np.dot(v, vertical)
    norm_v = np.linalg.norm(v)
    norm_vertical = np.linalg.norm(vertical)
    cosine_angle = dot / (norm_v * norm_vertical)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function to calculate the head angle using the positions of the left and right ears.
# The angle is computed relative to the horizontal. If the head is level, this should be near 0.
def calculate_head_angle(left_ear, right_ear):
    left_ear = np.array(left_ear)
    right_ear = np.array(right_ear)
    vector = right_ear - left_ear
    angle = np.degrees(np.arctan2(vector[1], vector[0]))
    return angle

# Function to analyze form using loaded models
def analyze_form(feature_vector, models, scaler):
    try:
        # Create feature array from current frame data
        features = np.array([feature_vector]).astype(np.float64)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Predict from each model
        predictions = {}
        for error_type, (model_type, model) in models.items():
            if model_type == 'nn':
                pred = model.predict(scaled_features, verbose=0)[0][0]
                predictions[error_type] = bool(pred > 0.5)
            else:  # logreg
                pred = model.predict(scaled_features)[0]
                predictions[error_type] = bool(pred)
        
        return predictions
    except Exception as e:
        print(f"Error in analyze_form: {e}")
        return {}  # Return empty dict on error

# Rep counting variables (using right arm movement)
repCount = 0
prev_angle = None
going_up = False
min_angle_threshold = 100  # Adjust this threshold as needed for rep segmentation

# Open the video capture
capture = cv.VideoCapture(videoPath)
if not capture.isOpened():
    print("Error: Could not open video")
    exit()

# Prepare CSV file for storing selected landmark data and computed angles
output_csv = 'output.csv'
csv_file = open(output_csv, 'w', newline='')
csv_writer = csv.writer(csv_file)

# CSV header: frame, rep, then landmarks for right_shoulder, right_elbow, right_wrist, right_hip,
# followed by computed angles: elbow_angle, head_angle, spine_angle.
header = ['frame', 'rep']
landmark_names = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip']
for name in landmark_names:
    header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_v'])
header.extend(['elbow_angle', 'head_angle', 'spine_angle'])
csv_writer.writerow(header)

frame_index = 0  # To track the current frame number
feature_buffer = []  # Store recent features for smoothing predictions

# Main loop: process video frame by frame
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Convert frame from BGR to RGB and process with MediaPipe
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = pose.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Get all landmarks for use in computations.
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
        # Extract key landmarks for right arm and right hip (for arm and trunk tracking)
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
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        ]

        # Extract additional landmarks for spine and head tracking
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        ]
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        ]
        left_ear = [
            landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR].y
        ]
        right_ear = [
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y
        ]

        # For drawing, convert normalized coordinates to pixel coordinates.
        h, w, _ = frame.shape
        def denormalize(point):
            return (int(point[0] * w), int(point[1] * h))
        pts = {
            'r_shoulder': denormalize(right_shoulder),
            'r_elbow': denormalize(right_elbow),
            'r_wrist': denormalize(right_wrist),
            'r_hip': denormalize(right_hip),
            'l_ear': denormalize(left_ear),
            'r_ear': denormalize(right_ear)
        }
        # Draw circles for the key landmarks.
        cv.circle(frame, pts['r_shoulder'], 5, (255, 0, 0), -1)
        cv.circle(frame, pts['r_elbow'], 5, (0, 255, 0), -1)
        cv.circle(frame, pts['r_wrist'], 5, (0, 0, 255), -1)
        cv.circle(frame, pts['r_hip'], 5, (255, 255, 0), -1)
        cv.circle(frame, pts['l_ear'], 5, (255, 0, 255), -1)
        cv.circle(frame, pts['r_ear'], 5, (0, 255, 255), -1)

        # Compute the elbow angle for the right arm.
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        cv.putText(frame, f"Elbow Angle: {int(elbow_angle)}", 
                   (10, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 0), 8, cv.LINE_AA)
        cv.putText(frame, f"Elbow Angle: {int(elbow_angle)}", 
                   (10, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv.LINE_AA)
        
        # Compute head angle using left ear and right ear.
        head_angle = calculate_head_angle(left_ear, right_ear)
        cv.putText(frame, f"Head Angle: {int(head_angle)}", 
                   (10, 80), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 0), 8, cv.LINE_AA)
        cv.putText(frame, f"Head Angle: {int(head_angle)}", 
                   (10, 80), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv.LINE_AA)

        # Compute spine angle using midpoints of shoulders and hips.
        mid_shoulder = [ (left_shoulder[0] + right_shoulder[0]) / 2,
                         (left_shoulder[1] + right_shoulder[1]) / 2 ]
        mid_hip = [ (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2 ]
        spine_angle = calculate_trunk_angle(mid_hip, mid_shoulder)
        cv.putText(frame, f"Spine Angle: {int(spine_angle)}", 
                   (10, 110), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 0), 8, cv.LINE_AA)
        cv.putText(frame, f"Spine Angle: {int(spine_angle)}", 
                   (10, 110), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv.LINE_AA)

        # Rep counting logic using the elbow angle (for the right arm).
        if prev_angle is not None:
            if elbow_angle > prev_angle:
                if not going_up and elbow_angle < min_angle_threshold:
                    going_up = True
            elif elbow_angle < prev_angle:
                if going_up and elbow_angle > min_angle_threshold:
                    going_up = False
                    repCount += 1
        prev_angle = elbow_angle
        
        stage = "up" if going_up else "down"
        cv.putText(frame, f"Stage: {stage}", 
                   (10, 140), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 0), 8, cv.LINE_AA)
        cv.putText(frame, f"Stage: {stage}", 
                   (10, 140), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, f"Reps: {repCount}", 
                   (10, 170), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 0), 8, cv.LINE_AA)
        cv.putText(frame, f"Reps: {repCount}", 
                   (10, 170), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv.LINE_AA)
        
        # Build a feature vector for the selected key landmarks:
        # right_shoulder, right_elbow, right_wrist, and right_hip.
        selected_landmarks = [
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.RIGHT_HIP
        ]
        feature_vector = []
        for lm_enum in selected_landmarks:
            lm = landmarks[lm_enum]
            feature_vector.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # Add computed angles to feature vector
        feature_vector.extend([elbow_angle, head_angle, spine_angle])
        
        # Store feature in buffer
        if len(feature_buffer) >= 5:  # Keep last 5 frames
            feature_buffer.pop(0)
        feature_buffer.append(feature_vector)
        
        # Only analyze form if we have enough frames and models are loaded
        if len(feature_buffer) >= 3 and models:
            # Use average of last few frames for smoother predictions
            avg_features = np.mean(feature_buffer, axis=0)
            
            # Analyze form
            form_errors = analyze_form(avg_features[0:15], models, scaler)  # First 15 elements match training features
            
            # Display form errors on frame
            y_pos = 200
            for error_type, is_error in form_errors.items():
                if is_error:
                    error_message = get_error_message(error_type)
                    cv.putText(frame, error_message, 
                              (10, y_pos), 
                              cv.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 0), 8, cv.LINE_AA)
                    cv.putText(frame, error_message, 
                              (10, y_pos), 
                              cv.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2, cv.LINE_AA)
                    y_pos += 30
        
        # Write current frame data to CSV:
        # Format: [frame_index, repCount, <landmark features>, elbow_angle, head_angle, spine_angle]
        csv_row = [frame_index, repCount] + feature_vector
        csv_writer.writerow(csv_row)
    
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

capture.release()
cv.destroyAllWindows()
csv_file.close()

# Save scaler for future use
joblib.dump(scaler, 'feature_scaler.joblib')
