# Lat pull-down Form Analyzer
A computer vision application that analyzes pull-up form using pose estimation and machine learning to identify common errors and count repetitions.

## Features

- Real-time form analysis using computer vision and machine learning
- Automatic rep counting to track workout progress
- Form error detection for five common issues:
  - Bar not pulled high enough
  - Not starting from full extension
  - Excessive back lean
  - Excessive elbow flare
  - Elbows positioned too far forward/back
- Angle tracking for key joints (elbow, head, spine)
- Data export to CSV for further analysis

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- TensorFlow
- scikit-learn
- NumPy
- tkinter

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pullup-form-analyzer.git
   cd pullup-form-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe tensorflow scikit-learn numpy
   ```

3. Download pre-trained models or train your own (see Training section).


## Usage

1. Run the main script:
2. Select a video file in the dialog that appears.
3. The application will:
  -  Display the video with pose landmarks
  -  Count repetitions
  -  Show angles for elbow, head, and spine
  -  Identify form errors in real-time
  -  Export data to output.csv

## Training Your Own Models

The application uses machine learning models to detect form errors. Pre-trained models should be named:

- `{error_type}_model.h5` for neural network models
- `{error_type}_logreg.pkl` for logistic regression models

Where `{error_type}` is one of: "not_high", "not_low", "excessive_lean", "excessive_elbow_flare", "elbows_too_far"
