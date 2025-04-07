# Hand Gesture Interpreter System

This project implements a real-time hand gesture interpreter that not only detects hand landmarks using MediaPipe and OpenCV but also interprets hand gestures into numbers and their corresponding meanings. The system leverages a CSV dataset for digit recognition (using a k‑NN classifier) and a custom-trained Teachable Machine Keras model for gesture meaning interpretation.

## Features
- ⁠Real-time hand landmark detection using webcam
- Visualization of detected hand landmarks
- ⁠Digit recognition using a k‑NN classifier based on CSV dataset
- ⁠Gesture meaning interpretation using a Teachable Machine Keras model
- ⁠Smoothing of predictions to enhance reliability

## Requirements
- Python 3.6 or higher
- OpenCV
- Mediapipe
- NumPy
- TensorFlow
- scikit-learn

## Project Structure
- **dataset_numbers.csv** - Dataset file containing landmark coordinates and digit labels  
- **hand_landmark_detection.py** - Main script for real-time hand gesture interpretation  
- **requirements.txt** - List of Python dependencies for the project  
- **README.md** - Project documentation  
- **converted_keras/** - Directory containing the Teachable Machine Keras model and labels  
  - **keras_model.h5** - Trained Keras model file  
  - **labels.txt** - File containing the model's class labels

## Installation

1. Clone the repository:
git clone https://github.com/deepikaksr/hand_gesture_interpreter.git   
cd hand_gesture_interpreter

3. Set the dependencies:
pip install -r requirements.txt

## Usage
Run the hand landmark detection script using : python3 hand_landmark_detection.py

Press `q` to quit the webcam window.
