import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from collections import deque

# Custom DepthwiseConv2D to ignore the 'groups' parameter
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if "groups" in kwargs:
            kwargs.pop("groups")
        super(FixedDepthwiseConv2D, self).__init__(*args, **kwargs)

# Loading CSV dataset and training the k-NN classifier for digit recognition
df = pd.read_csv("dataset_numbers.csv")
X = df.iloc[:, :-1].values   # Landmark coordinates
y = df.iloc[:, -1].values    # Digit labels

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Loading the Teachable Machine Keras model and labels
custom_objects = {"DepthwiseConv2D": FixedDepthwiseConv2D}
model = tf.keras.models.load_model("converted_keras/keras_model.h5", custom_objects=custom_objects)

with open("converted_keras/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Setup smoothing over 5 frames for sign predictions
smoothing_queue = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Digit Prediction using k-NN
            landmark_points = []
            for lm in hand_landmarks.landmark:
                landmark_points.extend([lm.x, lm.y, lm.z])
            landmark_points = np.array(landmark_points).reshape(1, -1)
            
            if landmark_points.shape[1] == X.shape[1]:
                digit_prediction = knn.predict(landmark_points)[0]
            else:
                digit_prediction = "Unknown"
            
            # Hand Sign Meaning Prediction using Keras model
            # 1) Calculate bounding box around hand
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # 2) Add padding so the crop isn't too tight
            padding = 20
            min_x = max(min_x - padding, 0)
            max_x = min(max_x + padding, w)
            min_y = max(min_y - padding, 0)
            max_y = min(max_y + padding, h)
            
            # 3) Crop and preprocess the hand region
            hand_region = frame[min_y:max_y, min_x:max_x]
            target_size = (224, 224)  
            try:
                hand_img = cv2.resize(hand_region, target_size)
            except:
                # In case resize fails, fill with black
                hand_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            
            # 4) Normalize and predict
            hand_img = hand_img.astype(np.float32) / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)
            prediction_probs = model.predict(hand_img)[0]
            class_index = np.argmax(prediction_probs)
            sign_prediction = labels[class_index]
            
            # Smoothing the Sign Prediction
            smoothing_queue.append(sign_prediction)
            smoothed_prediction = max(set(smoothing_queue), key=smoothing_queue.count)
            
            # Remove a leading digit from the label (if present)
            tokens = smoothed_prediction.split()
            if tokens and tokens[0].isdigit():
                tokens = tokens[1:]
            cleaned_prediction = " ".join(tokens)

            # Displaying Results
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.putText(frame, f"Number : {digit_prediction}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Meaning : {cleaned_prediction}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
