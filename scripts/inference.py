import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import pyttsx3

MODEL_PATH = '/Users/almarai/Private Projects/CogniScan/CogniScan/models/cnn_model.h5'
ROLLING_WINDOW_SIZE = 10

# === Load model ===
cnn_model = tf.keras.models.load_model(MODEL_PATH)

# === TTS ===
tts_engine = pyttsx3.init()
def speak_message(msg):
    tts_engine.say(msg)
    tts_engine.runAndWait()

# === Rolling window smoothing ===
rolling_preds = deque(maxlen=ROLLING_WINDOW_SIZE)

# === Webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    norm = resized / 255.0
    input_img = norm.reshape(1, 48, 48, 1)

    pred_probs = cnn_model.predict(input_img, verbose=0)
    pred_class = np.argmax(pred_probs)

    class_names = ['Focus', 'Stressed', 'Neutral']
    label = class_names[pred_class]

    rolling_preds.append(label)
    final_pred = max(set(rolling_preds), key=rolling_preds.count)

    cv2.putText(frame, f"State: {final_pred}",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow('CogniScan - CNN Inference', frame)

    if rolling_preds.count('Stressed') > ROLLING_WINDOW_SIZE // 2:
        speak_message("You seem stressed. Take a short break.")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()