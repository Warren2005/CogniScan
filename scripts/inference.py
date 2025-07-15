# inference.py
import cv2
import numpy as np
import joblib
import time
from collections import deque
#import pyttsx3  # For TTS
# from openai import OpenAI  # Uncomment if you're using GPT feedback

# === CONFIG ===
MODEL_PATH = "/Users/almarai/Private Projects/CogniScan/CogniScan/models/svm_model.pkl"
ROLLING_WINDOW_SIZE = 10
PREDICTION_THRESHOLD = 0.6  # adjust as needed

# === Load your trained model ===
svm_model = joblib.load(MODEL_PATH)

# === Set up Text-to-Speech ===
#tts_engine = pyttsx3.init()

# def speak_message(message):
#     tts_engine.say(message)
#     tts_engine.runAndWait()

# === Optional: GPT-4 Prompt (pseudo-code) ===
# client = OpenAI()
# def get_gpt_feedback(prediction):
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a motivational productivity coach."},
#             {"role": "user", "content": f"My current state is {prediction}. Give me one short, actionable tip."}
#         ]
#     )
#     return response.choices[0].message.content.strip()

# === Rolling predictions for smoothing ===
rolling_preds = deque(maxlen=ROLLING_WINDOW_SIZE)

# === Capture video ===
cap = cv2.VideoCapture(0)
if not cap.isOpened(): 
    cap = cv2.VideoCapture(1)
if not cap.isOpened(): 
    raise IOError("Cannot open window")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Preprocess frame ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))  # Match your training dimensions
    flat = resized.flatten().reshape(1, -1)

    # === Make prediction ===
    pred = svm_model.predict(flat)[0]
    rolling_preds.append(pred)

    # === Smooth prediction ===
    final_pred = max(set(rolling_preds), key=rolling_preds.count)

    # === Overlay on frame ===
    cv2.putText(
        frame,
        f"State: {final_pred}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow('CogniScan - Focus Analyzer', frame)

    # === Example: TTS feedback ===
    # if rolling_preds.count("Stressed") > ROLLING_WINDOW_SIZE // 2:
    #     speak_message("You seem stressed. Take a deep breath and refocus.")
    #     # feedback = get_gpt_feedback("Stressed")
    #     # speak_message(feedback)
    #     time.sleep(5)  # Avoid spamming

    # === Break ===
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
