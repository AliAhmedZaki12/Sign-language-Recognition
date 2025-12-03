# predict.py
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import sys
import time
import os

MODEL_FILE = "model.ASL"

# ----- تحميل الموديل -----
if not os.path.isfile(MODEL_FILE):
    print(f"ERROR: model file not found: {MODEL_FILE}. Run train_model.py first.")
    sys.exit(1)

with open(MODEL_FILE, "rb") as f:
    model_dict = pickle.load(f)
model = model_dict.get("model", None)
if model is None:
    print("ERROR: model not found inside the pickle file.")
    sys.exit(1)

# ----- إعداد Mediapipe -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)

# ----- دالة لاكتشاف أول كاميرا عاملة -----
def find_camera_index(max_test=4):
    for i in range(max_test + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return i
        else:
            cap.release()
    return None

cam_index = find_camera_index(3)
if cam_index is None:
    print("ERROR: No camera found. Connect a webcam and try again.")
    sys.exit(1)

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(cam_index)

# Text-to-speech
engine = pyttsx3.init()
word = ""
last_character = None
confirm_counter = 0
CONFIRM_THRESHOLD = 12  # عدد الفريمات المتتالية لتأكيد الحرف

print(f"Using camera index: {cam_index}. Press 's' to speak word, 'c' to clear, 'q' to quit.")

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        print("ERROR reading frame from camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # نأخذ اليد الأولى فقط
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        if x_ and y_:
            xmin = min(x_)
            ymin = min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - xmin)
                data_aux.append(lm.y - ymin)

        # bbox (محصور داخل حدود الصورة)
        try:
            x1 = max(int(min(x_) * W) - 10, 0)
            y1 = max(int(min(y_) * H) - 10, 0)
            x2 = min(int(max(x_) * W) + 10, W - 1)
            y2 = min(int(max(y_) * H) + 10, H - 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception:
            pass

        if len(data_aux) == 42:
            try:
                pred = model.predict([np.asarray(data_aux)])
                predicted_character = pred[0]
            except Exception as e:
                predicted_character = "ERROR"
                print("Prediction error:", e)

            # تأكيد الحرف لثبات التنبؤ
            if predicted_character == last_character:
                confirm_counter += 1
            else:
                confirm_counter = 0
            last_character = predicted_character

            if confirm_counter >= CONFIRM_THRESHOLD:
                if predicted_character == "space":
                    word += " "
                elif predicted_character == "del":
                    word = word[:-1]
                elif predicted_character not in ["nothing", "del", "space", "ERROR"]:
                    
                        word += predicted_character
                confirm_counter = 0

            cv2.putText(frame, str(predicted_character), (x1, max(y1 - 12, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # عرض الكلمة الحالية
    cv2.putText(frame, "Word: " + word, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)

    cv2.imshow("ASL Live (press q to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        if word.strip():
            engine.say(word)
            engine.runAndWait()
           
            word = ""
    elif key == ord("c"):
        word = ""

cap.release()
cv2.destroyAllWindows()
hands.close()
