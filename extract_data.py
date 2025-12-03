# extract_data.py
import os
import pickle
import cv2
import mediapipe as mp

# ----- إعداد -----
DATA_DIR = r"E:\Ali Zaki CV\ASL Project\asl_alphabet_train"
OUT_FILE = "data.ASL"
MAX_PER_CLASS = 200  # يمكنك تغييره

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

data = []
labels = []

# ----- المرور على الفولدرات -----
if not os.path.isdir(DATA_DIR):
    raise SystemExit(f"ERROR: DATA_DIR not found: {DATA_DIR}")

for dir_name in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, dir_name)
    if not os.path.isdir(class_dir):
        continue

    all_images = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not all_images:
        print(f" no images in {class_dir}, skipping.")
        continue

    img_list = all_images[:MAX_PER_CLASS]
    saved_count = 0
    print(f"Processing class '{dir_name}' ({len(img_list)} images)...")

    for img_name in img_list:
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"   can't read image: {img_path}")
            continue

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"   cvtColor failed for {img_path}: {e}")
            continue

        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            continue

        # إذا كان هناك أكثر من يد سنأخذ الأولى فقط
        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_ = [], []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            data_aux = []
            xmin = min(x_)
            ymin = min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - xmin)
                data_aux.append(lm.y - ymin)

            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_name)
                saved_count += 1

    print(f"  -> saved {saved_count} samples for '{dir_name}'")

hands.close()

# حفظ الملف
with open(OUT_FILE, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"\nDone. Total samples saved: {len(data)} -> {OUT_FILE}")
