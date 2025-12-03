# train_model.py
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
from collections import Counter

DATA_FILE = "data.ASL"
OUT_MODEL = "model.ASL"
RANDOM_STATE = 42

if not os.path.isfile(DATA_FILE):
    raise SystemExit(f"ERROR: data file not found: {DATA_FILE}. Run extract_data.py first.")

with open(DATA_FILE, "rb") as f:
    data_dict = pickle.load(f)

data = np.array(data_dict.get("data", []))
labels = np.array(data_dict.get("labels", []))

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

if data.size == 0 or labels.size == 0:
    raise SystemExit("ERROR: No data found. Check extraction step.")

# If stratify fails because some classes have 1 sample, fallback without stratify
label_counts = Counter(labels)
min_count = min(label_counts.values())
print("Samples per class (example):", list(label_counts.items())[:6], " ...")
stratify = labels if min_count >= 2 else None
if stratify is None:
    print("Warning: some classes have <2 samples; using non-stratified split.")

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=stratify, random_state=RANDOM_STATE
)

model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# حفظ الموديل
with open(OUT_MODEL, "wb") as f:
    pickle.dump({"model": model}, f)

print(f"Saved model to {OUT_MODEL}")
