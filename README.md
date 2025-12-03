# Sign language Recognition System

## Project Overview

This project implements an **American Sign Language (ASL) alphabet recognition system** that can detect and interpret hand gestures in real-time using a webcam. It leverages **Mediapipe Hands** for hand landmark detection and **machine learning models** (Random Forest) for gesture classification. The system supports features such as repeated letter handling, deletion, spacing, and real-time text-to-speech.

---

## Features

* **Real-time hand gesture recognition** using a standard webcam.
* **Machine learning-based classification** of ASL letters.
* **Text-to-speech** for the recognized words.
* **Repeat letter handling** for continuous letters (e.g., 'hello').
* **Word editing support** with space and delete gestures.
* **Error handling** for robust execution even if the hand is not detected.
* **Cross-platform support** (Windows/Linux).

---

## Project Structure

```
ASL-Recognition/
│
├─ extract_data.py       # Data extraction and preprocessing
├─ train_model.py        # Model training and saving
├─ predict.py            # Real-time prediction and TTS
├─ data.ASL              # Serialized dataset (after extraction)
├─ model.ASL             # Saved trained model
└─ README.md
```

---

## Requirements

* Python 3.8+
* OpenCV
* Mediapipe
* NumPy
* scikit-learn
* pyttsx3

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scikit-learn pyttsx3
```

---

## Usage

### 1. Data Extraction

```bash
python extract_data.py --data_dir "path/to/asl_dataset" --max_per_class 200
```

### 2. Model Training

```bash
python train_model.py --data_file "data.ASL" --output_model "model.ASL"
```

### 3. Real-time Prediction

```bash
python predict.py --model_file "model.ASL"
```

Controls during prediction:

* Press **`s`** → Speak the recognized word via TTS.
* Press **`c`** → Clear the current word.
* Press **`q`** → Quit the application.

---

هل تريد أن أصممها؟
