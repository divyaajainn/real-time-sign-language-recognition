# Real-Time Sign Language Recognition System

A real-time Sign Language Recognition System that converts hand gestures into text using Computer Vision and Machine Learning. The system uses a webcam for gesture capture, MediaPipe for hand landmark detection, and a Random Forest classifier for gesture recognition.

## Features

- Real-time hand gesture detection using webcam
- MediaPipe-based hand landmark extraction
- Random Forest gesture classification
- Character-to-sentence generation
- Gesture-based controls:
  - Space
  - Backspace
  - Clear
- User-friendly GUI using Tkinter/CustomTkinter
- ESP32 integration support for IoT and voice-output applications
- Lightweight and hardware-independent solution

## Tech Stack

- Python
- OpenCV
- MediaPipe
- Scikit-Learn
- Random Forest
- Tkinter / CustomTkinter
- ESP32 (Optional)

## Project Architecture

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
Feature Extraction (21 Hand Landmarks)
   ↓
Random Forest Classifier
   ↓
Character Prediction
   ↓
Sentence Formation
   ↓
GUI Display / ESP32 Output
```

## Project Structure

```text
.
├── application.py
├── realtime_detection.py
├── ASL_model.p
├── ASL.pickle
├── hand-gesture-ISL.py
├── recog-alpha-number-common.py
├── requirements.txt
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/real-time-sign-language-recognition.git
cd real-time-sign-language-recognition
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python application.py
```

## How It Works

1. Webcam captures live video feed.
2. MediaPipe detects hand landmarks.
3. Landmark coordinates are normalized and converted into feature vectors.
4. Random Forest model predicts the corresponding gesture.
5. Predicted characters are displayed in real time.
6. Characters are combined to form words and sentences.
7. Special gestures enable editing actions such as Space, Backspace, and Clear.

## Applications

- Communication assistance for deaf and mute individuals
- Human-Computer Interaction (HCI)
- Smart assistive technologies
- IoT and voice-assistance systems
- Educational tools for learning sign language

## Future Enhancements

- Deep Learning based gesture recognition
- Expanded gesture vocabulary
- Text-to-Speech integration
- Mobile application deployment
- Multi-language sign language support

## Novelty

Unlike sensor-based sign language systems that require specialized hardware, this project uses only a standard webcam and computer vision techniques. This makes the solution lightweight, cost-effective, easy to deploy, and accessible for real-world applications.

## Authors

Developed as a Computer Vision and Machine Learning project for real-time sign language recognition and communication assistance.

## License

This project is intended for educational and research purposes.
