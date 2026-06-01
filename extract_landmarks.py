import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


GESTURES = ["A", "B", "C", "D"]

RAW_PATH = "data/raw_videos"
LANDMARK_PATH = "data/landmarks"
MODEL_PATH = "hand_landmarker.task"

print("Script started")
print("Current directory:", os.getcwd())
print("Gesture folders found:", os.listdir(RAW_PATH))

os.makedirs(LANDMARK_PATH, exist_ok=True)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

for gesture in GESTURES:
    folder = os.path.join(RAW_PATH, gesture)
    print(f"\n Processing gesture: {gesture}")

    if not os.path.exists(folder):
        print(f"Folder missing: {folder}")
        continue

    data = []
    images = os.listdir(folder)
    print(f"Images found: {len(images)}")

    for img_name in images:
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        result = detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = []
            for lm in result.hand_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z])
            data.append(landmarks)

    data = np.array(data)
    np.save(os.path.join(LANDMARK_PATH, f"{gesture}.npy"), data)
    print(f"Saved {data.shape[0]} samples")

print("\n Landmark extraction complete")
