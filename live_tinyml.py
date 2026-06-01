import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


GESTURES = ["A", "B", "C", "D"]
MODEL_PATH = "model/gesture.tflite"
HAND_MODEL = "hand_landmarker.task"


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=HAND_MODEL)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

# opens Webcam
cap = cv2.VideoCapture(0)

print("✅ Live TinyML demo started (Press ESC to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect hand landmarks
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])

        sample = np.array(landmarks, dtype=np.float32)
        sample = np.expand_dims(sample, axis=0)

        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        pred = np.argmax(output)

        cv2.putText(
            frame,
            GESTURES[pred],
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("TinyML Live Demo", frame)

    # Exit: ESC 
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
