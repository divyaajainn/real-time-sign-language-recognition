import tensorflow as tf
import os

MODEL_PATH = "model/gesture.h5"
TFLITE_PATH = "model/gesture.tflite"

model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model/gesture.tflite")
