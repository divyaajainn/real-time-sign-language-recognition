import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/gesture.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]["shape"])
print("Output shape:", output_details[0]["shape"])

# Load one sample landmark
sample = np.load("data/landmarks/A.npy")[0]
sample = np.expand_dims(sample, axis=0).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]["index"], sample)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
prediction = np.argmax(output)

GESTURES = ["A", "B", "C", "D"]
print("Predicted gesture:", GESTURES[prediction])
