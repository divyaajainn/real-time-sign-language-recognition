import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


GESTURES = ["A", "B", "C", "D"]
LANDMARK_PATH = "data/landmarks"
MODEL_PATH = "model"

os.makedirs(MODEL_PATH, exist_ok=True)

# Load Dataset
X = []
y = []

for gesture in GESTURES:
    file_path = os.path.join(LANDMARK_PATH, f"{gesture}.npy")
    data = np.load(file_path)

    X.extend(data)
    y.extend([gesture] * len(data))

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# TinyML-friendly Model
model = Sequential([
    Dense(64, activation="relu", input_shape=(63,)),
    Dense(32, activation="relu"),
    Dense(len(GESTURES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

model.save(os.path.join(MODEL_PATH, "gesture.h5"))
print("Model saved as model/gesture.h5")
