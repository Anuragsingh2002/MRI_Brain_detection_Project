# app/models/mri_model.py
import os
from tensorflow.keras.models import load_model as keras_load_model
from app.config import MODEL_PATH
import numpy as np

def load_mri_model():
    """Load and return the Keras model from MODEL_PATH."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = keras_load_model(MODEL_PATH)
    return model


def make_prediction(model, preprocessed_image):
    """
    Run the model on a single preprocessed image array.
    preprocessed_image: numpy array shaped (1,H,W,C) or (H,W,C).
    Returns probability float.
    """
    arr = np.asarray(preprocessed_image)

    # Ensure batch dimension
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)

    if arr.ndim != 4:
        raise ValueError(f"Input must be 4D (batch,H,W,C); got shape {arr.shape}")

    # ensure dtype float32
    if arr.dtype != np.float32:
        arr = arr.astype("float32")

    # Run inference
    preds = model.predict(arr)

    # Interpret output: softmax(2) or sigmoid(1)
    if preds.ndim == 2 and preds.shape[1] == 2:
        prob = float(preds[0, 1])
    else:
        prob = float(preds[0, 0])

    return prob
