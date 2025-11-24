import os


# Path to your Keras model file (.h5, SavedModel folder, etc.)
MODEL_PATH = r'D:\\Brain_Tumor_Classification (MRI)\\brain_tumor_cnn.h5'


# Prediction threshold for binary classification (adjust to your model)
THRESHOLD = float(os.environ.get('PRED_THRESHOLD', 0.5))