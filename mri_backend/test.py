# test_predict.py
import numpy as np
from app.models.mri_model import load_mri_model
m = load_mri_model()
print("model input shape:", m.input_shape)
# create a dummy batch matching (1,H,W,C). Replace H,W,C to match model.input_shape
H, W, C = m.input_shape[1], m.input_shape[2], m.input_shape[3]
arr = np.random.rand(1, H, W, C).astype("float32")
preds = m.predict(arr)
print("preds shape:", preds.shape, "preds:", preds)
