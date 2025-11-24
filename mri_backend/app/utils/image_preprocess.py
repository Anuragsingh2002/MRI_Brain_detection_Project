# app/utils/image_preprocess.py
from io import BytesIO
from PIL import Image
import numpy as np

def preprocess_image(pil_img, target_size=(224, 224)):
    """
    Accepts a PIL.Image (or bytes) and produces a numpy array shaped (1, H, W, C), dtype float32.
    `target_size` should be (width, height) for PIL.resize.
    """
    # If caller passed bytes, convert to PIL first
    if isinstance(pil_img, (bytes, bytearray)):
        pil_img = Image.open(BytesIO(pil_img)).convert("RGB")

    if not isinstance(pil_img, Image.Image):
        raise ValueError("preprocess_image expects a PIL.Image or raw bytes")

    # PIL resize uses (width, height)
    img = pil_img.resize((target_size[0], target_size[1]), Image.BILINEAR).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0

    # Ensure batch dim (1, H, W, C)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    return arr
