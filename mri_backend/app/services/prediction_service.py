# app/services/prediction_service.py
import io
from PIL import Image
from app.utils.image_preprocess import preprocess_image
from app.models.mri_model import make_prediction, load_mri_model
from app.config import THRESHOLD

def _pil_from_input(image_input):
    """Return a PIL.Image from UploadFile-like, bytes, or PIL.Image."""
    if hasattr(image_input, "read") and callable(image_input.read):
        # UploadFile: route usually already reads bytes, but handle both cases
        b = image_input.read()
        if isinstance(b, (bytes, bytearray)):
            return Image.open(io.BytesIO(b)).convert("RGB")
        else:
            # if read returns a coroutine or unexpected, try to coerce
            return Image.open(io.BytesIO(bytes(b))).convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    else:
        raise ValueError("predict_mri expects UploadFile/bytes or PIL.Image; got: " + str(type(image_input)))


def predict_mri(image_input, model):
    """
    Synchronous pipeline: accepts PIL.Image / bytes / UploadFile-like and returns:
    {"label": "Tumor"/"Normal", "probability": float}
    It infers the model input size from model.input_shape when available.
    """
    pil_img = _pil_from_input(image_input)

    # infer model input size (H,W) from model.input_shape if possible
    target_size = (224, 224)  # fallback (width, height)
    in_shape = getattr(model, "input_shape", None)
    if in_shape and len(in_shape) >= 4 and in_shape[1] and in_shape[2]:
        H, W = int(in_shape[1]), int(in_shape[2])   # model.input_shape: (None, H, W, C)
        target_size = (W, H)  # PIL expects (width, height)

    # preprocess and predict
    arr = preprocess_image(pil_img, target_size=target_size)
    prob = make_prediction(model, arr)
    label = "Tumor" if prob >= float(THRESHOLD) else "Normal"
    return {"label": label, "probability": round(float(prob), 4)}
