# app/routes/predict_route.py
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette import status
from concurrent.futures import ThreadPoolExecutor
import asyncio, inspect, io
from PIL import Image

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=2)

from app.services.prediction_service import predict_mri

async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


@router.post("/predict", tags=["predict"])
async def predict(request: Request, file: UploadFile = File(...)):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload")
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image upload: {e}")

    try:
        # predict_mri is synchronous; run in threadpool
        result = await _run_in_thread(predict_mri, pil_img, model)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Prediction failed: {e}")

    if not isinstance(result, dict) or "label" not in result or "probability" not in result:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Unexpected prediction format: {result}")

    return JSONResponse({"prediction": result["label"], "probability": float(result["probability"])})
