# app/main.py
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mri_app")

BASE = Path(__file__).resolve().parent  # points to app/

app = FastAPI(title="Brain MRI Tumor Prediction")

# mount static directory if exists (serves /static/...)
STATIC_DIR = BASE / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info("Mounted static files from %s", STATIC_DIR)
else:
    logger.info("No static directory found at %s (that's OK)", STATIC_DIR)

# templates dir (optional)
TEMPLATES_DIR = BASE / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

# include GUI router if present (log problems, but continue)
try:
    from app.gui import router as gui_router  # optional package
    app.include_router(gui_router)
    logger.info("GUI router included")
except Exception as e:
    logger.info("GUI router not included: %s", e)

# include predict router (expected to exist)
try:
    from app.routes.predict_route import router as predict_router
    app.include_router(predict_router)
    logger.info("Predict router included")
except Exception as e:
    # if this import fails, you want to know immediately
    logger.exception("Failed to include predict_router - check app/routes/predict_route.py: %s", e)
    raise

# load model helper (wrap in try/except so reload worker races don't crash silently)
try:
    from app.models.mri_model import load_mri_model
except Exception as e:
    logger.exception("Failed to import load_mri_model from app.models.mri_model: %s", e)
    # optionally re-raise if you want startup to fail hard
    # raise

@app.on_event("startup")
def startup_event():
    """
    Load model and attach to app.state.model.
    Use try/except so that when --reload spawns both watcher and worker,
    we don't silently fail without a helpful log.
    """
    try:
        if "load_mri_model" in globals():
            logger.info("Loading model...")
            app.state.model = load_mri_model()
            logger.info("Model loaded and stored on app.state.model")
        else:
            logger.warning("load_mri_model not available; model not loaded.")
    except Exception as e:
        # don't let this crash silently during reload — log full traceback
        logger.exception("Error loading model at startup: %s", e)
        # If you want the server to fail hard on model load, uncomment:
        # raise

@app.on_event("shutdown")
def shutdown_event():
    # close resources here if needed
    logger.info("Shutting down...")

# Root route to avoid 404 at GET /
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # prefer serving a template if available
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    # fallback plain HTML
    return HTMLResponse(
        "<html><body><h1>Brain MRI Tumor Prediction</h1>"
        "<p>Upload image at /predict (see <a href='/docs'>/docs</a> for API)</p></body></html>"
    )

# favicon route to avoid browser 404s
@app.get("/favicon.ico")
def favicon():
    # try static favicon first
    fav = STATIC_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav))
    # else try templates/static fallback location
    alt = BASE / "favicon.ico"
    if alt.exists():
        return FileResponse(str(alt))
    # no favicon available -> return 204 (no content)
    return FileResponse(str(fav)) if fav.exists() else HTMLResponse(status_code=204)

# small debug helper (optional)
@app.get("/_routes")
def show_routes():
    """Return list of registered routes (useful for debugging)."""
    return {"routes": [r.path for r in app.routes]}
