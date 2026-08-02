from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

from model_registry import load_all_models, available_categories, run_inference

app = FastAPI(title="Industrial Defect Detection API", version="1.0.0")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    load_all_models()


@app.get("/api/health")
def health():
    return {"status": "ok", "categories": available_categories()}


@app.get("/api/categories")
def categories():
    return {"categories": available_categories()}


@app.post("/api/predict")
async def predict(category: str = Form(...), file: UploadFile = File(...)):
    if category not in available_categories():
        raise HTTPException(status_code=400, detail=f"Unknown or untrained category: {category}")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    result = run_inference(category, image)
    return result