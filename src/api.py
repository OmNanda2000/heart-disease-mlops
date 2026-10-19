
from pathlib import Path
import logging
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"

import json

with open(INFO_PATH, encoding="utf-8") as f:
    MODEL_INFO = json.load(f)

FEATURES = MODEL_INFO["features"]

# Only load a model artifact created by your own training process.
model = joblib.load(MODEL_PATH)

logging.basicConfig(
    filename=str(ROOT / "api.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Coursework demonstration of a binary classification model. "
        "Not intended for clinical diagnosis."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | None]


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features
    missing = sorted(set(FEATURES) - set(supplied))
    extra = sorted(set(supplied) - set(FEATURES))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    values = []
    for name in FEATURES:
        value = supplied[name]

        if value is not None:
            if isinstance(value, bool) or not math.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be a finite number or null.",
                )
        values.append(value)

    row = pd.DataFrame([values], columns=FEATURES)

    try:
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities))
    except Exception:
        logging.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        )

    logging.info(
        "Prediction completed; target=%s; confidence=%.4f",
        prediction,
        confidence,
    )

    return {
        "prediction": prediction,
        "label": (
            "Heart disease present"
            if prediction == 1
            else "No heart disease detected"
        ),
        "confidence": round(confidence, 4),
        "disclaimer": (
            "Educational model output, not a medical diagnosis."
        ),
    }
