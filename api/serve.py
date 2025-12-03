"""
Minimal FastAPI service for the phishing classification pipeline.

This module exposes:
  - GET /health  : basic liveness check
  - POST /predict: returns a phishing prediction given feature values

The service expects that a trained model (and optionally a preprocessor)
has been persisted using src.model_persistence.ModelPersistence into the
configured results directory (default: results/).

This script is intentionally conservative and focuses on a clear,
enterprise-grade serving pattern rather than advanced API features.
"""

from typing import Any, Dict, List, Optional

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import Config
from src.model_persistence import ModelPersistence


class PredictRequest(BaseModel):
    """
    Request schema for /predict.

    For simplicity, this API accepts features as a flat mapping from
    feature name to numeric value. It is the caller's responsibility
    to ensure that:

      - The feature set and order are compatible with the persisted model.
      - Any required preprocessing (e.g., scaling, encoding) has been
        applied in the same way as during training.

    In a production setting, you would typically load and apply the
    persisted preprocessing pipeline here as well.
    """

    features: Dict[str, float]


class PredictResponse(BaseModel):
    model_name: str
    prediction: int
    probability: Optional[float] = None


def load_model() -> Any:
    """
    Load a trained model using ModelPersistence.

    The model name can be supplied via the MODEL_NAME environment variable.
    If not set, it defaults to 'RandomForest'.
    """
    # Load configuration (results_dir and output settings)
    config = Config()
    persistence = ModelPersistence(config)

    model_name = os.getenv("MODEL_NAME", "RandomForest")

    try:
        model = persistence.load_model(model_name)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model '{model_name}' could not be loaded. "
            "Ensure it has been trained and saved using the pipeline."
        ) from exc

    return model_name, model


app = FastAPI(title="Phishing Detection API", version="1.0.0")

MODEL_NAME, MODEL = load_model()


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Basic health endpoint to verify the service is running.
    """
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Run a phishing prediction for a single example.

    Notes:
      - The API expects that the caller supplies features in an order and
        representation compatible with the trained model.
      - In many real-world deployments, the API would apply the persisted
        preprocessing pipeline to raw inputs before calling the model.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Convert feature mapping to ordered list, sorted by key for determinism.
    # In production, you would align this explicitly with training feature order.
    feature_items: List[float] = [
        v for _, v in sorted(request.features.items(), key=lambda kv: kv[0])
    ]

    try:
        # scikit-learn / XGBoost interface: predict and predict_proba
        y_pred = MODEL.predict([feature_items])[0]
        prob: Optional[float] = None

        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba([feature_items])[0]
            # Assuming binary classification: probability of phishing (class 1)
            if len(proba) == 2:
                prob = float(proba[1])

    except Exception as exc:  # pragma: no cover - defensive catch
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        ) from exc

    return PredictResponse(model_name=MODEL_NAME, prediction=int(y_pred), probability=prob)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


