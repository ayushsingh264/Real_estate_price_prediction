from datetime import datetime, timezone
from uuid import uuid4
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from ml_pipeline.predict import ModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Price Prediction API",
    version="1.0.0",
    description="REST API for the Industry-Ready Real Estate Price Prediction System.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_COUNT = Counter("prediction_requests_total", "Prediction requests")
ERROR_COUNT = Counter("prediction_errors_total", "Prediction errors")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

try:
    MODEL_SERVICE = ModelService()
except Exception as exc:
    MODEL_SERVICE = None
    logger.warning("Model service not loaded: %s", exc)


class PropertyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: float = Field(..., gt=0)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=20)
    age: int = Field(..., ge=0, le=300)
    location: str = Field(..., min_length=2, max_length=100)
    property_type: str = Field(..., min_length=2, max_length=100)


class BatchPropertyRequest(BaseModel):
    properties: list[PropertyRequest] = Field(..., min_length=1, max_length=500)


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_SERVICE else "degraded",
        "model_loaded": MODEL_SERVICE is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/model")
def model_info():
    if MODEL_SERVICE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return {
        "version": MODEL_SERVICE.version,
        "model": MODEL_SERVICE.selected_model,
        "residual_std": MODEL_SERVICE.residual_std,
    }


@app.post("/api/v1/predict")
def predict(request: PropertyRequest):
    REQUEST_COUNT.inc()

    if MODEL_SERVICE is None:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        with LATENCY.time():
            result = MODEL_SERVICE.predict(**request.model_dump())
    except Exception as exc:
        ERROR_COUNT.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "prediction_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "currency": "INR",
        **result,
        "metadata": request.model_dump(),
    }


@app.post("/api/v1/predict/batch")
def batch_predict(request: BatchPropertyRequest):
    if MODEL_SERVICE is None:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=503, detail="Model is not available.")

    predictions = [
        {
            "prediction_id": str(uuid4()),
            "currency": "INR",
            **MODEL_SERVICE.predict(**item.model_dump()),
            "metadata": item.model_dump(),
        }
        for item in request.properties
    ]

    return {
        "count": len(predictions),
        "predictions": predictions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
