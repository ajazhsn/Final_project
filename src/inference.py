# src/inference.py
import os
import io
import json
import time
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import mlflow.pytorch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response
import uvicorn

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────
IMAGES_PROCESSED = Counter(
    "plant_images_processed_total",
    "Total images processed",
    ["mode", "class_predicted"]
)
INFERENCE_LATENCY = Histogram(
    "plant_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
ACTIVE_REQUESTS = Gauge(
    "plant_active_requests",
    "Number of active requests"
)
MODEL_LOAD_TIME = Gauge(
    "plant_model_load_time_seconds",
    "Time taken to load model"
)
PREDICTION_CONFIDENCE = Histogram(
    "plant_prediction_confidence",
    "Confidence score of predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5,
             0.6, 0.7, 0.8, 0.9, 1.0]
)
ERROR_COUNTER = Counter(
    "plant_prediction_errors_total",
    "Total prediction errors",
    ["error_type"]
)
BULK_BATCH_SIZE = Histogram(
    "plant_bulk_batch_size",
    "Number of images in bulk requests",
    buckets=[1, 2, 5, 10, 20, 50]
)


# ─────────────────────────────────────────
# Config & Model Loading
# ─────────────────────────────────────────
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


config = load_config()


class PlantCNN(nn.Module):
    def __init__(self, num_classes, num_filters):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters[0], 3, padding=1),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(num_filters[0], num_filters[1],
                      3, padding=1),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(num_filters[1], num_filters[2],
                      3, padding=1),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(num_filters[2] * 16, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    """Load the champion model."""
    start = time.time()
    model_path = "models/cnn_finetuned.pt"
    if not Path(model_path).exists():
        model_path = "models/cnn_model.pt"

    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=True)

    num_filters = checkpoint.get("num_filters", [32, 64, 128])
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    image_size = checkpoint.get("image_size", 224)

    model = PlantCNN(num_classes, num_filters)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    load_time = time.time() - start
    MODEL_LOAD_TIME.set(load_time)
    logger.info(f"Model loaded in {load_time:.2f}s")
    logger.info(f"Classes: {class_names}")

    return model, class_names, image_size


# Load model at startup
model, CLASS_NAMES, IMAGE_SIZE = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Detection API",
    description=(
        "Detect plant diseases from leaf images. "
        "Supports single and bulk inference."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    all_probabilities: dict
    inference_time_ms: float
    model_version: str = "PlantDiseaseDetector@champion"
    container_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    classes: list
    container_id: str
    timestamp: str


# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────
def predict_image(image: Image.Image):
    """Run inference on a single PIL image."""
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    predicted_class = CLASS_NAMES[pred_idx]
    all_probs = {
        CLASS_NAMES[i]: round(probs[i].item(), 4)
        for i in range(len(CLASS_NAMES))
    }
    return predicted_class, confidence, all_probs


CONTAINER_ID = os.environ.get(
    "HOSTNAME", os.uname().nodename)


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Plant Disease Detection API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse,
         tags=["Health"])
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        num_classes=len(CLASS_NAMES),
        classes=CLASS_NAMES,
        container_id=CONTAINER_ID,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse,
          tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """Single image prediction endpoint."""
    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            ERROR_COUNTER.labels(
                error_type="invalid_file_type").inc()
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        predicted_class, confidence, all_probs = \
            predict_image(image)

        inference_time = (time.time() - start) * 1000

        # Record metrics
        IMAGES_PROCESSED.labels(
            mode="single",
            class_predicted=predicted_class
        ).inc()
        INFERENCE_LATENCY.observe(
            (time.time() - start))
        PREDICTION_CONFIDENCE.observe(confidence)

        logger.info(
            f"Predicted: {predicted_class} "
            f"({confidence:.3f}) in {inference_time:.1f}ms"
        )

        return PredictionResponse(
            filename=file.filename,
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            all_probabilities=all_probs,
            inference_time_ms=round(inference_time, 2),
            container_id=CONTAINER_ID
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.labels(
            error_type="inference_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(
        files: list[UploadFile] = File(...)):
    """Bulk image prediction endpoint."""
    ACTIVE_REQUESTS.inc()
    BULK_BATCH_SIZE.observe(len(files))
    start = time.time()
    results = []

    try:
        for file in files:
            try:
                contents = await file.read()
                image = Image.open(
                    io.BytesIO(contents)).convert("RGB")
                predicted_class, confidence, all_probs = \
                    predict_image(image)

                IMAGES_PROCESSED.labels(
                    mode="bulk",
                    class_predicted=predicted_class
                ).inc()
                PREDICTION_CONFIDENCE.observe(confidence)

                results.append({
                    "filename": file.filename,
                    "predicted_class": predicted_class,
                    "confidence": round(confidence, 4),
                    "all_probabilities": all_probs,
                    "status": "success"
                })
            except Exception as e:
                ERROR_COUNTER.labels(
                    error_type="batch_item_error").inc()
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })

        total_time = (time.time() - start) * 1000
        INFERENCE_LATENCY.observe(
            (time.time() - start) / len(files))

        logger.info(
            f"Batch: {len(files)} images in "
            f"{total_time:.1f}ms"
        )

        return {
            "total_images": len(files),
            "successful": sum(
                1 for r in results
                if r["status"] == "success"),
            "failed": sum(
                1 for r in results
                if r["status"] == "error"),
            "total_time_ms": round(total_time, 2),
            "container_id": CONTAINER_ID,
            "results": results
        }

    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/classes", tags=["Info"])
def get_classes():
    """Get list of supported disease classes."""
    return {
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "model": "PlantDiseaseDetector@champion"
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = config["deployment"]["api_port"]
    logger.info(f"Starting API on port {port}")
    uvicorn.run(
        "src.inference:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
