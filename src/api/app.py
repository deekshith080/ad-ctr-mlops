"""
Main FastAPI application for CTR prediction.
Exposes 3 endpoints:
  GET  /health      - is the API alive?
  GET  /model/info  - which model is loaded?
  POST /predict     - get a CTR prediction
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.model_loader import model_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "runs:/62ede5ab63ba4fdab06df673b287d5f8/model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the API starts up.
    Loads the model into memory before accepting any requests.
    This is FastAPI's modern way of handling startup events.
    """
    logger.info("API starting up — loading model...")
    try:
        mlflow.set_tracking_uri("mlruns")
        model_loader.load(
            model_uri=MODEL_URI,
            reference_data_path="data/processed/train.csv",
        )
        logger.info("Model loaded successfully. API ready.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Ad CTR Prediction API",
    description="Predicts click-through rate probability for ad impressions using XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.
    Load balancers ping this every few seconds in production.
    Returns 200 if healthy, 503 if model not loaded.
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """
    Returns information about the currently loaded model.
    Useful for debugging — tells you exactly which MLflow run is serving traffic.
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return ModelInfoResponse(
        model_uri=model_loader.model_uri,
        run_id=model_loader.run_id,
        baseline_auc=model_loader.baseline_auc,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Main prediction endpoint.
    Takes ad impression features and returns CTR probability.

    Example use case:
    Ad auction system sends impression features here.
    Gets back probability 0.034.
    Calculates bid = target_cpc * probability = $2.00 * 0.034 = $0.068
    Submits bid to exchange.
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        features = request.model_dump()
        probability = model_loader.predict(features)
        return PredictResponse(
            ctr_probability=round(probability, 6),
            will_click=probability >= 0.5,
            model_version=model_loader.run_id,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
