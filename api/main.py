"""
FastAPI Application
--------------------
Main entry point for the AI Stock Prediction API.

Run with: python -m api.main
Serves on: http://127.0.0.1:8000
"""

import sys
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.model_cache import ModelCache
from api.feature_pipeline import ServingFeaturePipeline
from api.routes import predict, indicators, sentiment, health, diagnostics
from api.auth import APIKeyMiddleware, RateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
model_cache = ModelCache()
feature_pipeline = ServingFeaturePipeline()
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up on shutdown."""
    logger.info("Starting AI Stock Prediction API...")

    # Load model artifacts
    loaded = model_cache.load()
    if loaded:
        logger.info("Model loaded successfully")
        # Configure feature pipeline with loaded artifacts
        feature_pipeline.feature_columns = model_cache.feature_columns
        feature_pipeline.feature_scaler = model_cache.feature_scaler
    else:
        logger.warning("No pre-trained model found; API will train on-demand")

    # Inject dependencies into route modules
    predict.set_dependencies(model_cache, feature_pipeline)
    health.set_model_cache(model_cache)

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="AI Stock Prediction Platform",
    description="Multi-modal deep learning stock prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# API Key authentication (enabled via API_KEY env var)
api_key = os.environ.get("API_KEY", "")
if api_key:
    app.add_middleware(
        APIKeyMiddleware,
        api_keys=[api_key],
    )
    logger.info("API key authentication enabled")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict.router)
app.include_router(indicators.router)
app.include_router(sentiment.router)
app.include_router(health.router)
app.include_router(diagnostics.router)


@app.get("/")
async def root():
    return {
        "name": "AI Stock Prediction Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
