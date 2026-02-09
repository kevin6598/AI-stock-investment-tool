"""
Health Check Route
-------------------
GET /api/v1/health - Service health + model status
"""

from fastapi import APIRouter
from api.schemas import HealthResponse, ModelInfo

router = APIRouter(prefix="/api/v1", tags=["health"])

_model_cache = None


def set_model_cache(model_cache):
    global _model_cache
    _model_cache = model_cache


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    model_info = None
    model_loaded = False

    if _model_cache is not None and _model_cache.is_loaded:
        model_loaded = True
        model_info = ModelInfo(
            model_type=_model_cache.config.get("model_type", "unknown"),
            version=_model_cache.model_version,
            n_features=len(_model_cache.feature_columns),
            trained_tickers=_model_cache.trained_tickers,
            last_updated=_model_cache.metadata.get("trained_at"),
        )

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_info=model_info,
        uptime_seconds=_model_cache.uptime_seconds if _model_cache else 0.0,
    )
