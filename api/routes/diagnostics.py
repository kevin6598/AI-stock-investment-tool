"""
Model Diagnostics API
-----------------------
GET /api/v1/model/diagnostics - Returns model diagnostics and health metrics.
"""

from typing import Dict, Any
import os
import json
import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/model", tags=["diagnostics"])

# Default artifact directory for diagnostics metadata
_ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")


@router.get("/diagnostics")
async def get_diagnostics() -> Dict[str, Any]:
    """Return model diagnostics from the most recent training run.

    Looks for diagnostics in:
      1. artifacts/training_metadata.json (embedded diagnostics)
      2. Computed on-demand from model registry

    Returns:
        Dict with diagnostics metrics, overfitting score, stress test results, etc.
    """
    # Try loading cached diagnostics from artifact dir
    meta_path = os.path.join(_ARTIFACT_DIR, "training_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            if "diagnostics" in metadata:
                return {
                    "status": "ok",
                    "source": "cached",
                    "diagnostics": metadata["diagnostics"],
                }
        except Exception as e:
            logger.warning("Failed to read cached diagnostics: %s", e)

    # Try computing from registry
    try:
        from training.model_versioning import ModelRegistry
        registry = ModelRegistry()

        version = registry.select_best_model(horizon="1M")
        if version is None:
            version = registry.get_best_model(horizon="1M")

        if version is not None:
            metrics = version.metrics
            return {
                "status": "ok",
                "source": "registry",
                "diagnostics": {
                    "model_type": version.model_type,
                    "version_id": version.version_id,
                    "horizon": version.horizon,
                    "created_at": version.created_at,
                    "metrics": metrics,
                    "overfitting_score": metrics.get("overfitting_score", None),
                    "stress_test": metrics.get("stress_test", {}),
                    "ic_mean": metrics.get("mean_ic", None),
                    "icir": metrics.get("icir", None),
                    "sharpe": metrics.get("mean_sharpe", None),
                    "hit_ratio": metrics.get("hit_ratio", None),
                    "brier_score": metrics.get("brier_score", None),
                    "calibration_error": metrics.get("calibration_error", None),
                },
            }
    except Exception as e:
        logger.debug("Registry diagnostics failed: %s", e)

    # No diagnostics available
    return {
        "status": "no_data",
        "source": None,
        "diagnostics": None,
        "message": "No model diagnostics available. Train a model first.",
    }
