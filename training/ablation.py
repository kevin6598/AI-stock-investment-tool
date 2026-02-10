"""
Ablation Study Utilities
-------------------------
Tools for systematically disabling model branches and measuring impact.

Usage:
    config = AblationConfig(disable_vae=True)
    results = run_ablation_study(panel, target, features, [config], params)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    disable_vae: bool = False
    disable_sentiment: bool = False
    disable_ti_embedding: bool = False
    disable_temporal: bool = False
    disable_ticker_embedding: bool = False
    name: str = ""

    def __post_init__(self):
        if not self.name:
            disabled = []
            if self.disable_vae:
                disabled.append("no_vae")
            if self.disable_sentiment:
                disabled.append("no_sentiment")
            if self.disable_ti_embedding:
                disabled.append("no_ti")
            if self.disable_temporal:
                disabled.append("no_temporal")
            if self.disable_ticker_embedding:
                disabled.append("no_ticker")
            self.name = "_".join(disabled) if disabled else "full_model"

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dict for passing to HybridMultiModalNet."""
        return {
            "disable_vae": self.disable_vae,
            "disable_sentiment": self.disable_sentiment,
            "disable_ti_embedding": self.disable_ti_embedding,
            "disable_temporal": self.disable_temporal,
            "disable_ticker_embedding": self.disable_ticker_embedding,
        }


def log_gradient_norms(model) -> Dict[str, float]:
    """Log gradient norms for each named parameter group.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Dict of param_group_name -> gradient L2 norm.
    """
    try:
        import torch
    except ImportError:
        return {}

    norms = {}  # type: Dict[str, float]
    group_norms = {}  # type: Dict[str, List[float]]

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = float(param.grad.data.norm(2).item())
            # Group by first part of name (e.g., "temporal_encoder", "vae")
            group = name.split(".")[0]
            if group not in group_norms:
                group_norms[group] = []
            group_norms[group].append(grad_norm)

    for group, grad_list in group_norms.items():
        norms[group] = float(np.sqrt(sum(g ** 2 for g in grad_list)))

    return norms


def run_ablation_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: List[str],
    configs: Optional[List[AblationConfig]] = None,
    base_model_params: Optional[Dict] = None,
) -> List[Dict]:
    """Run ablation study with multiple configurations.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        feature_cols: Feature column names.
        configs: List of AblationConfig. If None, uses default ablation suite.
        base_model_params: Base parameters for HybridMultiModalModel.

    Returns:
        List of dicts with {name, ic, rmse, config}.
    """
    from training.models import create_model
    from training.model_selection import compute_prediction_metrics

    if configs is None:
        configs = [
            AblationConfig(),  # full model
            AblationConfig(disable_vae=True),
            AblationConfig(disable_sentiment=True),
            AblationConfig(disable_ti_embedding=True),
            AblationConfig(disable_temporal=True),
            AblationConfig(disable_ticker_embedding=True),
        ]

    if base_model_params is None:
        base_model_params = {
            "epochs": 10,
            "sequence_length": 20,
            "patience": 5,
            "hidden_dim": 32,
            "fusion_dim": 32,
            "vae_latent_dim": 4,
            "batch_size": 32,
        }

    results = []
    for cfg in configs:
        params = dict(base_model_params)
        params["ablation_config"] = cfg.to_dict()

        try:
            model = create_model("hybrid_multimodal", params)
            model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
            preds = model.predict(X_test)

            # Filter NaN
            valid = ~np.isnan(preds)
            metrics = compute_prediction_metrics(
                y_test[valid], preds[valid],
            )

            results.append({
                "name": cfg.name,
                "ic": metrics.ic,
                "rmse": metrics.rmse,
                "hit_ratio": metrics.hit_ratio,
                "config": cfg,
            })
            logger.info("Ablation '%s': IC=%.4f, RMSE=%.4f",
                        cfg.name, metrics.ic, metrics.rmse)
        except Exception as e:
            logger.warning("Ablation '%s' failed: %s", cfg.name, e)
            results.append({
                "name": cfg.name,
                "ic": 0.0,
                "rmse": float("inf"),
                "hit_ratio": 0.0,
                "config": cfg,
                "error": str(e),
            })

    return results
