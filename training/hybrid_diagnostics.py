"""
Hybrid Model Diagnostics & Stability
--------------------------------------
Stability penalties, composite scoring with IC_std, and diagnostics
report generation for the GatedHybridModel.

Functions:
  - compute_stability_penalty: variance-based IC stability penalty
  - compute_composite_with_stability: composite score with IC_std term
  - build_diagnostics_report: full JSON diagnostics report
  - save_diagnostics: write report to artifacts/
"""

from typing import Dict, List, Optional, Any
import json
import logging
import math
import os
import numpy as np

logger = logging.getLogger(__name__)


def compute_stability_penalty(
    fold_ics: List[float],
    lambda_: float = 1.0,
) -> float:
    """Compute stability penalty based on IC variance across folds.

    Args:
        fold_ics: IC values per fold.
        lambda_: Weight for the variance penalty.

    Returns:
        Penalty value = lambda * Var(fold_ics).
    """
    if len(fold_ics) < 2:
        return 0.0
    return lambda_ * float(np.var(fold_ics))


def compute_composite_with_stability(
    metrics: Dict[str, float],
    fold_ics: Optional[List[float]] = None,
    stability_lambda: float = 1.0,
) -> float:
    """Compute composite score with IC stability penalty.

    Score = 1.0*IC + 0.5*ICIR + 0.3*Sharpe
            - 0.3*IC_std - 0.5*overfit - 0.3*stress_dd
            - stability_lambda * Var(fold_ics)

    Returns NaN-safe float (returns -inf if any key metric is NaN).
    """
    ic = metrics.get("mean_ic", 0.0)
    icir = metrics.get("icir", 0.0)
    sharpe = metrics.get("mean_sharpe", 0.0)
    ic_std = metrics.get("ic_std", 0.0)
    overfit = metrics.get("overfitting_score", 0.5)
    stress_dd = abs(metrics.get("stress_max_drawdown", 0.3))

    key_vals = [ic, icir, sharpe, ic_std, overfit, stress_dd]
    if any(isinstance(v, float) and math.isnan(v) for v in key_vals):
        return float("-inf")

    score = (
        1.0 * ic
        + 0.5 * icir
        + 0.3 * sharpe
        - 0.3 * ic_std
        - 0.5 * overfit
        - 0.3 * stress_dd
    )

    if fold_ics is not None:
        score -= compute_stability_penalty(fold_ics, stability_lambda)

    return float(score)


def build_diagnostics_report(
    model_name: str,
    metrics: Dict[str, float],
    gate_stats: Optional[List[Dict[str, float]]] = None,
    nlp_importance: Optional[Dict[str, float]] = None,
    fold_ics: Optional[List[float]] = None,
    param_count: int = 0,
    ablation_results: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a comprehensive diagnostics report for the hybrid model.

    Args:
        model_name: Model identifier.
        metrics: Evaluation metrics dict (IC, ICIR, Sharpe, etc).
        gate_stats: Gate activation statistics per epoch.
        nlp_importance: Permutation importance for NLP features.
        fold_ics: Per-fold IC values.
        param_count: Total model parameter count.
        ablation_results: 3-way ablation results.

    Returns:
        Dict report suitable for JSON serialization.
    """
    report = {
        "model_name": model_name,
        "param_count": param_count,
        "metrics": {},
        "stability": {},
        "gate_analysis": {},
        "nlp_analysis": {},
        "ablation": {},
    }

    # Metrics (NaN-safe)
    for k, v in metrics.items():
        if isinstance(v, float) and math.isnan(v):
            report["metrics"][k] = None
        else:
            report["metrics"][k] = v

    # Stability
    if fold_ics:
        report["stability"] = {
            "fold_ics": fold_ics,
            "ic_mean": float(np.mean(fold_ics)),
            "ic_std": float(np.std(fold_ics)),
            "ic_var": float(np.var(fold_ics)),
            "stability_penalty": compute_stability_penalty(fold_ics),
            "composite_with_stability": compute_composite_with_stability(
                metrics, fold_ics,
            ),
        }

    # Gate analysis
    if gate_stats:
        last_stats = gate_stats[-1] if gate_stats else {}
        report["gate_analysis"] = {
            "final_gate_mean": last_stats.get("gate_mean", 0.0),
            "final_gate_std": last_stats.get("gate_std", 0.0),
            "n_epochs": len(gate_stats),
            "gate_trajectory": [
                {"epoch": s.get("epoch", 0), "mean": s.get("gate_mean", 0.0)}
                for s in gate_stats[-10:]
            ],
        }

    # NLP analysis (top 5 by importance)
    if nlp_importance:
        sorted_nlp = sorted(
            nlp_importance.items(), key=lambda x: abs(x[1]), reverse=True,
        )
        report["nlp_analysis"] = {
            "top_5_features": [
                {"feature": k, "importance": v} for k, v in sorted_nlp[:5]
            ],
            "pruned_features": [
                k for k, v in sorted_nlp if v < 0.001
            ],
            "total_nlp_features": len(nlp_importance),
        }

    # Ablation
    if ablation_results:
        report["ablation"] = ablation_results

    return report


def save_diagnostics(
    report: Dict[str, Any],
    path: Optional[str] = None,
) -> str:
    """Write diagnostics report to JSON file.

    Args:
        report: Diagnostics report dict.
        path: Output path. Defaults to artifacts/model_diagnostics.json.

    Returns:
        Absolute path to the saved file.
    """
    if path is None:
        artifacts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts",
        )
        os.makedirs(artifacts_dir, exist_ok=True)
        path = os.path.join(artifacts_dir, "model_diagnostics.json")

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Diagnostics saved to %s", path)
    return os.path.abspath(path)
