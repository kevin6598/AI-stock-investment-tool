"""
Model Diagnostics & Transparency
----------------------------------
Computes comprehensive model diagnostics including overfitting scores,
expected accuracy, and structured diagnostic reports.

Reuses:
  - training.calibration for Brier score, ECE
  - training.model_selection.ModelEvaluation for fold metrics
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class ModelDiagnostics:
    """Comprehensive model diagnostic report."""
    model_type: str
    hyperparameters: Dict[str, Any]
    training_start: str
    training_end: str
    n_samples: int
    n_features: int
    best_fold_ic_mean: float
    ic_std: float
    icir: float
    max_drawdown: float
    sharpe_gross: float
    sharpe_net: float
    hit_ratio: float
    brier_score: float
    calibration_error: float
    overfitting_score: float
    stress_test: Dict[str, float]
    expected_accuracy_probability: float


class DiagnosticsEngine:
    """Computes model diagnostics from training artifacts.

    Usage:
        engine = DiagnosticsEngine()
        diag = engine.compute_diagnostics(...)
    """

    def __init__(self):
        pass

    def compute_overfitting_score(
        self,
        train_ic: float,
        val_ic: float,
        ic_variance: float = 0.0,
        weight_instability: float = 0.0,
        fold_dispersion: float = 0.0,
        max_ic_variance: float = 0.05,
        max_fold_dispersion: float = 0.10,
    ) -> float:
        """Compute overfitting score normalized to [0, 1].

        Higher score = more likely overfit.

        Formula:
          raw = (train_IC - val_IC) / max(train_IC, 0.01)
                + 0.3 * ic_variance / max_ic_variance
                + 0.2 * weight_instability
                + 0.2 * fold_dispersion / max_fold_dispersion
          overfitting_score = sigmoid(raw * 3)

        Args:
            train_ic: Average IC on training folds.
            val_ic: Average IC on validation/test folds.
            ic_variance: Variance of IC across folds.
            weight_instability: Instability of ensemble weights across folds [0, 1].
            fold_dispersion: Std of fold-level performance.
            max_ic_variance: Normalization constant for IC variance.
            max_fold_dispersion: Normalization constant for fold dispersion.

        Returns:
            Overfitting score in [0, 1]. <0.3=good, 0.3-0.6=moderate, >0.6=concerning.
        """
        # IC gap term
        ic_gap = (train_ic - val_ic) / max(abs(train_ic), 0.01)

        # Variance term
        var_term = 0.3 * ic_variance / max(max_ic_variance, 1e-8)

        # Instability term
        instab_term = 0.2 * weight_instability

        # Dispersion term
        disp_term = 0.2 * fold_dispersion / max(max_fold_dispersion, 1e-8)

        raw = ic_gap + var_term + instab_term + disp_term

        # Sigmoid mapping to [0, 1]
        score = 1.0 / (1.0 + math.exp(-raw * 3))

        return float(np.clip(score, 0.0, 1.0))

    def compute_expected_accuracy(
        self,
        hit_ratio: float,
        calibration_reliability: float = 1.0,
        meta_precision: float = 0.5,
    ) -> float:
        """Compute expected forward-looking accuracy probability.

        Combines historical hit ratio with calibration quality and
        meta-labeling precision.

        Args:
            hit_ratio: Historical hit ratio from walk-forward.
            calibration_reliability: 1 - ECE (higher = better calibrated).
            meta_precision: Precision of meta-labeling model.

        Returns:
            Expected accuracy probability in [0, 1].
        """
        # Weighted combination
        accuracy = (
            0.50 * hit_ratio +
            0.25 * calibration_reliability +
            0.25 * meta_precision
        )
        return float(np.clip(accuracy, 0.0, 1.0))

    def compute_diagnostics(
        self,
        model_type: str,
        hyperparameters: Optional[Dict] = None,
        training_start: str = "",
        training_end: str = "",
        n_samples: int = 0,
        n_features: int = 0,
        train_ic: float = 0.0,
        val_ic: float = 0.0,
        ic_std: float = 0.0,
        max_drawdown: float = 0.0,
        sharpe_gross: float = 0.0,
        sharpe_net: float = 0.0,
        hit_ratio: float = 0.5,
        brier_score: float = 1.0,
        calibration_error: float = 1.0,
        stress_results: Optional[Dict[str, float]] = None,
        ic_variance: float = 0.0,
        weight_instability: float = 0.0,
        fold_dispersion: float = 0.0,
        meta_precision: float = 0.5,
    ) -> ModelDiagnostics:
        """Compute full diagnostics report.

        Args:
            model_type: Name of the model (e.g., "lightgbm").
            hyperparameters: Model hyperparameters dict.
            training_start: Training data start date.
            training_end: Training data end date.
            n_samples: Number of training samples.
            n_features: Number of features.
            train_ic: Average IC on training set.
            val_ic: Average IC on validation/test set.
            ic_std: Standard deviation of IC across folds.
            max_drawdown: Maximum drawdown from backtest.
            sharpe_gross: Gross Sharpe ratio.
            sharpe_net: Net Sharpe ratio (after costs).
            hit_ratio: Directional accuracy.
            brier_score: Brier score for probabilistic predictions.
            calibration_error: Expected Calibration Error (ECE).
            stress_results: Dict of stress scenario -> Sharpe under stress.
            ic_variance: Variance of IC across folds.
            weight_instability: Instability of ensemble weights.
            fold_dispersion: Std of fold-level performance.
            meta_precision: Precision of meta-labeling model.

        Returns:
            ModelDiagnostics dataclass.
        """
        overfitting = self.compute_overfitting_score(
            train_ic, val_ic, ic_variance, weight_instability, fold_dispersion,
        )

        calibration_reliability = max(1.0 - calibration_error, 0.0)
        expected_accuracy = self.compute_expected_accuracy(
            hit_ratio, calibration_reliability, meta_precision,
        )

        icir = val_ic / max(ic_std, 1e-8) if ic_std > 0 else 0.0

        return ModelDiagnostics(
            model_type=model_type,
            hyperparameters=hyperparameters or {},
            training_start=training_start,
            training_end=training_end,
            n_samples=n_samples,
            n_features=n_features,
            best_fold_ic_mean=val_ic,
            ic_std=ic_std,
            icir=icir,
            max_drawdown=max_drawdown,
            sharpe_gross=sharpe_gross,
            sharpe_net=sharpe_net,
            hit_ratio=hit_ratio,
            brier_score=brier_score,
            calibration_error=calibration_error,
            overfitting_score=overfitting,
            stress_test=stress_results or {},
            expected_accuracy_probability=expected_accuracy,
        )

    @staticmethod
    def to_json(diag: ModelDiagnostics) -> Dict[str, Any]:
        """Convert diagnostics to JSON-serializable dict."""
        return {
            "model_type": diag.model_type,
            "hyperparameters": diag.hyperparameters,
            "training_start": diag.training_start,
            "training_end": diag.training_end,
            "n_samples": diag.n_samples,
            "n_features": diag.n_features,
            "best_fold_ic_mean": round(diag.best_fold_ic_mean, 6),
            "ic_std": round(diag.ic_std, 6),
            "icir": round(diag.icir, 4),
            "max_drawdown": round(diag.max_drawdown, 4),
            "sharpe_gross": round(diag.sharpe_gross, 4),
            "sharpe_net": round(diag.sharpe_net, 4),
            "hit_ratio": round(diag.hit_ratio, 4),
            "brier_score": round(diag.brier_score, 4),
            "calibration_error": round(diag.calibration_error, 4),
            "overfitting_score": round(diag.overfitting_score, 4),
            "stress_test": {
                k: round(v, 4) for k, v in diag.stress_test.items()
            },
            "expected_accuracy_probability": round(diag.expected_accuracy_probability, 4),
        }
