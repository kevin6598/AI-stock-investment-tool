"""
Uncertainty Calibration Module
-------------------------------
Calibration metrics and conformal prediction intervals.

Components:
  - Brier score for probabilistic predictions
  - Expected Calibration Error (ECE)
  - Reliability diagram data
  - Split conformal prediction intervals
  - Full calibration pipeline
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_brier_score(
    predicted_probs: np.ndarray,
    actuals_binary: np.ndarray,
) -> float:
    """Compute Brier score for binary probabilistic predictions.

    Brier score = mean((p - y)^2), where p is predicted probability
    and y is binary outcome. Lower is better; 0 = perfect.

    Args:
        predicted_probs: Predicted probabilities in [0, 1].
        actuals_binary: Binary outcomes (0 or 1).

    Returns:
        Brier score (float).
    """
    mask = ~(np.isnan(predicted_probs) | np.isnan(actuals_binary))
    if mask.sum() < 1:
        return 1.0
    p = predicted_probs[mask]
    y = actuals_binary[mask]
    return float(np.mean((p - y) ** 2))


def compute_ece(
    predicted_probs: np.ndarray,
    actuals_binary: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    ECE = sum(bin_count/N * |bin_accuracy - bin_confidence|) for each bin.

    Args:
        predicted_probs: Predicted probabilities in [0, 1].
        actuals_binary: Binary outcomes (0 or 1).
        n_bins: Number of probability bins.

    Returns:
        ECE value (lower is better).
    """
    mask = ~(np.isnan(predicted_probs) | np.isnan(actuals_binary))
    p = predicted_probs[mask]
    y = actuals_binary[mask]
    n = len(p)

    if n < 1:
        return 1.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
        count = bin_mask.sum()
        if count == 0:
            continue
        bin_accuracy = y[bin_mask].mean()
        bin_confidence = p[bin_mask].mean()
        ece += (count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def reliability_diagram_data(
    predicted_probs: np.ndarray,
    actuals_binary: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Compute data for reliability (calibration) diagram.

    Args:
        predicted_probs: Predicted probabilities in [0, 1].
        actuals_binary: Binary outcomes (0 or 1).
        n_bins: Number of probability bins.

    Returns:
        Dict with:
            bin_centers: Center of each bin.
            bin_accuracies: Observed accuracy in each bin.
            bin_counts: Number of samples in each bin.
    """
    mask = ~(np.isnan(predicted_probs) | np.isnan(actuals_binary))
    p = predicted_probs[mask]
    y = actuals_binary[mask]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        bin_mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])
        count = bin_mask.sum()
        bin_counts[i] = count
        bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        if count > 0:
            bin_accuracies[i] = y[bin_mask].mean()

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def conformal_prediction_interval(
    calibration_residuals: np.ndarray,
    new_predictions: np.ndarray,
    alpha: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute split conformal prediction intervals.

    Uses the calibration set residuals to determine the prediction
    interval width, guaranteeing (1-alpha) coverage asymptotically.

    Args:
        calibration_residuals: |y_true - y_pred| on calibration set.
        new_predictions: Point predictions for new data.
        alpha: Significance level (e.g., 0.10 for 90% coverage).

    Returns:
        (lower_bounds, upper_bounds) arrays.
    """
    # Compute the (1-alpha) quantile of absolute residuals
    n_cal = len(calibration_residuals)
    if n_cal < 1:
        width = 0.0
    else:
        # Finite-sample correction: use ceil((n+1)(1-alpha))/n quantile
        level = min((n_cal + 1) * (1 - alpha) / n_cal, 1.0)
        width = float(np.quantile(np.abs(calibration_residuals), level))

    lower = new_predictions - width
    upper = new_predictions + width

    return lower, upper


@dataclass
class CalibrationReport:
    """Full calibration report."""
    brier_score: float = 1.0
    ece: float = 1.0
    reliability_data: Dict[str, np.ndarray] = field(default_factory=dict)
    conformal_width: float = 0.0
    conformal_coverage: float = 0.0


def calibrate_model(
    y_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float = 0.10,
    prob_cal: Optional[np.ndarray] = None,
    prob_test: Optional[np.ndarray] = None,
) -> CalibrationReport:
    """Full calibration pipeline.

    Args:
        y_cal: Calibration set true values.
        y_pred_cal: Calibration set predictions.
        y_test: Test set true values.
        y_pred_test: Test set predictions.
        alpha: Significance level for conformal intervals.
        prob_cal: Optional probability predictions for Brier/ECE (calibration).
        prob_test: Optional probability predictions for Brier/ECE (test).

    Returns:
        CalibrationReport with all calibration metrics.
    """
    report = CalibrationReport()

    # Conformal prediction
    cal_residuals = np.abs(y_cal - y_pred_cal)
    cal_residuals = cal_residuals[~np.isnan(cal_residuals)]

    lower, upper = conformal_prediction_interval(
        cal_residuals, y_pred_test, alpha=alpha,
    )

    valid = ~(np.isnan(y_test) | np.isnan(y_pred_test))
    if valid.sum() > 0:
        coverage = np.mean(
            (y_test[valid] >= lower[valid]) & (y_test[valid] <= upper[valid])
        )
        report.conformal_coverage = float(coverage)
        report.conformal_width = float(np.mean(upper[valid] - lower[valid]))

    # Brier score and ECE (if probability predictions provided)
    if prob_test is not None:
        # Convert continuous targets to binary for Brier score
        # Convention: positive return = 1, negative = 0
        binary_test = (y_test > 0).astype(float)
        binary_test[np.isnan(y_test)] = np.nan

        report.brier_score = compute_brier_score(prob_test, binary_test)
        report.ece = compute_ece(prob_test, binary_test)
        report.reliability_data = reliability_diagram_data(prob_test, binary_test)
    elif valid.sum() > 0:
        # Use sign-based binary for Brier approximation
        binary_test = (y_test > 0).astype(float)
        pred_probs = 1.0 / (1.0 + np.exp(-y_pred_test * 100))  # sigmoid scaling
        pred_probs = np.clip(pred_probs, 0.01, 0.99)
        report.brier_score = compute_brier_score(pred_probs, binary_test)
        report.ece = compute_ece(pred_probs, binary_test)

    return report
