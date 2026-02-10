"""
Drift Detection Module
-----------------------
Comprehensive drift detection for ML models in production.

Components:
  - Population Stability Index (PSI) for distribution shifts
  - Per-feature KL divergence
  - Rolling IC decay detection
  - DriftDetector: orchestrates all checks
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Structured drift alert."""
    alert_type: str       # "psi", "kl_divergence", "ic_decay"
    severity: str         # "low", "medium", "high"
    feature_name: str     # which feature or metric
    current_value: float
    threshold: float
    message: str


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI measures how much a distribution has shifted from a baseline.
    PSI < 0.10: no significant change
    PSI 0.10-0.20: moderate drift
    PSI > 0.20: significant drift

    Args:
        expected: Baseline distribution (training data).
        actual: Current distribution (live/test data).
        n_bins: Number of histogram bins.

    Returns:
        PSI value (>= 0).
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # Create bins from baseline percentiles
    edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    base_counts, _ = np.histogram(expected, bins=edges)
    curr_counts, _ = np.histogram(actual, bins=edges)

    eps = 1e-6
    base_pct = base_counts / max(len(expected), 1) + eps
    curr_pct = curr_counts / max(len(actual), 1) + eps

    psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    return max(psi, 0.0)


def compute_feature_kl_divergence(
    train_features: np.ndarray,
    live_features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_bins: int = 20,
) -> Dict[str, float]:
    """Compute per-feature KL divergence using histogram approximation.

    Args:
        train_features: Training feature matrix (n_train, n_features).
        live_features: Live feature matrix (n_live, n_features).
        feature_names: Feature names for the output dict.
        n_bins: Number of histogram bins.

    Returns:
        Dict of feature_name -> KL divergence value.
    """
    n_features = train_features.shape[1]
    if feature_names is None:
        feature_names = ["feature_{}".format(i) for i in range(n_features)]

    kl_values = {}  # type: Dict[str, float]

    for i in range(min(n_features, len(feature_names))):
        train_col = train_features[:, i]
        live_col = live_features[:, i]

        train_col = train_col[~np.isnan(train_col)]
        live_col = live_col[~np.isnan(live_col)]

        if len(train_col) < 10 or len(live_col) < 10:
            kl_values[feature_names[i]] = 0.0
            continue

        # Create shared bins from training data
        edges = np.percentile(train_col, np.linspace(0, 100, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        edges = np.unique(edges)

        if len(edges) < 3:
            kl_values[feature_names[i]] = 0.0
            continue

        eps = 1e-8
        p_counts, _ = np.histogram(train_col, bins=edges)
        q_counts, _ = np.histogram(live_col, bins=edges)

        p = p_counts / max(len(train_col), 1) + eps
        q = q_counts / max(len(live_col), 1) + eps

        # KL(P || Q) = sum(p * log(p/q))
        kl = float(np.sum(p * np.log(p / q)))
        kl_values[feature_names[i]] = max(kl, 0.0)

    return kl_values


def detect_rolling_ic_decay(
    ic_series: np.ndarray,
    window: int = 63,
    decay_threshold: float = 0.5,
) -> bool:
    """Detect if rolling IC mean has decayed below threshold.

    Args:
        ic_series: Time series of IC values.
        window: Rolling window for computing mean.
        decay_threshold: Fraction of baseline IC below which decay is flagged.

    Returns:
        True if rolling IC mean drops below decay_threshold * baseline.
    """
    clean = ic_series[~np.isnan(ic_series)]
    if len(clean) < window * 2:
        return False

    # Baseline: first half mean
    baseline_ic = np.mean(clean[:len(clean) // 2])
    if baseline_ic <= 0:
        return np.mean(clean[-window:]) < 0

    # Recent IC: last window mean
    recent_ic = np.mean(clean[-window:])

    return recent_ic < decay_threshold * baseline_ic


class DriftDetector:
    """Orchestrates drift detection across features and model performance.

    Usage:
        detector = DriftDetector(baseline_features, baseline_ic, feature_names)
        alerts = detector.check(new_features, new_ic)
        if detector.should_retrain(alerts):
            trigger_retraining()
    """

    def __init__(
        self,
        baseline_features: np.ndarray,
        baseline_ic: float,
        feature_names: Optional[List[str]] = None,
        psi_threshold: float = 0.20,
        kl_threshold: float = 0.50,
        ic_decay_threshold: float = 0.50,
    ):
        """
        Args:
            baseline_features: Training feature matrix.
            baseline_ic: Baseline IC from training evaluation.
            feature_names: Feature names.
            psi_threshold: PSI threshold for significant drift.
            kl_threshold: KL divergence threshold per feature.
            ic_decay_threshold: IC decay fraction threshold.
        """
        self.baseline_features = baseline_features
        self.baseline_ic = baseline_ic
        self.feature_names = feature_names or [
            "feature_{}".format(i) for i in range(baseline_features.shape[1])
        ]
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.ic_decay_threshold = ic_decay_threshold
        self._ic_history = []  # type: List[float]

    def check(
        self,
        new_features: np.ndarray,
        new_ic: Optional[float] = None,
    ) -> List[DriftAlert]:
        """Run all drift checks.

        Args:
            new_features: Current feature matrix.
            new_ic: Current IC value (optional).

        Returns:
            List of DriftAlert objects.
        """
        alerts = []  # type: List[DriftAlert]

        # 1. PSI check per feature
        n_features = min(
            self.baseline_features.shape[1],
            new_features.shape[1],
        )
        for i in range(n_features):
            psi = compute_psi(
                self.baseline_features[:, i],
                new_features[:, i],
            )
            if psi > self.psi_threshold:
                severity = "high" if psi > 0.40 else "medium"
                fname = self.feature_names[i] if i < len(self.feature_names) else "feature_{}".format(i)
                alerts.append(DriftAlert(
                    alert_type="psi",
                    severity=severity,
                    feature_name=fname,
                    current_value=psi,
                    threshold=self.psi_threshold,
                    message="PSI={:.3f} for '{}' exceeds threshold {:.3f}".format(
                        psi, fname, self.psi_threshold),
                ))

        # 2. KL divergence check
        kl_values = compute_feature_kl_divergence(
            self.baseline_features[:, :n_features],
            new_features[:, :n_features],
            self.feature_names[:n_features],
        )
        for fname, kl in kl_values.items():
            if kl > self.kl_threshold:
                alerts.append(DriftAlert(
                    alert_type="kl_divergence",
                    severity="medium",
                    feature_name=fname,
                    current_value=kl,
                    threshold=self.kl_threshold,
                    message="KL divergence={:.3f} for '{}' exceeds threshold {:.3f}".format(
                        kl, fname, self.kl_threshold),
                ))

        # 3. IC decay check
        if new_ic is not None:
            self._ic_history.append(new_ic)
            if len(self._ic_history) >= 10:
                decayed = detect_rolling_ic_decay(
                    np.array(self._ic_history),
                    window=min(63, len(self._ic_history)),
                    decay_threshold=self.ic_decay_threshold,
                )
                if decayed:
                    alerts.append(DriftAlert(
                        alert_type="ic_decay",
                        severity="high",
                        feature_name="rolling_ic",
                        current_value=float(np.mean(self._ic_history[-10:])),
                        threshold=self.baseline_ic * self.ic_decay_threshold,
                        message="IC has decayed: recent={:.4f}, baseline={:.4f}".format(
                            np.mean(self._ic_history[-10:]), self.baseline_ic),
                    ))

        return alerts

    def should_retrain(self, alerts: List[DriftAlert]) -> bool:
        """Determine if retraining is needed based on alerts.

        Returns True if any HIGH severity alert is present.
        """
        return any(a.severity == "high" for a in alerts)
