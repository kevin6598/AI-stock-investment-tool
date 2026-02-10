"""
Performance Monitoring
-----------------------
Detect model degradation and feature drift to trigger retraining.

Components:
  - DriftAlert: structured alert with severity levels
  - ModelMonitor: tracks prediction drift, feature drift (PSI),
    and generates monitoring reports
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Structured drift/degradation alert."""
    alert_type: str          # "prediction_drift", "feature_drift", "performance_drop"
    severity: str            # "low", "medium", "high"
    metric_name: str
    current_value: float
    threshold: float
    message: str


class ModelMonitor:
    """Monitor model performance and detect drift.

    Usage:
        monitor = ModelMonitor()
        monitor.set_baseline(train_features)
        alerts = monitor.check_prediction_drift(predictions, actuals)
        alerts += monitor.check_feature_drift(new_features)
        report = monitor.generate_report()
    """

    def __init__(
        self,
        ic_threshold: float = 0.03,
        psi_threshold: float = 0.20,
        performance_window: int = 63,
    ):
        """
        Args:
            ic_threshold: minimum acceptable rolling IC.
            psi_threshold: PSI threshold for feature drift (>0.20 = significant).
            performance_window: rolling window for metric computation.
        """
        self.ic_threshold = ic_threshold
        self.psi_threshold = psi_threshold
        self.window = performance_window
        self._baseline_distributions = {}  # type: Dict[str, np.ndarray]
        self._prediction_history = []  # type: List[float]
        self._actual_history = []  # type: List[float]
        self._alerts = []  # type: List[DriftAlert]
        self._drift_detector = None  # Optional[DriftDetector]

    def set_drift_detector(
        self,
        baseline_features: np.ndarray,
        baseline_ic: float,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Configure the enhanced drift detector.

        Args:
            baseline_features: Training feature matrix.
            baseline_ic: Baseline IC from training.
            feature_names: Feature names.
        """
        try:
            from training.drift import DriftDetector
            self._drift_detector = DriftDetector(
                baseline_features=baseline_features,
                baseline_ic=baseline_ic,
                feature_names=feature_names,
                psi_threshold=self.psi_threshold,
            )
        except ImportError:
            logger.debug("training.drift not available")

    def set_baseline(self, features: np.ndarray) -> None:
        """Store reference feature distributions for PSI computation.

        Args:
            features: training feature matrix (n_samples, n_features).
        """
        self._baseline_distributions = {}
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            col = col[~np.isnan(col)]
            if len(col) > 0:
                self._baseline_distributions[col_idx] = col

    def check_prediction_drift(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> List[DriftAlert]:
        """Check for prediction quality degradation via rolling IC.

        Args:
            predictions: recent model predictions.
            actuals: corresponding realized values.

        Returns:
            List of drift alerts (if any).
        """
        alerts = []

        self._prediction_history.extend(predictions.tolist())
        self._actual_history.extend(actuals.tolist())

        if len(self._prediction_history) < self.window:
            return alerts

        recent_preds = np.array(self._prediction_history[-self.window:])
        recent_actuals = np.array(self._actual_history[-self.window:])

        # Remove NaN
        mask = ~(np.isnan(recent_preds) | np.isnan(recent_actuals))
        if mask.sum() < 10:
            return alerts

        from scipy import stats
        ic, _ = stats.spearmanr(recent_preds[mask], recent_actuals[mask])

        if ic < self.ic_threshold:
            severity = "high" if ic < 0 else "medium" if ic < self.ic_threshold / 2 else "low"
            alert = DriftAlert(
                alert_type="prediction_drift",
                severity=severity,
                metric_name="rolling_ic",
                current_value=float(ic),
                threshold=self.ic_threshold,
                message=f"Rolling IC ({ic:.4f}) below threshold ({self.ic_threshold:.4f})",
            )
            alerts.append(alert)
            self._alerts.append(alert)

        # Check hit ratio
        correct = np.sum(np.sign(recent_preds[mask]) == np.sign(recent_actuals[mask]))
        hit_ratio = correct / mask.sum()
        if hit_ratio < 0.48:
            alert = DriftAlert(
                alert_type="performance_drop",
                severity="medium" if hit_ratio > 0.45 else "high",
                metric_name="hit_ratio",
                current_value=float(hit_ratio),
                threshold=0.48,
                message=f"Hit ratio ({hit_ratio:.3f}) below 48%",
            )
            alerts.append(alert)
            self._alerts.append(alert)

        return alerts

    def check_feature_drift(
        self,
        features: np.ndarray,
        new_ic: Optional[float] = None,
    ) -> List[DriftAlert]:
        """Check for feature distribution drift using PSI.

        Population Stability Index (PSI) measures the shift between
        the baseline and current feature distributions.

        PSI < 0.10: no significant change
        PSI 0.10-0.20: moderate drift
        PSI > 0.20: significant drift

        Also uses DriftDetector for richer alerts if configured.

        Args:
            features: current feature matrix (n_samples, n_features).
            new_ic: optional current IC value for decay detection.

        Returns:
            List of drift alerts.
        """
        alerts = []

        # Use enhanced DriftDetector if available
        if self._drift_detector is not None:
            try:
                from training.drift import DriftAlert as DriftAlertExt
                ext_alerts = self._drift_detector.check(features, new_ic)
                for ea in ext_alerts:
                    alert = DriftAlert(
                        alert_type=ea.alert_type,
                        severity=ea.severity,
                        metric_name=ea.feature_name,
                        current_value=ea.current_value,
                        threshold=ea.threshold,
                        message=ea.message,
                    )
                    alerts.append(alert)
                    self._alerts.append(alert)
                return alerts
            except Exception:
                pass  # fall through to legacy detection

        if not self._baseline_distributions:
            return alerts

        for col_idx, baseline in self._baseline_distributions.items():
            if col_idx >= features.shape[1]:
                continue

            current = features[:, col_idx]
            current = current[~np.isnan(current)]
            if len(current) < 10:
                continue

            psi = self._compute_psi(baseline, current)

            if psi > self.psi_threshold:
                severity = "high" if psi > 0.40 else "medium"
                alert = DriftAlert(
                    alert_type="feature_drift",
                    severity=severity,
                    metric_name="psi_feature_{}".format(col_idx),
                    current_value=float(psi),
                    threshold=self.psi_threshold,
                    message="Feature {}: PSI={:.3f} exceeds threshold {:.3f}".format(
                        col_idx, psi, self.psi_threshold),
                )
                alerts.append(alert)
                self._alerts.append(alert)

        return alerts

    @staticmethod
    def _compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions.

        Args:
            baseline: reference distribution.
            current: current distribution.
            n_bins: number of histogram bins.

        Returns:
            PSI value.
        """
        # Create bins from baseline
        edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Make edges unique
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0

        base_counts, _ = np.histogram(baseline, bins=edges)
        curr_counts, _ = np.histogram(current, bins=edges)

        # Convert to proportions with smoothing
        eps = 1e-6
        base_pct = base_counts / max(len(baseline), 1) + eps
        curr_pct = curr_counts / max(len(current), 1) + eps

        # PSI = sum((current - baseline) * ln(current / baseline))
        psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
        return max(psi, 0.0)

    def generate_report(self) -> Dict:
        """Generate a full monitoring report.

        Returns:
            Dict with:
              - n_alerts: total alerts
              - alerts_by_severity: count per severity level
              - alerts_by_type: count per alert type
              - latest_alerts: most recent alerts
              - rolling_ic: latest IC value (if available)
        """
        by_severity = {"low": 0, "medium": 0, "high": 0}
        by_type = {}  # type: Dict[str, int]

        for alert in self._alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1

        # Latest rolling IC
        rolling_ic = None
        if len(self._prediction_history) >= self.window and len(self._actual_history) >= self.window:
            from scipy import stats
            p = np.array(self._prediction_history[-self.window:])
            a = np.array(self._actual_history[-self.window:])
            mask = ~(np.isnan(p) | np.isnan(a))
            if mask.sum() > 5:
                rolling_ic, _ = stats.spearmanr(p[mask], a[mask])
                rolling_ic = float(rolling_ic)

        return {
            "n_alerts": len(self._alerts),
            "alerts_by_severity": by_severity,
            "alerts_by_type": by_type,
            "latest_alerts": [
                {
                    "type": a.alert_type,
                    "severity": a.severity,
                    "metric": a.metric_name,
                    "value": a.current_value,
                    "message": a.message,
                }
                for a in self._alerts[-10:]  # last 10
            ],
            "rolling_ic": rolling_ic,
        }


# ---------------------------------------------------------------------------
# Model Drift Detection
# ---------------------------------------------------------------------------

def detect_model_drift(
    recent_ic: float,
    baseline_ic: float,
    threshold: float = 0.5,
) -> bool:
    """Detect if model performance has drifted below acceptable level.

    Args:
        recent_ic: recent rolling IC value.
        baseline_ic: baseline IC from training evaluation.
        threshold: fraction of baseline IC below which drift is flagged.

    Returns:
        True if recent IC dropped below threshold * baseline IC.
    """
    if baseline_ic <= 0:
        return recent_ic < 0
    return recent_ic < threshold * baseline_ic


# ---------------------------------------------------------------------------
# Retraining Scheduler
# ---------------------------------------------------------------------------

class RetrainingScheduler:
    """Determines when a model should be retrained.

    Rules:
      - Retrain if > max_days since last training
      - Retrain if model drift detected
      - Retrain if regime changed since last training

    Usage:
        scheduler = RetrainingScheduler()
        if scheduler.should_retrain(last_train_date, drift_detected, regime_change):
            # trigger retraining
    """

    def __init__(
        self,
        max_days: int = 90,
    ):
        """
        Args:
            max_days: maximum days between retraining cycles.
        """
        self.max_days = max_days

    def should_retrain(
        self,
        last_train_date: Optional[datetime] = None,
        drift_detected: bool = False,
        regime_change: bool = False,
    ) -> bool:
        """Check whether retraining should be triggered.

        Args:
            last_train_date: when the model was last trained.
            drift_detected: whether model drift has been detected.
            regime_change: whether market regime has changed since last train.

        Returns:
            True if retraining is recommended.
        """
        # Drift detected -> retrain immediately
        if drift_detected:
            logger.info("Retraining triggered: model drift detected")
            return True

        # Regime change -> retrain
        if regime_change:
            logger.info("Retraining triggered: regime change detected")
            return True

        # Time-based: retrain if too long since last training
        if last_train_date is not None:
            days_since = (datetime.now() - last_train_date).days
            if days_since > self.max_days:
                logger.info(
                    f"Retraining triggered: {days_since} days since last train "
                    f"(max={self.max_days})"
                )
                return True

        return False


class WeeklyRetrainingTrigger:
    """Checks whether a weekly retraining cycle should execute.

    Combines time-based schedule with drift and regime checks.
    Runs every Sunday (or the specified weekday) unless suppressed
    by a lock file.

    Usage:
        trigger = WeeklyRetrainingTrigger()
        if trigger.should_run():
            # run retraining pipeline
            trigger.record_run()
    """

    def __init__(
        self,
        weekday: int = 6,  # 0=Mon, 6=Sun
        lock_file: str = ".retrain_lock",
        min_hours_between: float = 120.0,  # ~5 days
    ):
        self.weekday = weekday
        self.lock_file = lock_file
        self.min_hours_between = min_hours_between

    def should_run(
        self,
        drift_detected: bool = False,
        regime_change: bool = False,
    ) -> bool:
        """Check if retraining should trigger.

        Args:
            drift_detected: Force retrain on model drift.
            regime_change: Force retrain on regime change.

        Returns:
            True if retraining should proceed.
        """
        import os

        # Drift or regime change -> immediate retrain
        if drift_detected:
            logger.info("Weekly trigger: drift detected, forcing retrain")
            return True
        if regime_change:
            logger.info("Weekly trigger: regime changed, forcing retrain")
            return True

        # Check time since last run
        if os.path.exists(self.lock_file):
            try:
                mtime = os.path.getmtime(self.lock_file)
                last_run = datetime.fromtimestamp(mtime)
                hours_since = (datetime.now() - last_run).total_seconds() / 3600
                if hours_since < self.min_hours_between:
                    logger.debug(
                        f"Weekly trigger: {hours_since:.1f}h since last run "
                        f"(min={self.min_hours_between}h)"
                    )
                    return False
            except Exception:
                pass

        # Check if it's the right weekday
        now = datetime.now()
        if now.weekday() == self.weekday:
            logger.info(f"Weekly trigger: weekday {self.weekday} matched, time to retrain")
            return True

        return False

    def record_run(self):
        """Record that a retraining run completed (update lock file)."""
        import os
        with open(self.lock_file, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Weekly trigger: recorded run timestamp")
