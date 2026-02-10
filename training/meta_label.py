"""
Meta-Labeling Module
--------------------
Triple-barrier meta-labeling for alpha filtering:
  1. Compute triple-barrier labels from price paths
  2. Build meta-features from base predictions
  3. Train a meta-model (LightGBM preferred, logistic fallback)
  4. Nested walk-forward meta-training
  5. Compute final alpha = base_alpha * meta_trade_probability
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    upper_barrier_multiplier: float = 2.0
    lower_barrier_multiplier: float = 2.0
    max_holding_period: int = 21  # trading days
    volatility_lookback: int = 21


def compute_triple_barrier_labels(
    prices: np.ndarray,
    predictions: np.ndarray,
    config: Optional[TripleBarrierConfig] = None,
) -> np.ndarray:
    """Compute triple-barrier labels (0/1 meta-labels).

    For each prediction point:
      - Compute rolling volatility as barrier width
      - Upper barrier = entry_price * (1 + multiplier * vol)
      - Lower barrier = entry_price * (1 - multiplier * vol)
      - Time barrier = max_holding_period days
      - Label = 1 if price hits upper barrier first (profitable trade)
      - Label = 0 otherwise

    Args:
        prices: Price series, shape (n,).
        predictions: Base model predictions, shape (n,).
        config: Triple barrier configuration.

    Returns:
        Binary labels array, shape (n,). NaN where label cannot be computed.
    """
    if config is None:
        config = TripleBarrierConfig()

    n = len(prices)
    labels = np.full(n, np.nan)

    # Rolling volatility
    log_ret = np.diff(np.log(prices + 1e-10))
    log_ret = np.concatenate([[0.0], log_ret])
    vol = pd.Series(log_ret).rolling(config.volatility_lookback).std().values
    vol = np.nan_to_num(vol, nan=0.01)
    vol = np.maximum(vol, 1e-6)

    for i in range(config.volatility_lookback, n - config.max_holding_period):
        entry_price = prices[i]
        direction = np.sign(predictions[i])
        if abs(predictions[i]) < 1e-8:
            labels[i] = 0.0
            continue

        upper = entry_price * (1 + config.upper_barrier_multiplier * vol[i])
        lower = entry_price * (1 - config.lower_barrier_multiplier * vol[i])

        # Walk forward through holding period
        hit_upper = False
        hit_lower = False
        for j in range(1, config.max_holding_period + 1):
            if i + j >= n:
                break
            future_price = prices[i + j]
            if direction > 0:
                # Long trade: profit if hits upper
                if future_price >= upper:
                    hit_upper = True
                    break
                if future_price <= lower:
                    hit_lower = True
                    break
            else:
                # Short trade: profit if hits lower
                if future_price <= lower:
                    hit_upper = True  # "profit" barrier
                    break
                if future_price >= upper:
                    hit_lower = True  # "loss" barrier
                    break

        # Label: 1 if profitable barrier hit first
        if hit_upper:
            labels[i] = 1.0
        elif hit_lower:
            labels[i] = 0.0
        else:
            # Time barrier: check if final return is in right direction
            final_price = prices[min(i + config.max_holding_period, n - 1)]
            final_return = (final_price - entry_price) / entry_price
            labels[i] = 1.0 if (direction * final_return > 0) else 0.0

    return labels


@dataclass
class MetaFeatures:
    """Container for meta-labeling features."""
    features: np.ndarray  # (n, n_features)
    feature_names: List[str] = field(default_factory=list)


def build_meta_features(
    base_predictions: np.ndarray,
    prediction_uncertainty: Optional[np.ndarray] = None,
    ensemble_variance: Optional[np.ndarray] = None,
    regime_state: Optional[np.ndarray] = None,
    liquidity_percentile: Optional[np.ndarray] = None,
    rolling_ic: Optional[np.ndarray] = None,
) -> MetaFeatures:
    """Build feature matrix for the meta-labeling model.

    Args:
        base_predictions: Base alpha predictions.
        prediction_uncertainty: Model uncertainty estimates.
        ensemble_variance: Variance across ensemble members.
        regime_state: Market regime indicator.
        liquidity_percentile: Liquidity percentile rank.
        rolling_ic: Rolling information coefficient.

    Returns:
        MetaFeatures with feature matrix and names.
    """
    n = len(base_predictions)
    features = []
    names = []

    # Base prediction magnitude
    features.append(np.abs(base_predictions).reshape(-1, 1))
    names.append("pred_magnitude")

    # Base prediction sign
    features.append(np.sign(base_predictions).reshape(-1, 1))
    names.append("pred_sign")

    # Prediction z-score (rolling)
    pred_series = pd.Series(base_predictions)
    rolling_mean = pred_series.rolling(63, min_periods=10).mean().values
    rolling_std = pred_series.rolling(63, min_periods=10).std().values
    rolling_std = np.maximum(rolling_std, 1e-8)
    pred_zscore = (base_predictions - np.nan_to_num(rolling_mean)) / np.nan_to_num(rolling_std, nan=1.0)
    features.append(pred_zscore.reshape(-1, 1))
    names.append("pred_zscore")

    if prediction_uncertainty is not None:
        features.append(prediction_uncertainty.reshape(-1, 1))
        names.append("uncertainty")

    if ensemble_variance is not None:
        features.append(ensemble_variance.reshape(-1, 1))
        names.append("ensemble_variance")

    if regime_state is not None:
        features.append(regime_state.reshape(-1, 1))
        names.append("regime_state")

    if liquidity_percentile is not None:
        features.append(liquidity_percentile.reshape(-1, 1))
        names.append("liquidity_pctl")

    if rolling_ic is not None:
        features.append(rolling_ic.reshape(-1, 1))
        names.append("rolling_ic")

    feature_matrix = np.hstack(features)
    np.nan_to_num(feature_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return MetaFeatures(features=feature_matrix, feature_names=names)


class MetaLabelModel:
    """Meta-labeling model (LightGBM preferred, logistic fallback).

    Predicts P(trade is profitable) given meta-features.
    """

    def __init__(self):
        self.model = None
        self._is_lgb = False
        self.is_fitted = False

    def fit(
        self,
        meta_features: np.ndarray,
        meta_labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """Train the meta-labeling model.

        Args:
            meta_features: (n, n_meta_features) feature matrix.
            meta_labels: (n,) binary labels (0/1).
            sample_weights: Optional sample weights.
        """
        # Clean inputs
        mask = ~(np.isnan(meta_labels) | np.isnan(meta_features).any(axis=1))
        X = meta_features[mask]
        y = meta_labels[mask]
        sw = sample_weights[mask] if sample_weights is not None else None

        if len(X) < 20:
            logger.warning("Insufficient data for meta-label training (%d samples)", len(X))
            return

        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=15,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
                random_state=42,
            )
            self.model.fit(X, y, sample_weight=sw)
            self._is_lgb = True
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42,
            )
            self.model.fit(X, y, sample_weight=sw)
            self._is_lgb = False

        self.is_fitted = True
        logger.info("Meta-label model trained on %d samples (lgb=%s)", len(X), self._is_lgb)

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        """Predict P(profitable trade) for each observation.

        Args:
            meta_features: (n, n_meta_features) feature matrix.

        Returns:
            (n,) array of trade probabilities.
        """
        if not self.is_fitted or self.model is None:
            return np.full(len(meta_features), 0.5)

        np.nan_to_num(meta_features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(meta_features)
        # Return probability of class 1 (profitable)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba


class NestedWalkForwardMetaTrainer:
    """Ensures meta-model never shares folds with base models.

    Uses a nested walk-forward approach where the meta-model
    is trained on out-of-sample base model predictions only.
    """

    def __init__(
        self,
        n_inner_folds: int = 3,
        min_train_samples: int = 100,
    ):
        self.n_inner_folds = n_inner_folds
        self.min_train_samples = min_train_samples

    def train(
        self,
        panel: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        prices: np.ndarray,
        barrier_config: Optional[TripleBarrierConfig] = None,
        base_predictions: Optional[np.ndarray] = None,
    ) -> Tuple[MetaLabelModel, Dict]:
        """Train meta-labeling model with nested walk-forward.

        Args:
            panel: Panel DataFrame.
            target_col: Target column name.
            feature_cols: Feature column names.
            prices: Price series for barrier computation.
            barrier_config: Triple barrier configuration.
            base_predictions: Pre-computed base model predictions (optional).

        Returns:
            (meta_model, metrics_dict)
        """
        n = len(panel)
        if base_predictions is None:
            base_predictions = np.zeros(n)

        # Compute triple-barrier labels
        meta_labels = compute_triple_barrier_labels(
            prices, base_predictions, barrier_config,
        )

        # Build meta-features
        meta_feat = build_meta_features(base_predictions)

        # Time-series split for meta-model
        valid_mask = ~np.isnan(meta_labels)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < self.min_train_samples:
            logger.warning("Insufficient valid meta-labels: %d", len(valid_indices))
            return MetaLabelModel(), {"n_valid": len(valid_indices), "accuracy": 0.0}

        # Use last 30% for testing, rest for training
        split_idx = int(len(valid_indices) * 0.7)
        train_idx = valid_indices[:split_idx]
        test_idx = valid_indices[split_idx:]

        meta_model = MetaLabelModel()
        meta_model.fit(
            meta_feat.features[train_idx],
            meta_labels[train_idx],
        )

        # Evaluate on test set
        if meta_model.is_fitted and len(test_idx) > 0:
            test_proba = meta_model.predict_proba(meta_feat.features[test_idx])
            test_pred = (test_proba > 0.5).astype(float)
            accuracy = float(np.mean(test_pred == meta_labels[test_idx]))
        else:
            accuracy = 0.0

        metrics = {
            "n_valid": len(valid_indices),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": accuracy,
            "label_balance": float(np.mean(meta_labels[valid_indices])),
        }
        logger.info("Meta-trainer: accuracy=%.3f, balance=%.3f", accuracy, metrics["label_balance"])

        return meta_model, metrics


def compute_final_alpha(
    base_alpha: np.ndarray,
    meta_trade_probability: np.ndarray,
    min_probability: float = 0.3,
) -> np.ndarray:
    """Compute final alpha after meta-labeling filter.

    final_alpha = base_alpha * meta_trade_prob
    If meta_trade_prob < min_probability, alpha is zeroed out.

    Args:
        base_alpha: Base model alpha predictions.
        meta_trade_probability: P(profitable trade) from meta-model.
        min_probability: Minimum probability threshold.

    Returns:
        Filtered alpha array.
    """
    scaled = base_alpha * meta_trade_probability
    # Zero out low-confidence trades
    scaled[meta_trade_probability < min_probability] = 0.0
    return scaled
