"""
Serving Feature Pipeline
-------------------------
Reuses build_feature_matrix() from training, applies saved scaler,
ensures column ordering matches training.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingFeaturePipeline:
    """Feature computation pipeline for model serving.

    Reuses the training feature engineering code to ensure consistency
    between training and serving features.

    Args:
        feature_columns: Ordered list of feature columns from training.
        feature_scaler: Fitted scaler from training (optional).
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        feature_scaler=None,
    ):
        self.feature_columns = feature_columns or []
        self.feature_scaler = feature_scaler

    def compute_features(
        self,
        ticker: str,
        period: str = "5y",
        horizon_days: int = 21,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute features for a single ticker.

        Args:
            ticker: Stock ticker symbol.
            period: Historical data period.
            horizon_days: Forward return horizon.

        Returns:
            (X_array, feature_df) where X_array is scaled and column-aligned.
        """
        from data.stock_api import get_historical_data, get_stock_info
        from training.feature_engineering import build_feature_matrix

        # Fetch data
        stock_df = get_historical_data(ticker, period=period)
        if stock_df.empty:
            raise ValueError(f"No data available for {ticker}")

        stock_info = get_stock_info(ticker) or {}

        # Fetch market data for macro features
        market_df = get_historical_data("SPY", period=period)
        if market_df.empty:
            market_df = None

        # Build features using the same pipeline as training
        feat = build_feature_matrix(
            stock_df, stock_info, market_df,
            forward_horizons=[horizon_days],
            ticker=ticker,
        )

        if feat.empty:
            raise ValueError(f"Feature matrix empty for {ticker}")

        # Get feature columns (exclude targets and metadata)
        all_feature_cols = [c for c in feat.columns
                           if not c.startswith("fwd_return_") and c != "_close"]

        # Align columns to match training order
        if self.feature_columns:
            # Add missing columns with zeros, drop extra columns
            for col in self.feature_columns:
                if col not in feat.columns:
                    feat[col] = 0.0
            X_df = feat[self.feature_columns]
        else:
            X_df = feat[all_feature_cols]

        X = X_df.values.astype(np.float32)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply scaler if available
        if self.feature_scaler is not None:
            try:
                X = self.feature_scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")

        return X, feat

    def compute_features_for_sequence(
        self,
        ticker: str,
        seq_len: int = 60,
        period: str = "5y",
        horizon_days: int = 21,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Compute features with sequence context for sequential models.

        Returns:
            (X_seq, X_static, feature_df) where:
              X_seq: (1, seq_len, n_features) for temporal encoder
              X_static: (1, n_features) latest features
              feature_df: full DataFrame for indicator display
        """
        X, feat = self.compute_features(ticker, period, horizon_days)

        if X.shape[0] < seq_len:
            # Pad with zeros
            pad_len = seq_len - X.shape[0]
            X = np.vstack([np.zeros((pad_len, X.shape[1]), dtype=np.float32), X])

        # Take last seq_len rows for sequence input
        X_seq = X[-seq_len:].reshape(1, seq_len, -1)
        X_static = X[-1:].reshape(1, -1)

        return X_seq, X_static, feat

    def compute_meta_features(
        self,
        point_estimate: float,
        uncertainty: float,
        confidence: float,
    ) -> np.ndarray:
        """Compute meta-labeling features for a single prediction.

        Args:
            point_estimate: Base model point prediction.
            uncertainty: Model uncertainty estimate.
            confidence: Model confidence score.

        Returns:
            Feature array for meta-label model.
        """
        return np.array([[
            abs(point_estimate),
            uncertainty,
            confidence,
            1.0 if point_estimate > 0 else 0.0,
            abs(point_estimate) / max(uncertainty, 1e-8),
        ]], dtype=np.float32)

    def get_indicators(
        self,
        ticker: str,
        period: str = "2y",
    ) -> pd.DataFrame:
        """Get technical indicators for display.

        Returns:
            DataFrame with indicator columns and DatetimeIndex.
        """
        from data.stock_api import get_historical_data
        from training.feature_engineering import compute_technical_features

        stock_df = get_historical_data(ticker, period=period)
        if stock_df.empty:
            raise ValueError(f"No data for {ticker}")

        tech = compute_technical_features(stock_df)
        return tech
