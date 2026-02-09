"""
Inference Pipeline
-------------------
End-to-end daily inference pipeline that orchestrates:
  1. Data fetching
  2. Feature engineering
  3. Model prediction (point estimates + quantiles)
  4. Regime classification
  5. CVaR portfolio optimization
  6. Risk engine evaluation

Produces a structured InferenceResult with per-ticker signals and
portfolio-level risk metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TickerSignal:
    """Per-ticker inference output."""
    ticker: str
    predicted_return: float
    return_std: float
    confidence_interval: Tuple[float, float]
    position_weight: float
    risk_adjusted_weight: float
    regime: str
    confidence_score: float
    cvar_contribution: float


@dataclass
class InferenceResult:
    """Full inference pipeline output."""
    timestamp: str
    signals: List[TickerSignal]
    regime: str
    portfolio_cvar_95: float
    portfolio_vol: float
    risk_off_mode: bool
    total_confidence: float


class ModelLoader:
    """Load pre-trained models from the registry or file paths.

    Supports loading from:
      - Model registry (by type/horizon or version ID)
      - File paths (exported Colab artifacts)
      - Auto-selection by best metric

    Usage:
        loader = ModelLoader()
        model = loader.load_from_registry("lightgbm", "1M")
        model = loader.load_from_path("models_registry/lightgbm_v0/model.pkl")
        model = loader.auto_select("1M", metric="mean_ic")
    """

    def load_from_registry(
        self, model_type: str, horizon: str,
    ) -> Optional[Any]:
        """Load active model from registry.

        Args:
            model_type: e.g. "lightgbm", "lstm_attention".
            horizon: e.g. "1M", "3M".

        Returns:
            Model instance or None if not found.
        """
        try:
            from training.model_versioning import ModelRegistry
            registry = ModelRegistry()
            version = registry.get_active_model(model_type, horizon)
            if version is None:
                logger.info(f"No active {model_type}/{horizon} model in registry")
                return None
            model = registry.load_model(version.version_id)
            logger.info(f"Loaded model from registry: {version.version_id}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load from registry: {e}")
            return None

    def load_from_path(self, path: str) -> Optional[Any]:
        """Load model from file path (exported Colab artifact).

        Args:
            path: path to model file (.pkl or .pt).

        Returns:
            Model instance or None on failure.
        """
        import os
        import pickle

        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return None

        try:
            if path.endswith(".pt"):
                import torch
                return torch.load(path, map_location="cpu")
            else:
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}")
            return None

    def auto_select(
        self, horizon: str, metric: str = "mean_ic",
    ) -> Optional[Any]:
        """Auto-select best model for a horizon by metric.

        Args:
            horizon: forecast horizon.
            metric: metric to maximize.

        Returns:
            Best model instance or None.
        """
        try:
            from training.model_versioning import ModelRegistry
            registry = ModelRegistry()
            best_version = registry.get_best_model(horizon=horizon, metric=metric)
            if best_version is None:
                return None
            model = registry.load_model(best_version.version_id)
            logger.info(
                f"Auto-selected: {best_version.version_id} "
                f"({metric}={best_version.metrics.get(metric, 'N/A')})"
            )
            return model
        except Exception as e:
            logger.warning(f"Auto-select failed: {e}")
            return None


class InferencePipeline:
    """End-to-end inference pipeline.

    Orchestrates data fetching, feature engineering, model prediction,
    regime classification, portfolio optimization, and risk controls.

    Usage:
        pipeline = InferencePipeline(
            tickers=["AAPL", "MSFT", "GOOGL"],
            model_type="lightgbm",
        )
        result = pipeline.run("1M")

        # With pre-trained model:
        loader = ModelLoader()
        pipeline = InferencePipeline(
            tickers=["AAPL", "MSFT"],
            model_loader=loader,
        )
        result = pipeline.run("1M")
    """

    def __init__(
        self,
        tickers: List[str],
        model_type: str = "lightgbm",
        market_ticker: str = "SPY",
        data_period: str = "5y",
        model_loader: Optional[ModelLoader] = None,
    ):
        self.tickers = [t.upper() for t in tickers]
        self.model_type = model_type
        self.market_ticker = market_ticker.upper()
        self.data_period = data_period
        self.model_loader = model_loader

    def run(self, horizon_name: str = "1M") -> InferenceResult:
        """Execute the full inference pipeline.

        Args:
            horizon_name: Forecast horizon ("1M", "3M", "6M").

        Returns:
            InferenceResult with signals and risk metrics.
        """
        from data.stock_api import get_historical_data, get_stock_info
        from training.feature_engineering import (
            build_feature_matrix, cross_sectional_normalize,
        )
        from training.models import create_model
        from training.weight_optimizer import RegimeClassifier
        from engine.cvar_optimizer import CVaROptimizer, PortfolioConstraints
        from engine.risk_engine import RiskEngine

        horizon_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
        horizon_days = horizon_map.get(horizon_name, 21)

        logger.info(f"Running inference: {self.tickers}, horizon={horizon_name}")

        # 1. Fetch data
        stock_dfs = {}
        stock_infos = {}
        for ticker in self.tickers:
            df = get_historical_data(ticker, period=self.data_period)
            if not df.empty:
                stock_dfs[ticker] = df
                info = get_stock_info(ticker) or {}
                stock_infos[ticker] = info

        market_df = get_historical_data(self.market_ticker, period=self.data_period)
        if market_df.empty:
            market_df = None

        if not stock_dfs:
            raise ValueError("No stock data fetched")

        # 2. Build features & train model per ticker
        predictions = {}  # ticker -> (mean, std)
        all_means = []
        valid_tickers = []

        for ticker, df in stock_dfs.items():
            try:
                info = stock_infos.get(ticker, {})
                feat = build_feature_matrix(
                    df, info, market_df, [horizon_days], ticker=ticker,
                )
                if feat.empty or len(feat) < 100:
                    logger.warning(f"Insufficient features for {ticker}")
                    continue

                target_col = f"fwd_return_{horizon_days}d"
                feature_cols = [c for c in feat.columns
                                if not c.startswith("fwd_return_") and c != "_close"]

                X = feat[feature_cols].values.astype(np.float32)
                y = feat[target_col].values.astype(np.float32)
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                # Train/val split (use last portion for prediction)
                split = int(len(X) * 0.8)
                val_split = int(len(X) * 0.9)

                # Try loading pre-trained model first
                pre_trained = None
                if self.model_loader is not None:
                    pre_trained = self.model_loader.auto_select(horizon_name)
                    if pre_trained is None:
                        pre_trained = self.model_loader.load_from_registry(
                            self.model_type, horizon_name,
                        )

                if pre_trained is not None:
                    model = pre_trained
                else:
                    model = create_model(self.model_type)
                    model.fit(
                        X[:split], y[:split],
                        X[split:val_split], y[split:val_split],
                        feature_names=feature_cols,
                    )

                # Predict on most recent data
                recent_X = X[-1:].reshape(1, -1)
                point_pred = model.predict(recent_X)
                quantile_pred = model.predict_quantiles(recent_X, [0.10, 0.50, 0.90])

                # Extract point estimate
                pred_val = float(np.nanmean(point_pred))
                # Estimate std from quantile spread
                p10 = float(np.nanmean(quantile_pred.get(0.10, [0.0])))
                p90 = float(np.nanmean(quantile_pred.get(0.90, [0.0])))
                pred_std = max((p90 - p10) / 2.56, 1e-6)  # ~80% CI width / z

                predictions[ticker] = (pred_val, pred_std)
                all_means.append(pred_val)
                valid_tickers.append(ticker)

            except Exception as e:
                logger.warning(f"Prediction failed for {ticker}: {e}")
                continue

        if not valid_tickers:
            raise ValueError("No valid predictions generated")

        # 3. Regime classification
        regime_str = "normal"
        if market_df is not None:
            try:
                classifier = RegimeClassifier()
                regime_info = classifier.classify(market_df)
                regime_str = regime_info.regime
            except Exception as e:
                logger.warning(f"Regime classification failed: {e}")

        # 4. CVaR portfolio optimization
        n = len(valid_tickers)
        means = np.array([predictions[t][0] for t in valid_tickers])
        stds = np.array([predictions[t][1] for t in valid_tickers])
        # Build covariance: use predicted variances + assumed correlation of 0.3
        corr = np.full((n, n), 0.3)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(stds, stds) * corr

        optimizer = CVaROptimizer()
        constraints = PortfolioConstraints()
        port_result = optimizer.optimize(
            valid_tickers, means, cov, constraints,
        )

        # 5. Risk engine
        risk_engine = RiskEngine()
        risk_report = risk_engine.evaluate(
            proposed_weights=port_result.weights,
            covariance=cov,
            tickers=valid_tickers,
            corr_matrix=corr,
        )

        # 6. Build signals
        signals = []
        total_conf = 0.0
        for ticker in valid_tickers:
            pred_mean, pred_std = predictions[ticker]
            raw_weight = port_result.weights.get(ticker, 0.0)
            adj_weight = risk_report.adjusted_weights.get(ticker, 0.0)

            ci_low = pred_mean - 1.96 * pred_std
            ci_high = pred_mean + 1.96 * pred_std

            # Confidence: based on prediction std relative to mean
            conf = max(0.0, min(1.0, 1.0 - pred_std / max(abs(pred_mean), 1e-6)))
            total_conf += conf

            # CVaR contribution (proportional to weight * std)
            cvar_contrib = adj_weight * pred_std

            signals.append(TickerSignal(
                ticker=ticker,
                predicted_return=pred_mean,
                return_std=pred_std,
                confidence_interval=(ci_low, ci_high),
                position_weight=raw_weight,
                risk_adjusted_weight=adj_weight,
                regime=regime_str,
                confidence_score=conf,
                cvar_contribution=cvar_contrib,
            ))

        avg_conf = total_conf / len(signals) if signals else 0.0

        return InferenceResult(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            signals=signals,
            regime=regime_str,
            portfolio_cvar_95=port_result.cvar_95,
            portfolio_vol=risk_report.vol,
            risk_off_mode=risk_report.risk_off_mode,
            total_confidence=avg_conf,
        )
