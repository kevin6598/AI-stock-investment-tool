"""
Prediction Routes
-----------------
POST /api/v1/predict - Single ticker prediction
POST /api/v1/predict/batch - Multiple ticker predictions
"""

from typing import Optional
from datetime import datetime
import numpy as np
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import (
    PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse,
    RetailPrediction, QuantileForecast,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predict"])

# These will be set by main.py at startup
_model_cache = None
_feature_pipeline = None


def set_dependencies(model_cache, feature_pipeline):
    global _model_cache, _feature_pipeline
    _model_cache = model_cache
    _feature_pipeline = feature_pipeline


def _predict_single(ticker: str, horizon: str) -> RetailPrediction:
    """Run prediction for a single ticker."""
    horizon_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    horizon_days = horizon_map.get(horizon, 21)

    # Check if zero-shot
    is_zero_shot = False
    if _model_cache is not None and _model_cache.is_loaded:
        is_zero_shot = not _model_cache.is_trained_ticker(ticker)

    try:
        # Compute features
        if _model_cache is not None and _model_cache.is_loaded and _model_cache.model is not None:
            X_seq, X_static, feat = _feature_pipeline.compute_features_for_sequence(
                ticker, seq_len=60, period="5y", horizon_days=horizon_days,
            )

            model = _model_cache.model

            # Check if model has predict_full (hybrid model)
            if hasattr(model, 'predict_full'):
                preds = model.predict(X_static.reshape(X_static.shape[0], -1))
                q_preds = model.predict_quantiles(
                    X_static.reshape(X_static.shape[0], -1),
                    [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                )
            else:
                preds = model.predict(X_static.reshape(X_static.shape[0], -1))
                q_preds = model.predict_quantiles(
                    X_static.reshape(X_static.shape[0], -1),
                    [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                )

            point_est = float(np.nanmean(preds))
            q_vals = {q: float(np.nanmean(v)) for q, v in q_preds.items()}
        else:
            # On-demand training with lightweight model
            point_est, q_vals = _train_on_demand(ticker, horizon_days)

        # Determine direction
        if point_est > 0.005:
            direction = "UP"
        elif point_est < -0.005:
            direction = "DOWN"
        else:
            direction = "HOLD"

        # Compute p_up from point estimate and spread
        spread = q_vals.get(0.90, 0.01) - q_vals.get(0.10, -0.01)
        if spread > 0:
            p_up = max(0.0, min(1.0, 0.5 + point_est / (spread + 1e-6)))
        else:
            p_up = 0.5

        # Confidence (narrower spread = higher confidence)
        confidence = max(0.0, min(1.0, 1.0 - abs(spread) * 5))
        if is_zero_shot:
            confidence *= 0.5  # lower confidence for unseen tickers

        # Risk score
        downside = abs(q_vals.get(0.05, -0.1))
        risk_score = min(1.0, downside * 10)

        # Regime (from weight optimizer)
        regime = _get_regime(ticker)

        # Uncertainty estimation via MC dropout or quantile spread fallback
        uncertainty_val = None
        meta_prob = None
        scaled_alpha_val = None
        try:
            if _model_cache is not None and _model_cache.is_loaded and _model_cache.model is not None:
                model = _model_cache.model
                if hasattr(model, 'predict_with_uncertainty'):
                    _, unc_var = model.predict_with_uncertainty(
                        X_seq.reshape(X_seq.shape[0], -1) if 'X_seq' in dir() else X_static.reshape(X_static.shape[0], -1)
                    )
                    uncertainty_val = float(np.mean(unc_var))
                else:
                    # Fallback: use quantile spread
                    uncertainty_val = spread / 2.56 if spread > 0 else 0.0

                # Meta-labeling probability
                if _model_cache.meta_model is not None:
                    meta_features = _feature_pipeline.compute_meta_features(
                        point_est, uncertainty_val, confidence,
                    )
                    meta_prob = float(_model_cache.meta_model.predict_proba(meta_features)[0])

                # Scaled alpha
                if uncertainty_val is not None:
                    base_alpha = point_est
                    mp = meta_prob if meta_prob is not None else 1.0
                    scaled_alpha_val = float(base_alpha * mp / (1.0 + uncertainty_val))
        except Exception as e:
            logger.debug("Uncertainty/meta computation skipped: %s", e)

        return RetailPrediction(
            ticker=ticker.upper(),
            horizon=horizon,
            direction=direction,
            p_up=round(p_up, 4),
            confidence=round(confidence, 4),
            point_estimate=round(point_est, 6),
            quantiles=QuantileForecast(
                q05=round(q_vals.get(0.05, -0.1), 6),
                q10=round(q_vals.get(0.10, -0.08), 6),
                q25=round(q_vals.get(0.25, -0.03), 6),
                q50=round(q_vals.get(0.50, 0.0), 6),
                q75=round(q_vals.get(0.75, 0.03), 6),
                q90=round(q_vals.get(0.90, 0.08), 6),
                q95=round(q_vals.get(0.95, 0.1), 6),
            ),
            risk_score=round(risk_score, 4),
            hold_signal=0.0,
            regime=regime,
            is_zero_shot=is_zero_shot,
            meta_trade_probability=round(meta_prob, 4) if meta_prob is not None else None,
            uncertainty=round(uncertainty_val, 6) if uncertainty_val is not None else None,
            scaled_alpha=round(scaled_alpha_val, 6) if scaled_alpha_val is not None else None,
        )

    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def _train_on_demand(ticker: str, horizon_days: int):
    """Fallback: train a lightweight model on-demand."""
    from data.stock_api import get_historical_data, get_stock_info
    from training.feature_engineering import build_feature_matrix
    from training.models import create_model

    stock_df = get_historical_data(ticker, period="5y")
    if stock_df.empty:
        raise ValueError(f"No data for {ticker}")

    stock_info = get_stock_info(ticker) or {}
    market_df = get_historical_data("SPY", period="5y")
    if market_df.empty:
        market_df = None

    feat = build_feature_matrix(stock_df, stock_info, market_df, [horizon_days], ticker=ticker)
    if feat.empty or len(feat) < 100:
        raise ValueError(f"Insufficient data for {ticker}")

    target_col = f"fwd_return_{horizon_days}d"
    feature_cols = [c for c in feat.columns
                    if not c.startswith("fwd_return_") and c != "_close"]

    X = feat[feature_cols].values.astype(np.float32)
    y = feat[target_col].values.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    split = int(len(X) * 0.8)
    model = create_model("elastic_net")
    model.fit(X[:split], y[:split], X[split:], y[split:], feature_names=feature_cols)

    recent_X = X[-1:].reshape(1, -1)
    point_est = float(model.predict(recent_X)[0])
    q_preds = model.predict_quantiles(recent_X, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    q_vals = {q: float(np.nanmean(v)) for q, v in q_preds.items()}

    return point_est, q_vals


def _get_regime(ticker: str) -> str:
    """Get market regime for the ticker."""
    try:
        from data.stock_api import get_historical_data
        from training.weight_optimizer import RegimeClassifier
        market_df = get_historical_data("SPY", period="2y")
        if not market_df.empty:
            classifier = RegimeClassifier()
            info = classifier.classify(market_df)
            return info.regime
    except Exception:
        pass
    return "normal"


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict stock direction and returns for a single ticker."""
    prediction = _predict_single(request.ticker, request.horizon)
    return PredictResponse(
        prediction=prediction,
        model_version=_model_cache.model_version if _model_cache else None,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict for multiple tickers."""
    predictions = []
    for ticker in request.tickers:
        try:
            pred = _predict_single(ticker, request.horizon)
            predictions.append(pred)
        except HTTPException:
            logger.warning(f"Skipping {ticker}: prediction failed")
            continue

    if not predictions:
        raise HTTPException(status_code=500, detail="All predictions failed")

    return BatchPredictResponse(
        predictions=predictions,
        model_version=_model_cache.model_version if _model_cache else None,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
