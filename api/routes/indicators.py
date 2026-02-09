"""
Technical Indicator Routes
---------------------------
GET /api/v1/indicators/{ticker} - Technical indicator data
"""

from typing import List, Optional
import logging

from fastapi import APIRouter, HTTPException, Query

from api.schemas import IndicatorResponse, IndicatorSeries, IndicatorValue

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["indicators"])

# Available indicators for the API
_INDICATOR_GROUPS = {
    "moving_averages": ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26"],
    "momentum": ["rsi_14", "macd", "macd_signal", "macd_histogram", "stoch_k", "stoch_d"],
    "volatility": ["bb_upper", "bb_lower", "bb_width", "atr_14", "vol_cluster_ew"],
    "volume": ["volume_ratio", "obv", "vwap_deviation", "volume_spike_intensity"],
    "trend": ["adx", "cci_20", "regime_trend"],
}


@router.get("/indicators/{ticker}", response_model=IndicatorResponse)
async def get_indicators(
    ticker: str,
    period: str = Query(default="1y", regex="^(3mo|6mo|1y|2y|5y)$"),
    indicators: Optional[str] = Query(
        default=None,
        description="Comma-separated indicator names or group names",
    ),
):
    """Get technical indicators for a ticker."""
    try:
        from data.stock_api import get_historical_data
        from training.feature_engineering import compute_technical_features

        stock_df = get_historical_data(ticker, period=period)
        if stock_df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        tech = compute_technical_features(stock_df)

        # Determine which indicators to return
        if indicators:
            requested = [s.strip() for s in indicators.split(",")]
        else:
            requested = ["sma_20", "sma_50", "rsi_14", "macd", "bb_upper", "bb_lower"]

        # Expand group names
        expanded = []
        for name in requested:
            if name in _INDICATOR_GROUPS:
                expanded.extend(_INDICATOR_GROUPS[name])
            else:
                expanded.append(name)

        # Build response
        series_list = []
        for col_name in expanded:
            if col_name in tech.columns:
                values = []
                for date, val in tech[col_name].dropna().items():
                    values.append(IndicatorValue(
                        date=str(date.date()) if hasattr(date, 'date') else str(date),
                        value=round(float(val), 6),
                    ))
                series_list.append(IndicatorSeries(name=col_name, values=values))

        return IndicatorResponse(
            ticker=ticker.upper(),
            indicators=series_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indicator fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{ticker}/groups")
async def get_indicator_groups(ticker: str):
    """List available indicator groups."""
    return {"ticker": ticker.upper(), "groups": _INDICATOR_GROUPS}
