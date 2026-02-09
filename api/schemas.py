"""
Pydantic v2 Schemas for the FastAPI backend.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# --- Request Models ---

class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", examples=["AAPL"])
    horizon: str = Field(
        default="1M",
        description="Forecast horizon",
        pattern="^(1M|3M|6M|1Y)$",
    )
    include_indicators: bool = Field(default=False)
    include_sentiment: bool = Field(default=False)


class BatchPredictRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols")
    horizon: str = Field(default="1M", pattern="^(1M|3M|6M|1Y)$")


# --- Response Models ---

class QuantileForecast(BaseModel):
    q05: float
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    q95: float


class RetailPrediction(BaseModel):
    ticker: str
    horizon: str
    direction: str = Field(description="UP, DOWN, or HOLD")
    p_up: float = Field(description="Probability of positive return", ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    point_estimate: float
    quantiles: QuantileForecast
    risk_score: float = Field(ge=0, le=1)
    hold_signal: float = Field(ge=0, le=1)
    regime: str
    is_zero_shot: bool = Field(
        default=False,
        description="True if ticker was not in training set",
    )


class PredictResponse(BaseModel):
    status: str = "ok"
    prediction: RetailPrediction
    model_version: Optional[str] = None
    timestamp: str


class BatchPredictResponse(BaseModel):
    status: str = "ok"
    predictions: List[RetailPrediction]
    model_version: Optional[str] = None
    timestamp: str


class IndicatorValue(BaseModel):
    date: str
    value: float


class IndicatorSeries(BaseModel):
    name: str
    values: List[IndicatorValue]


class IndicatorResponse(BaseModel):
    ticker: str
    indicators: List[IndicatorSeries]


class SentimentScore(BaseModel):
    sentiment_mean: float
    sentiment_weighted: float
    positive_ratio: float
    negative_ratio: float
    news_volume: int
    sentiment_momentum: float
    event_direction: float = 0.0
    event_magnitude: float = 0.0
    macro_impact: float = 0.0


class SentimentResponse(BaseModel):
    ticker: str
    sentiment: SentimentScore
    keywords: Dict[str, float] = {}


class ModelInfo(BaseModel):
    model_type: str
    version: Optional[str] = None
    n_features: int = 0
    trained_tickers: List[str] = []
    last_updated: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[ModelInfo] = None
    uptime_seconds: float = 0.0
