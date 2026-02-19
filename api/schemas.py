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
    hold_signal: float = Field(default=0.0, ge=0, le=1, description="Deprecated, kept for backward compatibility")
    regime: str
    is_zero_shot: bool = Field(
        default=False,
        description="True if ticker was not in training set",
    )
    meta_trade_probability: Optional[float] = Field(default=None, ge=0, le=1)
    uncertainty: Optional[float] = Field(default=None, ge=0)
    scaled_alpha: Optional[float] = Field(default=None)
    model_weights: Optional[Dict[str, float]] = Field(default=None)


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


# --- Top 10 Models ---

class Top10StockSchema(BaseModel):
    rank: int
    ticker: str
    score: float
    direction: str
    p_up: float = Field(ge=0, le=1)
    expected_return: float
    confidence: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    sentiment_score: float
    allocation_weight: float = Field(ge=0, le=1)
    reasons: List[str]


class Top10Response(BaseModel):
    market: str
    horizon: str
    stocks: List[Top10StockSchema]
    generated_at: str
    model_version: Optional[str] = None
    total_candidates: int
    pass_rate: float


# --- Strategy Candidates Models ---

class StrategyCandidateSchema(BaseModel):
    rank: int
    ticker: str
    score: float
    direction: str
    p_up: float = Field(ge=0, le=1)
    expected_return: float
    confidence: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    sentiment_score: float
    allocation_weight: float = Field(ge=0, le=1)
    reasons: List[str]
    mom_60d: Optional[float] = None
    high_52w_pct: Optional[float] = None
    mom_60d_decile: Optional[int] = None
    high_52w_pct_decile: Optional[int] = None


class StrategyCandidatesResponse(BaseModel):
    strategy_id: str
    strategy_name: str
    market: str
    horizon: str
    stocks: List[StrategyCandidateSchema]
    generated_at: str
    model_version: Optional[str] = None
    universe_size: int
    signal_matches: int
    pass_rate: float


# --- Strategy Governance Models ---

class StrategySignalSchema(BaseModel):
    feature_1: str
    feature_1_decile: int
    feature_2: str
    feature_2_decile: int
    logic: str

class StrategyBacktestSchema(BaseModel):
    sharpe: float
    beta_neutral_sharpe: float
    total_return: float
    precision_buy: float
    win_rate: float
    win_folds: str
    monthly_consistency: float
    cvar: float
    turnover: float

class StrategyGovernanceSchema(BaseModel):
    trust_score: float
    trust_level: str
    recommendation: str

class StrategyStatusResponse(BaseModel):
    strategy_id: str
    version: str
    market: str
    horizon_days: int
    type: str
    thesis: str
    signal: StrategySignalSchema
    backtest: StrategyBacktestSchema
    governance: StrategyGovernanceSchema

class WarningSignalSchema(BaseModel):
    name: str
    score: float
    weight: float
    weighted_contribution: float
    detail: str
    raw_value: Optional[float] = None

class EarlyWarningResponse(BaseModel):
    warning_score: float
    level: str
    exposure_multiplier: float
    signals: List[WarningSignalSchema]
    timestamp: str
    strategy_id: str

class ExposureGuidanceResponse(BaseModel):
    strategy_id: str
    warning_score: float
    warning_level: str
    exposure_multiplier: float
    recommended_action: str
    position_guidance: str
    timestamp: str
