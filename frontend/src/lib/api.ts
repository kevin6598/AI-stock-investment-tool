const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface QuantileForecast {
  q05: number;
  q10: number;
  q25: number;
  q50: number;
  q75: number;
  q90: number;
  q95: number;
}

export interface RetailPrediction {
  ticker: string;
  horizon: string;
  direction: "UP" | "DOWN" | "HOLD";
  p_up: number;
  confidence: number;
  point_estimate: number;
  quantiles: QuantileForecast;
  risk_score: number;
  hold_signal: number;
  regime: string;
  is_zero_shot: boolean;
  meta_trade_probability?: number;
  uncertainty?: number;
  scaled_alpha?: number;
  model_weights?: Record<string, number>;
}

export interface PredictResponse {
  status: string;
  prediction: RetailPrediction;
  model_version: string | null;
  timestamp: string;
}

export interface IndicatorValue {
  date: string;
  value: number;
}

export interface IndicatorSeries {
  name: string;
  values: IndicatorValue[];
}

export interface IndicatorResponse {
  ticker: string;
  indicators: IndicatorSeries[];
}

export interface SentimentScore {
  sentiment_mean: number;
  sentiment_weighted: number;
  positive_ratio: number;
  negative_ratio: number;
  news_volume: number;
  sentiment_momentum: number;
  event_direction: number;
  event_magnitude: number;
  macro_impact: number;
}

export interface SentimentResponse {
  ticker: string;
  sentiment: SentimentScore;
  keywords: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_info: {
    model_type: string;
    version: string | null;
    n_features: number;
    trained_tickers: string[];
    last_updated: string | null;
  } | null;
  uptime_seconds: number;
}

export interface Top10Stock {
  rank: number;
  ticker: string;
  score: number;
  direction: "UP" | "DOWN" | "HOLD";
  p_up: number;
  expected_return: number;
  confidence: number;
  risk_score: number;
  sentiment_score: number;
  allocation_weight: number;
  reasons: string[];
}

export interface Top10Response {
  market: string;
  horizon: string;
  stocks: Top10Stock[];
  generated_at: string;
  model_version: string | null;
  total_candidates: number;
  pass_rate: number;
}

// --- Strategy Candidates Types ---

export interface StrategyCandidateStock {
  rank: number;
  ticker: string;
  score: number;
  direction: "UP" | "DOWN" | "HOLD";
  p_up: number;
  expected_return: number;
  confidence: number;
  risk_score: number;
  sentiment_score: number;
  allocation_weight: number;
  reasons: string[];
  mom_60d: number | null;
  high_52w_pct: number | null;
  mom_60d_decile: number | null;
  high_52w_pct_decile: number | null;
}

export interface StrategyCandidatesResponse {
  strategy_id: string;
  strategy_name: string;
  market: string;
  horizon: string;
  stocks: StrategyCandidateStock[];
  generated_at: string;
  model_version: string | null;
  universe_size: number;
  signal_matches: number;
  pass_rate: number;
}

// --- Strategy Governance Types ---

export interface StrategySignal {
  feature_1: string;
  feature_1_decile: number;
  feature_2: string;
  feature_2_decile: number;
  logic: string;
}

export interface StrategyBacktest {
  sharpe: number;
  beta_neutral_sharpe: number;
  total_return: number;
  precision_buy: number;
  win_rate: number;
  win_folds: string;
  monthly_consistency: number;
  cvar: number;
  turnover: number;
}

export interface StrategyGovernance {
  trust_score: number;
  trust_level: string;
  recommendation: string;
}

export interface StrategyStatusResponse {
  strategy_id: string;
  version: string;
  market: string;
  horizon_days: number;
  type: string;
  thesis: string;
  signal: StrategySignal;
  backtest: StrategyBacktest;
  governance: StrategyGovernance;
}

export interface WarningSignal {
  name: string;
  score: number;
  weight: number;
  weighted_contribution: number;
  detail: string;
  raw_value: number | null;
}

export interface EarlyWarningResponse {
  warning_score: number;
  level: string;
  exposure_multiplier: number;
  signals: WarningSignal[];
  timestamp: string;
  strategy_id: string;
}

export interface ExposureGuidanceResponse {
  strategy_id: string;
  warning_score: number;
  warning_level: string;
  exposure_multiplier: number;
  recommended_action: string;
  position_guidance: string;
  timestamp: string;
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  predict(ticker: string, horizon: string = "1M"): Promise<PredictResponse> {
    return fetchJSON("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify({ ticker, horizon }),
    });
  },

  getIndicators(
    ticker: string,
    period: string = "1y",
    indicators?: string
  ): Promise<IndicatorResponse> {
    const params = new URLSearchParams({ period });
    if (indicators) params.set("indicators", indicators);
    return fetchJSON(`/api/v1/indicators/${ticker}?${params}`);
  },

  getSentiment(ticker: string): Promise<SentimentResponse> {
    return fetchJSON(`/api/v1/sentiment/${ticker}`);
  },

  getHealth(): Promise<HealthResponse> {
    return fetchJSON("/api/v1/health");
  },

  getDiagnostics(): Promise<any> {
    return fetchJSON("/api/v1/model/diagnostics");
  },

  getTop10(
    market: string = "US",
    horizon: string = "1M",
    allocation: string = "risk_parity"
  ): Promise<Top10Response> {
    const params = new URLSearchParams({ horizon, allocation });
    return fetchJSON(`/api/v1/portfolio/top10/${market}?${params}`);
  },

  // Strategy Candidates endpoint
  getStrategyCandidates(
    horizon: string = "3M",
    allocation: string = "risk_parity",
    market: string = "US"
  ): Promise<StrategyCandidatesResponse> {
    const params = new URLSearchParams({ horizon, allocation, market });
    return fetchJSON(`/api/v1/strategy/candidates?${params}`);
  },

  // Strategy Governance endpoints
  getStrategyStatus(): Promise<StrategyStatusResponse> {
    return fetchJSON("/api/v1/strategy/status");
  },

  getEarlyWarning(): Promise<EarlyWarningResponse> {
    return fetchJSON("/api/v1/strategy/early-warning");
  },

  getExposureGuidance(): Promise<ExposureGuidanceResponse> {
    return fetchJSON("/api/v1/strategy/exposure-guidance");
  },
};
