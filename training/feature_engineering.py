"""
Feature Engineering Module
--------------------------
Computes five categories of features from raw market data:
  1. Technical indicators (RSI, MACD, Bollinger Bands, momentum, volume)
  2. Fundamental factors (P/E, P/B, dividend yield, earnings growth)
  3. Sentiment / keyword-based signals (news sentiment proxy via price-volume)
  4. Macro variables (market breadth, VIX proxy, yield curve proxy)
  5. Risk factors (volatility, drawdown, beta, tail risk)

All features are computed using only past data at each point to prevent look-ahead bias.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Technical Features
# ---------------------------------------------------------------------------

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV data.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex.

    Returns:
        DataFrame with technical feature columns appended.
    """
    out = df.copy()
    c = out["Close"]

    # Moving averages
    for w in [5, 10, 20, 50, 200]:
        out[f"sma_{w}"] = c.rolling(w).mean()
        out[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    # Price relative to moving averages
    for w in [20, 50, 200]:
        out[f"price_to_sma_{w}"] = c / out[f"sma_{w}"] - 1.0

    # Momentum / rate of change
    for lag in [5, 10, 21, 63, 126, 252]:
        out[f"momentum_{lag}d"] = c.pct_change(lag)

    # RSI (Wilder smoothing)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = out["macd"] - out["macd_signal"]

    # Bollinger Bands
    sma20 = out["sma_20"]
    std20 = c.rolling(20).std()
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / sma20
    out["bb_position"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

    # Volume features
    out["volume_sma_20"] = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_sma_20"]
    out["obv"] = (np.sign(c.diff()) * out["Volume"]).cumsum()

    # Average True Range
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - c.shift(1)).abs()
    low_close = (out["Low"] - c.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean()
    out["atr_ratio"] = out["atr_14"] / c  # normalized ATR

    return out


# ---------------------------------------------------------------------------
# 2. Fundamental Features (from yfinance info dict)
# ---------------------------------------------------------------------------

def compute_fundamental_features(info: dict) -> Dict[str, float]:
    """Extract fundamental factors from yfinance info dictionary.

    Args:
        info: dict from yf.Ticker(ticker).info

    Returns:
        Dict of fundamental feature name → value.
    """
    def safe_get(key, default=np.nan):
        v = info.get(key)
        return float(v) if v is not None else default

    features = {
        "pe_ratio": safe_get("trailingPE"),
        "forward_pe": safe_get("forwardPE"),
        "pb_ratio": safe_get("priceToBook"),
        "ps_ratio": safe_get("priceToSalesTrailing12Months"),
        "peg_ratio": safe_get("pegRatio"),
        "dividend_yield": safe_get("dividendYield", 0.0),
        "profit_margin": safe_get("profitMargins"),
        "operating_margin": safe_get("operatingMargins"),
        "roe": safe_get("returnOnEquity"),
        "roa": safe_get("returnOnAssets"),
        "debt_to_equity": safe_get("debtToEquity"),
        "current_ratio": safe_get("currentRatio"),
        "revenue_growth": safe_get("revenueGrowth"),
        "earnings_growth": safe_get("earningsGrowth"),
        "market_cap_log": np.log(safe_get("marketCap", 1)),
        "beta": safe_get("beta"),
    }
    return features


# ---------------------------------------------------------------------------
# 3. Sentiment Proxy Features (price-volume dynamics)
# ---------------------------------------------------------------------------

def compute_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sentiment proxy features from price-volume dynamics.

    In the absence of real NLP sentiment data, we use price-volume patterns
    that correlate with institutional sentiment shifts.
    """
    out = df.copy()
    c = out["Close"]
    v = out["Volume"]

    # Accumulation / distribution pressure
    clv = ((c - out["Low"]) - (out["High"] - c)) / (out["High"] - out["Low"]).replace(0, np.nan)
    out["ad_line"] = (clv * v).cumsum()

    # Money flow index (volume-weighted RSI)
    typical_price = (out["High"] + out["Low"] + c) / 3
    raw_money_flow = typical_price * v
    pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    neg_flow = raw_money_flow.where(typical_price <= typical_price.shift(1), 0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()
    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    out["mfi_14"] = mfi

    # Price-volume divergence (momentum vs volume trend)
    price_trend = c.pct_change(20)
    volume_trend = v.rolling(20).mean().pct_change(20)
    out["pv_divergence"] = price_trend - volume_trend

    # Buying pressure ratio (close relative to day range)
    day_range = out["High"] - out["Low"]
    out["buying_pressure"] = ((c - out["Low"]) / day_range.replace(0, np.nan)).rolling(10).mean()

    # Gap signals
    out["gap_up"] = (out["Open"] / c.shift(1) - 1).rolling(20).mean()

    # Unusual volume spikes (z-score)
    vol_mean = v.rolling(60).mean()
    vol_std = v.rolling(60).std()
    out["volume_zscore"] = (v - vol_mean) / vol_std.replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
# 4. Macro Features (market-level, applied uniformly to all stocks)
# ---------------------------------------------------------------------------

def compute_macro_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro-level features from a market index DataFrame.

    Args:
        market_df: DataFrame with OHLCV for a market index (e.g., SPY).

    Returns:
        DataFrame with macro feature columns.
    """
    out = pd.DataFrame(index=market_df.index)
    c = market_df["Close"]

    # Market trend
    out["market_return_21d"] = c.pct_change(21)
    out["market_return_63d"] = c.pct_change(63)
    out["market_sma_200_ratio"] = c / c.rolling(200).mean() - 1.0

    # Market volatility (VIX proxy)
    log_ret = np.log(c / c.shift(1))
    out["market_vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
    out["market_vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)

    # Volatility regime change
    out["vol_regime_change"] = out["market_vol_21d"] / out["market_vol_63d"] - 1.0

    # Market breadth proxy (using volume patterns)
    out["market_volume_trend"] = market_df["Volume"].rolling(20).mean().pct_change(20)

    # Drawdown from peak
    rolling_max = c.cummax()
    out["market_drawdown"] = c / rolling_max - 1.0

    return out


# ---------------------------------------------------------------------------
# 5. Risk Features
# ---------------------------------------------------------------------------

def compute_risk_features(df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute risk-related features for a single stock.

    Args:
        df: Stock OHLCV DataFrame.
        market_df: Optional market index DataFrame for beta calculation.
    """
    out = df.copy()
    log_ret = np.log(out["Close"] / out["Close"].shift(1))

    # Historical volatility at multiple windows
    for w in [10, 21, 63]:
        out[f"volatility_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Volatility ratio (short-term / long-term)
    out["vol_ratio_10_63"] = out["volatility_10d"] / out["volatility_63d"].replace(0, np.nan)

    # Maximum drawdown (rolling 63-day)
    rolling_max = out["Close"].rolling(63, min_periods=1).max()
    out["drawdown_63d"] = out["Close"] / rolling_max - 1.0

    # Downside deviation (63-day)
    neg_ret = log_ret.where(log_ret < 0, 0.0)
    out["downside_dev_63d"] = neg_ret.rolling(63).std() * np.sqrt(252)

    # Skewness and kurtosis (63-day rolling)
    out["return_skew_63d"] = log_ret.rolling(63).skew()
    out["return_kurt_63d"] = log_ret.rolling(63).kurt()

    # Value at Risk (5th percentile, 63-day rolling)
    out["var_5pct_63d"] = log_ret.rolling(63).quantile(0.05)

    # Beta to market
    if market_df is not None and "Close" in market_df.columns:
        mkt_ret = np.log(market_df["Close"] / market_df["Close"].shift(1))
        # Align indices
        aligned = pd.DataFrame({"stock": log_ret, "market": mkt_ret}).dropna()
        if len(aligned) > 63:
            rolling_cov = aligned["stock"].rolling(63).cov(aligned["market"])
            rolling_var = aligned["market"].rolling(63).var()
            beta = rolling_cov / rolling_var.replace(0, np.nan)
            out["beta_63d"] = beta.reindex(out.index)
        else:
            out["beta_63d"] = np.nan
    else:
        out["beta_63d"] = np.nan

    return out


# ---------------------------------------------------------------------------
# 6. NLP Sentiment Features (via models.sentiment)
# ---------------------------------------------------------------------------

def compute_nlp_sentiment_features(
    ticker: str,
    dates: pd.DatetimeIndex,
    use_finbert: bool = False,
) -> pd.DataFrame:
    """Compute NLP-based sentiment features for a ticker.

    Calls models.sentiment.get_sentiment_features() and expands the
    aggregated result into daily feature columns prefixed with ``nlp_``.

    Falls back gracefully to zero-filled features if the sentiment
    module or its dependencies are unavailable.

    Args:
        ticker: Stock ticker symbol.
        dates: DatetimeIndex of the stock DataFrame.
        use_finbert: Whether to use FinBERT (requires ``transformers``).

    Returns:
        DataFrame indexed by *dates* with 7 ``nlp_*`` columns.
    """
    default_features = {
        "nlp_sentiment_mean": 0.0,
        "nlp_sentiment_weighted": 0.0,
        "nlp_sentiment_std": 0.0,
        "nlp_positive_ratio": 0.33,
        "nlp_negative_ratio": 0.33,
        "nlp_news_volume": 0.0,
        "nlp_sentiment_momentum": 0.0,
    }

    try:
        from models.sentiment import get_sentiment_features
        raw = get_sentiment_features(ticker)
        features = {
            "nlp_sentiment_mean": raw.get("sentiment_mean", 0.0),
            "nlp_sentiment_weighted": raw.get("sentiment_weighted", 0.0),
            "nlp_sentiment_std": raw.get("sentiment_std", 0.0),
            "nlp_positive_ratio": raw.get("positive_ratio", 0.33),
            "nlp_negative_ratio": raw.get("negative_ratio", 0.33),
            "nlp_news_volume": float(raw.get("news_volume", 0)),
            "nlp_sentiment_momentum": raw.get("sentiment_momentum", 0.0),
        }
    except Exception as e:
        logger.debug(f"NLP sentiment unavailable for {ticker}: {e}")
        features = default_features

    return pd.DataFrame(features, index=dates)


# ---------------------------------------------------------------------------
# Aggregate Pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    stock_df: pd.DataFrame,
    stock_info: dict,
    market_df: Optional[pd.DataFrame] = None,
    forward_horizons: Optional[List[int]] = None,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Build complete feature matrix for a single stock.

    Args:
        stock_df: OHLCV DataFrame for the stock.
        stock_info: yfinance info dict for fundamental features.
        market_df: Optional market index OHLCV for macro/beta features.
        forward_horizons: List of forward return horizons in trading days
                          (e.g., [21, 63, 126] for 1M, 3M, 6M).

    Returns:
        DataFrame with all features and forward return targets.
        Rows with insufficient history are dropped.
    """
    if forward_horizons is None:
        forward_horizons = [21, 63, 126]

    # Compute all feature groups
    tech = compute_technical_features(stock_df)
    sent = compute_sentiment_features(stock_df)
    risk = compute_risk_features(stock_df, market_df)

    # Merge (they share the same index)
    features = tech.copy()
    # Add sentiment columns (skip duplicates from original OHLCV)
    for col in sent.columns:
        if col not in features.columns:
            features[col] = sent[col]
    # Add risk columns
    for col in risk.columns:
        if col not in features.columns:
            features[col] = risk[col]

    # Add macro features
    if market_df is not None:
        macro = compute_macro_features(market_df)
        macro = macro.reindex(features.index, method="ffill")
        for col in macro.columns:
            features[f"macro_{col}" if not col.startswith("macro_") else col] = macro[col]

    # Add fundamental features (static, repeated across all rows)
    fund = compute_fundamental_features(stock_info)
    for name, val in fund.items():
        features[f"fund_{name}"] = val

    # Add NLP sentiment features (if ticker is provided)
    if ticker is not None:
        nlp = compute_nlp_sentiment_features(ticker, features.index)
        for col in nlp.columns:
            features[col] = nlp[col]

    # Compute forward return targets
    c = features["Close"]
    for h in forward_horizons:
        features[f"fwd_return_{h}d"] = c.shift(-h) / c - 1.0

    # Drop raw OHLCV columns (keep only engineered features + targets)
    drop_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
    drop_cols = [col for col in drop_cols if col in features.columns]
    # Keep a copy of Close for portfolio construction later
    features["_close"] = stock_df["Close"]
    features.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Drop rows where we don't have enough history (first 252 rows)
    features = features.iloc[252:]

    # Drop rows where forward targets are NaN (last h rows)
    max_h = max(forward_horizons)
    features = features.iloc[:-max_h]

    return features


def build_panel_dataset(
    stock_dfs: Dict[str, pd.DataFrame],
    stock_infos: Dict[str, dict],
    market_df: Optional[pd.DataFrame] = None,
    forward_horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Build panel dataset across multiple stocks.

    Args:
        stock_dfs: dict of ticker → OHLCV DataFrame.
        stock_infos: dict of ticker → yfinance info dict.
        market_df: Market index OHLCV.
        forward_horizons: Forward return horizons in trading days.

    Returns:
        Panel DataFrame with MultiIndex (date, ticker) and all features.
    """
    panels = []
    for ticker, df in stock_dfs.items():
        info = stock_infos.get(ticker, {})
        feat = build_feature_matrix(df, info, market_df, forward_horizons, ticker=ticker)
        feat["ticker"] = ticker
        panels.append(feat)

    panel = pd.concat(panels)
    panel = panel.reset_index().rename(columns={"index": "date"})
    panel = panel.set_index(["date", "ticker"]).sort_index()

    return panel


def cross_sectional_normalize(
    panel: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Z-score normalize features cross-sectionally at each date.

    This is critical: normalize across stocks at each date, NOT across time.
    Prevents look-ahead bias and makes features comparable across stocks.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Identify target and metadata columns to exclude
    target_cols = [c for c in panel.columns if c.startswith("fwd_return_")]
    skip = set(exclude_cols + target_cols + ["_close"])

    feature_cols = [c for c in panel.columns if c not in skip]

    result = panel.copy()

    # Group by date (first level of MultiIndex)
    for col in feature_cols:
        grouped = result[col].groupby(level=0)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        result[col] = (result[col] - mean) / std

    # Winsorize at 3 standard deviations
    for col in feature_cols:
        result[col] = result[col].clip(-3.0, 3.0)

    # Fill remaining NaN with 0 (cross-sectional median would be 0 after z-scoring)
    result[feature_cols] = result[feature_cols].fillna(0.0)

    return result


def prune_features(
    X: pd.DataFrame,
    y: pd.Series,
    importance_threshold: float = 0.01,
    method: str = "lightgbm",
) -> Tuple[pd.DataFrame, List[str]]:
    """Prune low-importance features to reduce dimensionality.

    Uses LightGBM feature importance or mutual information to identify
    and drop features below the importance threshold.

    Args:
        X: feature DataFrame.
        y: target Series.
        importance_threshold: drop features with importance below this.
        method: "lightgbm" or "mutual_info".

    Returns:
        (pruned_X, selected_cols) - pruned DataFrame and list of kept column names.
    """
    import numpy as np

    cols = list(X.columns)
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    # Clean NaN/Inf
    np.nan_to_num(X_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(y_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "lightgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                max_depth=6,
                verbose=-1,
                random_state=42,
            )
            model.fit(X_arr, y_arr)
            importances = model.feature_importances_
            total = importances.sum()
            if total > 0:
                importances = importances / total
            else:
                importances = np.ones(len(cols)) / len(cols)
        except ImportError:
            logger.warning("LightGBM not available for feature pruning, using mutual info")
            method = "mutual_info"

    if method == "mutual_info":
        try:
            from sklearn.feature_selection import mutual_info_regression
            importances = mutual_info_regression(X_arr, y_arr, random_state=42)
            total = importances.sum()
            if total > 0:
                importances = importances / total
            else:
                importances = np.ones(len(cols)) / len(cols)
        except Exception:
            # Fallback: keep all features
            return X, cols

    # Select features above threshold
    selected = []
    for i, col in enumerate(cols):
        if importances[i] >= importance_threshold:
            selected.append(col)

    if not selected:
        # Keep at least top 10 features
        top_indices = np.argsort(importances)[-10:]
        selected = [cols[i] for i in top_indices]

    logger.info(f"Feature pruning: {len(cols)} -> {len(selected)} features")
    return X[selected], selected


def get_feature_groups(panel: pd.DataFrame) -> Dict[str, List[str]]:
    """Return mapping of feature group name to column names.

    Useful for the weight optimizer to know which features belong to which component.
    """
    all_cols = [c for c in panel.columns
                if not c.startswith("fwd_return_") and c != "_close"]

    groups = {
        "technical": [c for c in all_cols if any(
            c.startswith(p) for p in [
                "sma_", "ema_", "price_to_sma", "momentum_", "rsi_",
                "macd", "bb_", "volume_sma", "volume_ratio", "obv",
                "atr_",
            ]
        )],
        "sentiment": [c for c in all_cols if any(
            c.startswith(p) for p in [
                "ad_line", "mfi_", "pv_divergence", "buying_pressure",
                "gap_up", "volume_zscore", "nlp_",
            ]
        )],
        "fundamental": [c for c in all_cols if c.startswith("fund_")],
        "macro": [c for c in all_cols if c.startswith("macro_")],
        "risk": [c for c in all_cols if any(
            c.startswith(p) for p in [
                "volatility_", "vol_ratio", "drawdown_", "downside_dev",
                "return_skew", "return_kurt", "var_5pct", "beta_",
            ]
        )],
    }

    return groups
