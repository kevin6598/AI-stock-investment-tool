from typing import List, Optional

import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """Add Simple Moving Average columns."""
    if periods is None:
        periods = [20, 50]
    for p in periods:
        df[f"SMA_{p}"] = df["Close"].rolling(window=p).mean()
    return df


def add_ema(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """Add Exponential Moving Average columns."""
    if periods is None:
        periods = [12, 26]
    for p in periods:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column using Wilder smoothing."""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD, Signal line, and Histogram columns."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_stochastic_oscillator(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Add Stochastic Oscillator (%K and %D) columns."""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["Close"] - low_min) / denom
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index (ADX) column."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Commodity Channel Index (CCI) column."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    return df


def add_vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add VWAP deviation column (price distance from rolling VWAP)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_tpv = (tp * df["Volume"]).rolling(period).sum()
    cumulative_vol = df["Volume"].rolling(period).sum().replace(0, np.nan)
    vwap = cumulative_tpv / cumulative_vol
    df["vwap"] = vwap
    df["vwap_deviation"] = (df["Close"] - vwap) / vwap.replace(0, np.nan)
    return df


def add_volume_spike_encoding(df: pd.DataFrame, window: int = 60, threshold: float = 2.0) -> pd.DataFrame:
    """Add volume spike binary encoding (z-score > threshold)."""
    vol_mean = df["Volume"].rolling(window).mean()
    vol_std = df["Volume"].rolling(window).std().replace(0, np.nan)
    z = (df["Volume"] - vol_mean) / vol_std
    df["volume_spike"] = (z > threshold).astype(float)
    df["volume_spike_intensity"] = z.clip(lower=0)
    return df


def add_volatility_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility clustering features (GARCH-like proxy)."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    sq_ret = log_ret ** 2

    # Exponentially weighted squared returns (proxy for GARCH conditional variance)
    df["vol_cluster_ew"] = sq_ret.ewm(span=21, adjust=False).mean().apply(np.sqrt) * np.sqrt(252)
    # Ratio of recent to longer-term vol clustering
    short_vol = sq_ret.ewm(span=5, adjust=False).mean()
    long_vol = sq_ret.ewm(span=63, adjust=False).mean().replace(0, np.nan)
    df["vol_cluster_ratio"] = short_vol / long_vol
    # Persistence: autocorrelation of squared returns
    df["vol_persistence"] = sq_ret.rolling(63).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0.0, raw=True
    )
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime proxy features."""
    c = df["Close"]
    log_ret = np.log(c / c.shift(1))

    # Trend strength: price vs 200-day SMA
    sma_200 = c.rolling(200).mean()
    df["regime_trend"] = (c / sma_200.replace(0, np.nan) - 1.0)

    # Volatility regime: 21d vol / 63d vol
    vol_21 = log_ret.rolling(21).std() * np.sqrt(252)
    vol_63 = log_ret.rolling(63).std() * np.sqrt(252)
    df["regime_vol_ratio"] = vol_21 / vol_63.replace(0, np.nan)

    # Drawdown regime
    rolling_max = c.cummax()
    dd = c / rolling_max - 1.0
    df["regime_drawdown"] = dd

    # Mean reversion vs momentum (Hurst exponent proxy via R/S method)
    def hurst_proxy(series):
        if len(series) < 20:
            return 0.5
        n = len(series)
        mean_val = series.mean()
        deviations = np.cumsum(series - mean_val)
        r = deviations.max() - deviations.min()
        s = series.std()
        if s == 0:
            return 0.5
        return np.log(r / s) / np.log(n) if r > 0 else 0.5

    df["regime_hurst"] = log_ret.rolling(126).apply(hurst_proxy, raw=True)
    return df
