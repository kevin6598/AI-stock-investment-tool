from typing import Optional

import yfinance as yf
import pandas as pd


# Simple KRW/USD exchange rate constant (fallback when yfinance FX unavailable)
_KRW_USD_RATE = 1350.0  # approximate


def detect_exchange(ticker: str) -> str:
    """Infer exchange from ticker suffix.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Exchange name: "KOSPI", "KOSDAQ", "NYSE", or "NASDAQ".
    """
    upper = ticker.upper()
    if upper.endswith(".KS"):
        return "KOSPI"
    if upper.endswith(".KQ"):
        return "KOSDAQ"
    # For US stocks, try yfinance info; fallback to NASDAQ
    try:
        stock = yf.Ticker(ticker)
        exchange = stock.info.get("exchange", "")
        if "NYS" in exchange.upper() or "NYSE" in exchange.upper():
            return "NYSE"
    except Exception:
        pass
    return "NASDAQ"


def normalize_currency(
    price: float,
    currency: str,
    target: str = "USD",
) -> float:
    """Convert price to target currency using simple FX rates.

    Args:
        price: Price in source currency.
        currency: Source currency code (e.g., "KRW", "USD").
        target: Target currency code.

    Returns:
        Price in target currency.
    """
    if currency == target:
        return price
    if currency == "KRW" and target == "USD":
        return price / _KRW_USD_RATE
    if currency == "USD" and target == "KRW":
        return price * _KRW_USD_RATE
    # Unknown pair: return as-is
    return price


def get_stock_info(ticker: str) -> Optional[dict]:
    """Fetch company info: name, sector, price, market cap, P/E, 52-week range, exchange."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get("trailingPegRatio") is None and info.get("shortName") is None:
            return None

        exchange = detect_exchange(ticker)
        currency = info.get("currency", "USD")

        return {
            "ticker": ticker.upper(),
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "week_52_low": info.get("fiftyTwoWeekLow", None),
            "week_52_high": info.get("fiftyTwoWeekHigh", None),
            "volume": info.get("volume", 0),
            "avg_volume": info.get("averageVolume", 0),
            "dividend_yield": info.get("dividendYield", None),
            "exchange": exchange,
            "currency": currency,
        }
    except Exception:
        return None


def get_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data as a DataFrame."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        # yfinance >= 0.2.36 returns MultiIndex columns ("Price", "Ticker")
        # for single-ticker queries.  Flatten to simple column names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def get_historical_data_range(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data for a specific date range.

    Supports 15+ year data windows via yfinance start/end params.

    Args:
        ticker: Stock ticker symbol.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: Data interval (default "1d").

    Returns:
        DataFrame with OHLCV data, or empty DataFrame on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def get_current_price(ticker: str) -> Optional[float]:
    """Get current price for portfolio valuation."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        return float(price) if price else None
    except Exception:
        return None
