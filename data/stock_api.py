from typing import Optional

import yfinance as yf
import pandas as pd


def get_stock_info(ticker: str) -> Optional[dict]:
    """Fetch company info: name, sector, price, market cap, P/E, 52-week range."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get("trailingPegRatio") is None and info.get("shortName") is None:
            return None
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
