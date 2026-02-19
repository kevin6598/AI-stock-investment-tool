"""
Strategy Signal Filter
-----------------------
Computes live strategy signals for the surviving V4 strategy:
  US_63d_mom_60d_decile_d4_AND_high_52w_pct_decile_d0

Downloads price data in batch via yf.download, computes cross-sectional
decile ranks, and filters tickers matching the strategy signal.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy_definition.us_63d_mom_60d_d4_52w_d0 import (
    IDENTITY,
    FeatureDefinitions,
)

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60


@dataclass
class StrategyCandidate:
    """A ticker that matches the strategy signal."""
    ticker: str
    mom_60d: float
    high_52w_pct: float
    mom_60d_decile: int
    high_52w_pct_decile: int


@dataclass
class StrategyFilterResult:
    """Result of running the strategy filter on a universe."""
    strategy_id: str
    candidates: List[StrategyCandidate]
    universe_size: int
    computed_at: float
    all_features: Dict[str, Dict[str, float]] = field(default_factory=dict)


class StrategyFilter:
    """Filter a stock universe using the surviving strategy's signal rules.

    Uses yf.download for batch price fetching, FeatureDefinitions for
    feature computation, and cross-sectional decile ranking.
    Caches results with a 60-second TTL.
    """

    def __init__(self):
        self._cache: Optional[StrategyFilterResult] = None
        self._cache_time: float = 0.0

    def filter(self, market: str = "US") -> StrategyFilterResult:
        """Compute strategy signal and return matching tickers.

        Args:
            market: "US", "KR", or "ALL".

        Returns:
            StrategyFilterResult with matching candidates.
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < CACHE_TTL_SECONDS:
            return self._cache

        from data.universe_manager import UniverseManager

        um = UniverseManager()
        market_upper = market.upper()
        tickers = um.get_universe_by_market(
            "all" if market_upper == "ALL" else market_upper,
        )
        universe_size = len(tickers)
        logger.info("StrategyFilter: %d tickers for market=%s", universe_size, market)

        if not tickers:
            result = StrategyFilterResult(
                strategy_id=IDENTITY.strategy_id,
                candidates=[],
                universe_size=0,
                computed_at=now,
            )
            self._cache = result
            self._cache_time = now
            return result

        # Batch download price data
        close_df = self._batch_download(tickers)

        if close_df.empty:
            result = StrategyFilterResult(
                strategy_id=IDENTITY.strategy_id,
                candidates=[],
                universe_size=universe_size,
                computed_at=now,
            )
            self._cache = result
            self._cache_time = now
            return result

        # Compute features for each ticker
        all_features: Dict[str, Dict[str, float]] = {}
        mom_60d_values = {}
        high_52w_pct_values = {}

        for ticker in close_df.columns:
            series = close_df[ticker].dropna()
            if len(series) < 61:
                continue
            mom = FeatureDefinitions.compute_mom_60d(series)
            h52 = FeatureDefinitions.compute_high_52w_pct(series)
            if np.isnan(mom) or np.isnan(h52):
                continue
            mom_60d_values[ticker] = mom
            high_52w_pct_values[ticker] = h52
            all_features[ticker] = {"mom_60d": mom, "high_52w_pct": h52}

        if not mom_60d_values:
            result = StrategyFilterResult(
                strategy_id=IDENTITY.strategy_id,
                candidates=[],
                universe_size=universe_size,
                computed_at=now,
            )
            self._cache = result
            self._cache_time = now
            return result

        # Compute cross-sectional deciles
        all_mom = list(mom_60d_values.values())
        all_h52 = list(high_52w_pct_values.values())

        candidates = []
        for ticker in mom_60d_values:
            mom_decile = FeatureDefinitions.compute_decile(
                mom_60d_values[ticker], all_mom,
            )
            h52_decile = FeatureDefinitions.compute_decile(
                high_52w_pct_values[ticker], all_h52,
            )
            all_features[ticker]["mom_60d_decile"] = mom_decile
            all_features[ticker]["high_52w_pct_decile"] = h52_decile

            # Strategy signal: mom_60d decile == 4 AND high_52w_pct decile == 0
            if (
                mom_decile == IDENTITY.feature_1_decile
                and h52_decile == IDENTITY.feature_2_decile
            ):
                candidates.append(StrategyCandidate(
                    ticker=ticker,
                    mom_60d=mom_60d_values[ticker],
                    high_52w_pct=high_52w_pct_values[ticker],
                    mom_60d_decile=mom_decile,
                    high_52w_pct_decile=h52_decile,
                ))

        logger.info(
            "StrategyFilter: %d/%d tickers match signal",
            len(candidates), len(mom_60d_values),
        )

        result = StrategyFilterResult(
            strategy_id=IDENTITY.strategy_id,
            candidates=candidates,
            universe_size=universe_size,
            computed_at=now,
            all_features=all_features,
        )
        self._cache = result
        self._cache_time = now
        return result

    def _batch_download(self, tickers: List[str]) -> pd.DataFrame:
        """Download 1 year of close prices for all tickers via yf.download.

        Falls back to per-ticker download on failure.

        Returns:
            DataFrame with columns=tickers, index=dates, values=close prices.
        """
        import yfinance as yf

        try:
            raw = yf.download(
                tickers,
                period="1y",
                interval="1d",
                progress=False,
                threads=True,
            )
            if raw.empty:
                return pd.DataFrame()

            # Handle MultiIndex columns from yf.download
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw["Close"]
            else:
                # Single ticker returns flat columns
                close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})

            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(name=tickers[0])

            return close_df

        except Exception as e:
            logger.warning("Batch download failed: %s, falling back to per-ticker", e)
            return self._per_ticker_fallback(tickers)

    def _per_ticker_fallback(self, tickers: List[str]) -> pd.DataFrame:
        """Fall back to per-ticker download using get_historical_data."""
        from data.stock_api import get_historical_data

        frames = {}
        for ticker in tickers:
            try:
                df = get_historical_data(ticker, period="1y")
                if not df.empty and "Close" in df.columns:
                    frames[ticker] = df["Close"]
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames)


# Module-level singleton
_strategy_filter = StrategyFilter()


def get_strategy_filter() -> StrategyFilter:
    """Return the module-level StrategyFilter singleton."""
    return _strategy_filter
