"""
Liquidity & Capacity Control
------------------------------
Liquidity filtering, ADV computation, market impact estimation,
and portfolio capacity analysis.

Reuses the sqrt market impact model from training/backtesting.py.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquidityConfig:
    """Configuration for liquidity filtering."""
    min_adv_usd: float = 1_000_000       # min average daily volume in USD
    max_position_pct_adv: float = 0.05    # max 5% of ADV per position
    min_adv_percentile: float = 0.10      # exclude bottom 10% liquidity
    market_impact_coeff: float = 0.1      # sqrt impact model coefficient


class LiquidityFilter:
    """Filter stocks by liquidity and estimate capacity constraints.

    Usage:
        lf = LiquidityFilter()
        passed, rejected = lf.filter_by_liquidity(tickers)
        max_size = lf.max_position_size("AAPL", adv=50_000_000)
    """

    def __init__(self, config: Optional[LiquidityConfig] = None):
        self.config = config or LiquidityConfig()
        self._adv_cache = {}  # type: Dict[str, float]

    def compute_adv(self, ticker: str, period: str = "3mo") -> float:
        """Compute average daily volume in USD for a ticker.

        Args:
            ticker: Stock ticker symbol.
            period: Lookback period for ADV calculation.

        Returns:
            Average daily dollar volume (price * volume).
        """
        if ticker in self._adv_cache:
            return self._adv_cache[ticker]

        try:
            from data.stock_api import get_historical_data
            df = get_historical_data(ticker, period=period)
            if df.empty or "Close" not in df.columns or "Volume" not in df.columns:
                self._adv_cache[ticker] = 0.0
                return 0.0
            dollar_vol = (df["Close"] * df["Volume"]).dropna()
            adv = float(dollar_vol.mean()) if len(dollar_vol) > 0 else 0.0
        except Exception as e:
            logger.warning("Failed to compute ADV for %s: %s", ticker, e)
            adv = 0.0

        self._adv_cache[ticker] = adv
        return adv

    def filter_by_liquidity(
        self,
        tickers: List[str],
        advs: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Filter tickers by minimum ADV and percentile thresholds.

        Args:
            tickers: List of ticker symbols.
            advs: Pre-computed ADV dict. If None, computes on the fly.

        Returns:
            (passed, rejected) tuple of ticker lists.
        """
        if advs is None:
            advs = {}
            for t in tickers:
                advs[t] = self.compute_adv(t)

        # Compute percentile cutoff
        adv_values = [advs.get(t, 0.0) for t in tickers]
        if adv_values:
            percentile_cutoff = float(np.percentile(
                adv_values, self.config.min_adv_percentile * 100
            ))
        else:
            percentile_cutoff = 0.0

        passed = []
        rejected = []
        for t in tickers:
            adv = advs.get(t, 0.0)
            if adv >= self.config.min_adv_usd and adv >= percentile_cutoff:
                passed.append(t)
            else:
                rejected.append(t)

        logger.info("Liquidity filter: %d/%d passed (min_adv=$%.0f)",
                     len(passed), len(tickers), self.config.min_adv_usd)
        return passed, rejected

    def max_position_size(self, ticker: str, adv: float) -> float:
        """Compute maximum position size in USD based on ADV constraint.

        Args:
            ticker: Stock ticker (for logging).
            adv: Average daily dollar volume.

        Returns:
            Maximum position size in USD.
        """
        return adv * self.config.max_position_pct_adv

    def estimate_market_impact(self, trade_size: float, adv: float) -> float:
        """Estimate market impact using the sqrt model.

        impact = coeff * sqrt(trade_size / adv)

        This is the same formula used in training/backtesting.py.

        Args:
            trade_size: Trade size in USD.
            adv: Average daily dollar volume.

        Returns:
            Estimated market impact as a fraction (e.g., 0.001 = 10 bps).
        """
        if adv <= 0 or trade_size <= 0:
            return 0.0
        return self.config.market_impact_coeff * np.sqrt(trade_size / adv)

    def estimate_capacity(
        self,
        weights: Dict[str, float],
        advs: Dict[str, float],
        target_sharpe_decay: float = 0.10,
    ) -> float:
        """Estimate strategy capacity (max AUM before Sharpe decays by target).

        For each asset, capacity is limited by the ADV constraint.
        Overall capacity = min across assets of (ADV * max_pct / weight).

        Args:
            weights: Portfolio weights {ticker: weight}.
            advs: ADV per ticker {ticker: adv_usd}.
            target_sharpe_decay: Acceptable Sharpe ratio decay fraction.

        Returns:
            Estimated maximum AUM in USD.
        """
        if not weights or not advs:
            return 0.0

        capacities = []
        for ticker, weight in weights.items():
            if weight <= 0:
                continue
            adv = advs.get(ticker, 0.0)
            if adv <= 0:
                continue
            # Max position = max_pct * adv
            # AUM = max_position / weight
            max_position = self.max_position_size(ticker, adv)
            asset_capacity = max_position / weight
            capacities.append(asset_capacity)

        if not capacities:
            return 0.0

        # Conservative: use the binding constraint (minimum)
        raw_capacity = min(capacities)

        # Adjust for target Sharpe decay (less capacity = more impact)
        # Simple scaling: capacity * sqrt(target_decay) to account for
        # the sqrt impact model
        adjusted = raw_capacity * np.sqrt(max(target_sharpe_decay, 0.01))

        return float(adjusted)
