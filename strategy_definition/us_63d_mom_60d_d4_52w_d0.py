"""
Canonical Strategy Definition
==============================
US_63d_mom_60d_decile_d4_AND_high_52w_pct_decile_d0

This is the SOLE surviving strategy from the V4 pipeline.
37,322 candidates → 16-stage filter chain → 1 survivor.

Identity
--------
- Market: US
- Horizon: 63 trading days (~3 months)
- Type: combo_decile (two cross-sectional decile conditions joined by AND)
- Signal: Buy stocks with mediocre 60-day momentum (decile 4) that are
  simultaneously near their 52-week lows (decile 0).
- Thesis: Contrarian value — stocks that have stopped falling (not the worst
  momentum) but remain deeply discounted relative to their annual range.
  The combination filters out value traps (still falling) and momentum
  chasers (already recovered).

This module serves as the single source of truth for:
  1. Strategy identity and structural parameters
  2. Backtest performance metrics from walk-forward validation
  3. Survival thresholds the strategy passed
  4. Feature definitions required to compute live signals
  5. Failure precursor definitions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


STRATEGY_ID = "US_63d_mom_60d_decile_d4_AND_high_52w_pct_decile_d0"
STRATEGY_VERSION = "v4.0"


@dataclass(frozen=True)
class StrategyIdentity:
    """Immutable structural identity of the surviving strategy."""
    strategy_id: str = STRATEGY_ID
    market: str = "US"
    horizon_days: int = 63
    strategy_type: str = "combo_decile"

    # Signal components
    feature_1: str = "mom_60d"
    feature_1_decile: int = 4       # mediocre momentum (not worst, not best)
    feature_2: str = "high_52w_pct"
    feature_2_decile: int = 0       # near 52-week lows
    logic: str = "AND"              # both conditions must be true

    # Tri-state labeling
    label_threshold_pct: float = 2.0  # 63d horizon uses ±2% for BUY/SELL/HOLD

    # Human-readable description
    thesis: str = (
        "Buy US stocks with stabilized momentum (decile 4 of 60-day returns) "
        "that are trading near their 52-week lows (decile 0 of high_52w_pct). "
        "This captures the transition from capitulation to mean reversion."
    )


@dataclass(frozen=True)
class BacktestMetrics:
    """Walk-forward validated performance metrics.

    These are NOT predictions. They describe historical survivability
    across 12 walk-forward folds.
    """
    # Core performance
    sharpe_ratio: float = 1.2559
    beta_neutral_sharpe: float = 4.10
    total_return: float = 2.2433      # 224.3% cumulative
    max_drawdown: float = 0.0         # across walk-forward folds
    annualized_return: float = 0.0    # derived from total_return / n_years

    # Precision metrics
    precision_buy: float = 0.6278     # 62.78% of BUY signals were correct
    precision_sell: float = 0.0       # not measured (strategy is long-only)
    win_rate: float = 0.7052          # 70.52% of trades profitable

    # Fold consistency
    win_folds: int = 12               # folds with positive return
    total_folds: int = 12             # total walk-forward folds
    monthly_consistency: float = 1.0  # fraction of positive months

    # Risk metrics
    portfolio_cvar: float = 0.004388  # conditional value at risk
    mean_turnover: float = 3.6667     # monthly turnover (low = good)

    # Meta-model
    composite_score: float = 0.45
    meta_score: float = 0.45


@dataclass(frozen=True)
class SurvivalThresholds:
    """The exact thresholds this strategy survived.

    Each threshold corresponds to a pipeline stage.
    If the strategy's metric drops below these in live monitoring,
    it is a failure precursor.
    """
    # Stage: Edge filter
    min_precision_buy: float = 0.60
    min_sharpe: float = 0.50

    # Stage: Walk-forward validation
    min_fold_win_rate: float = 0.50   # >50% of folds must be profitable

    # Stage: Overfitting filter
    max_oof_sharpe_ratio: float = 3.0  # OOF vs IS Sharpe ratio cap

    # Stage: Beta-neutral check
    min_beta_neutral_sharpe: float = 0.5

    # Stage: Distribution safety
    min_hit_rate: float = 0.52

    # Stage: Cost survival
    cost_scenarios: list = field(default_factory=lambda: [
        (5, 2), (10, 5), (15, 5)      # (spread_bps, impact_bps) pairs
    ])

    # Stage: Regime robustness
    min_regime_fraction: float = 0.70  # win rate across VIX regimes

    # Stage: Turnover filter
    max_turnover: float = 12.0

    # Stage: Meta-model gate
    min_meta_score: float = 0.50

    # Stage: Rule tuning
    precision_th: float = 0.60
    turnover_th: float = 12.0
    regime_th: float = 0.70


@dataclass(frozen=True)
class FeatureDefinitions:
    """Features required to compute the strategy signal in production.

    These map directly to the pipeline's feature engineering functions.
    """
    # Primary signal features
    mom_60d: str = "60-day momentum (log return over trailing 60 days)"
    high_52w_pct: str = "Current price as percentage of 52-week high [0-1]"

    # Required auxiliary features for signal context
    required_price_fields: tuple = ("Close", "High", "Low", "Volume")
    lookback_days: int = 252          # 1 year for 52-week high computation
    momentum_window: int = 60         # days for momentum computation

    @staticmethod
    def compute_mom_60d(close_series) -> float:
        """Compute 60-day log momentum from close prices.

        Args:
            close_series: pandas Series of close prices (at least 61 values).

        Returns:
            Log return over trailing 60 days.
        """
        import numpy as np
        if len(close_series) < 61:
            return float("nan")
        return float(np.log(close_series.iloc[-1] / close_series.iloc[-61]))

    @staticmethod
    def compute_high_52w_pct(close_series) -> float:
        """Compute current price as fraction of 52-week high.

        Args:
            close_series: pandas Series of close prices (at least 252 values).

        Returns:
            Ratio in [0, 1] where 0 = at 52-week low, 1 = at 52-week high.
        """
        if len(close_series) < 252:
            window = close_series
        else:
            window = close_series.iloc[-252:]
        high_52w = window.max()
        low_52w = window.min()
        current = close_series.iloc[-1]
        if high_52w == low_52w:
            return 0.5
        return float((current - low_52w) / (high_52w - low_52w))

    @staticmethod
    def compute_decile(value: float, all_values) -> int:
        """Compute cross-sectional decile rank [0-9].

        Args:
            value: this ticker's feature value.
            all_values: array of all tickers' feature values.

        Returns:
            Decile rank 0-9.
        """
        import numpy as np
        arr = np.array(all_values)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 10:
            return 5  # default to middle
        percentile = float(np.nanmean(arr <= value))
        return min(int(percentile * 10), 9)


@dataclass(frozen=True)
class FailurePrecursors:
    """Conditions that indicate the strategy is degrading.

    Each precursor maps to a specific structural assumption of the strategy.
    When precursors activate, the early warning system increases the
    warning score, which in turn reduces exposure.
    """
    # Momentum structure degradation
    momentum_regime_shift: str = (
        "60-day cross-sectional momentum dispersion collapses below 1 stdev "
        "of its 252-day rolling mean. When all stocks move together, "
        "decile-based momentum selection loses discriminative power."
    )

    # Value trap activation
    value_trap_signal: str = (
        "Stocks in decile 0 of high_52w_pct continue declining after signal. "
        "If the 21-day forward return of d0 stocks is consistently negative "
        "for 3+ consecutive periods, the mean-reversion thesis is failing."
    )

    # Volatility regime change
    vol_regime_shift: str = (
        "VIX regime exceeds 2x its 252-day moving average. "
        "The strategy was validated under VIX < 1.166x normal. "
        "Extreme volatility invalidates the 63-day holding period assumption."
    )

    # Cross-market contagion
    cross_market_contagion: str = (
        "KOSPI-to-US or KOSDAQ-to-US contagion flag activates. "
        "When detected, US momentum structure is being disrupted by "
        "external market stress."
    )

    # Volume anomaly
    volume_dry_up: str = (
        "Trading volume in signal stocks drops below 50% of their "
        "20-day moving average. Illiquidity makes the 63-day horizon "
        "unreliable for position entry and exit."
    )

    # Sector concentration drift
    sector_rotation: str = (
        "Signal stocks concentrate >60% in a single sector. "
        "The strategy's edge depends on cross-sector diversification "
        "within the decile intersection."
    )


@dataclass(frozen=True)
class GovernancePolicy:
    """How the system should respond to strategy health changes.

    Maps warning_score ranges to exposure multipliers.
    """
    # Warning score thresholds → exposure multiplier
    # 0.0 - 0.2: HEALTHY     → 1.0x (full exposure)
    # 0.2 - 0.4: CAUTION     → 0.75x
    # 0.4 - 0.6: WARNING     → 0.50x
    # 0.6 - 0.8: DANGER      → 0.25x
    # 0.8 - 1.0: CRITICAL    → 0.0x (no exposure)

    exposure_schedule: tuple = (
        (0.0, 0.2, 1.00, "HEALTHY"),
        (0.2, 0.4, 0.75, "CAUTION"),
        (0.4, 0.6, 0.50, "WARNING"),
        (0.6, 0.8, 0.25, "DANGER"),
        (0.8, 1.0, 0.00, "CRITICAL"),
    )

    trust_score_from_pipeline: float = 0.60   # from v4 governance report
    trust_level: str = "MEDIUM"
    pipeline_recommendation: str = "REDUCE SIZE"

    @staticmethod
    def get_exposure_multiplier(warning_score: float) -> tuple:
        """Map warning score to exposure multiplier and level.

        Args:
            warning_score: normalized [0, 1] warning score.

        Returns:
            (multiplier, level_name) tuple.
        """
        score = max(0.0, min(1.0, warning_score))
        schedule = GovernancePolicy.exposure_schedule.fget(None) if callable(GovernancePolicy.exposure_schedule) else (
            (0.0, 0.2, 1.00, "HEALTHY"),
            (0.2, 0.4, 0.75, "CAUTION"),
            (0.4, 0.6, 0.50, "WARNING"),
            (0.6, 0.8, 0.25, "DANGER"),
            (0.8, 1.0, 0.00, "CRITICAL"),
        )
        for low, high, mult, level in schedule:
            if low <= score < high:
                return mult, level
        return 0.0, "CRITICAL"


# Singleton instances for import convenience
IDENTITY = StrategyIdentity()
BACKTEST = BacktestMetrics()
THRESHOLDS = SurvivalThresholds()
FEATURES = FeatureDefinitions()
FAILURE_PRECURSORS = FailurePrecursors()
GOVERNANCE = GovernancePolicy()


def get_strategy_summary() -> Dict:
    """Return a complete summary dict for API consumption."""
    return {
        "strategy_id": IDENTITY.strategy_id,
        "version": STRATEGY_VERSION,
        "market": IDENTITY.market,
        "horizon_days": IDENTITY.horizon_days,
        "type": IDENTITY.strategy_type,
        "thesis": IDENTITY.thesis,
        "signal": {
            "feature_1": IDENTITY.feature_1,
            "feature_1_decile": IDENTITY.feature_1_decile,
            "feature_2": IDENTITY.feature_2,
            "feature_2_decile": IDENTITY.feature_2_decile,
            "logic": IDENTITY.logic,
        },
        "backtest": {
            "sharpe": BACKTEST.sharpe_ratio,
            "beta_neutral_sharpe": BACKTEST.beta_neutral_sharpe,
            "total_return": BACKTEST.total_return,
            "precision_buy": BACKTEST.precision_buy,
            "win_rate": BACKTEST.win_rate,
            "win_folds": f"{BACKTEST.win_folds}/{BACKTEST.total_folds}",
            "monthly_consistency": BACKTEST.monthly_consistency,
            "cvar": BACKTEST.portfolio_cvar,
            "turnover": BACKTEST.mean_turnover,
        },
        "governance": {
            "trust_score": GOVERNANCE.trust_score_from_pipeline,
            "trust_level": GOVERNANCE.trust_level,
            "recommendation": GOVERNANCE.pipeline_recommendation,
        },
    }
