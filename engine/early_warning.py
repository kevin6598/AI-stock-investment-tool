"""
Early Warning Engine
=====================
Failure precursor detection for the surviving strategy.

Monitors 6 structural health indicators and produces a normalized
warning score [0, 1] that drives exposure decisions.

Conceptual model:
  Strategy Structure → Health Indicators → Warning Score → Exposure Multiplier

Warning Score Composition (weights sum to 1.0):
  0.25  Momentum dispersion collapse
  0.20  Value trap detection (d0 forward returns)
  0.20  Volatility regime (VIX relative to historical)
  0.15  Cross-market contagion
  0.10  Volume anomaly
  0.10  Sector concentration drift
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EarlyWarningConfig:
    """Configuration for early warning thresholds."""
    # Momentum dispersion
    mom_dispersion_window: int = 252
    mom_dispersion_zscore_threshold: float = -1.0  # below -1 stdev = warning

    # Value trap
    value_trap_lookback_periods: int = 3
    value_trap_return_threshold: float = 0.0  # consecutive negative = trap

    # Volatility regime
    vix_ma_window: int = 252
    vix_danger_multiple: float = 2.0   # VIX > 2x MA = full warning
    vix_caution_multiple: float = 1.5  # VIX > 1.5x MA = partial warning

    # Cross-market contagion
    contagion_return_threshold: float = -0.03  # both markets down >3%
    contagion_correlation_window: int = 21

    # Volume anomaly
    volume_ma_window: int = 20
    volume_dry_threshold: float = 0.5  # below 50% of MA = warning

    # Sector concentration
    max_sector_pct: float = 0.60  # >60% in one sector = warning


@dataclass
class WarningSignal:
    """Individual warning signal with score and context."""
    name: str
    score: float           # 0.0 = no warning, 1.0 = maximum warning
    weight: float          # contribution weight to total score
    detail: str            # human-readable explanation
    raw_value: float = 0.0 # the underlying metric value


@dataclass
class EarlyWarningReport:
    """Complete early warning assessment."""
    warning_score: float           # normalized [0, 1]
    level: str                     # HEALTHY / CAUTION / WARNING / DANGER / CRITICAL
    exposure_multiplier: float     # recommended exposure [0, 1]
    signals: List[WarningSignal]
    timestamp: str = ""
    strategy_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "warning_score": round(self.warning_score, 4),
            "level": self.level,
            "exposure_multiplier": round(self.exposure_multiplier, 4),
            "signals": [
                {
                    "name": s.name,
                    "score": round(s.score, 4),
                    "weight": s.weight,
                    "weighted_contribution": round(s.score * s.weight, 4),
                    "detail": s.detail,
                    "raw_value": round(s.raw_value, 4) if s.raw_value is not None else None,
                }
                for s in self.signals
            ],
            "timestamp": self.timestamp,
            "strategy_id": self.strategy_id,
        }


class EarlyWarningEngine:
    """Detects failure precursors for the surviving strategy.

    Usage:
        engine = EarlyWarningEngine()
        report = engine.evaluate(
            us_close=spy_close_series,
            us_returns=spy_daily_returns,
            signal_stock_returns=signal_stocks_fwd_returns,
            vix_series=vix_close_series,
            kr_returns=kospi_daily_returns,
            signal_volumes=signal_stock_volumes,
            signal_avg_volumes=signal_stock_avg_volumes,
            signal_sectors=signal_stock_sector_map,
        )
    """

    # Weight allocation for each warning dimension
    WEIGHTS = {
        "momentum_dispersion": 0.25,
        "value_trap": 0.20,
        "volatility_regime": 0.20,
        "cross_market_contagion": 0.15,
        "volume_anomaly": 0.10,
        "sector_concentration": 0.10,
    }

    def __init__(self, config: Optional[EarlyWarningConfig] = None):
        self.config = config or EarlyWarningConfig()

    def evaluate(
        self,
        us_close: Optional[Any] = None,
        us_returns: Optional[Any] = None,
        signal_stock_returns: Optional[Any] = None,
        vix_series: Optional[Any] = None,
        kr_returns: Optional[Any] = None,
        signal_volumes: Optional[Any] = None,
        signal_avg_volumes: Optional[Any] = None,
        signal_sectors: Optional[Dict[str, str]] = None,
        momentum_values: Optional[Any] = None,
    ) -> EarlyWarningReport:
        """Run all warning checks and produce composite score.

        All inputs are optional. Missing data produces a neutral (0.5)
        score for that dimension, with a note indicating data was unavailable.

        Args:
            us_close: US market close prices (e.g., SPY), pandas Series.
            us_returns: US market daily returns, numpy array.
            signal_stock_returns: Forward returns of stocks in signal universe.
            vix_series: VIX close price series, pandas Series.
            kr_returns: Korean market daily returns, numpy array.
            signal_volumes: Recent volumes of signal stocks, numpy array.
            signal_avg_volumes: Average volumes of signal stocks, numpy array.
            signal_sectors: Dict of ticker -> sector for signal stocks.
            momentum_values: Cross-sectional 60d momentum values for all stocks.

        Returns:
            EarlyWarningReport with composite warning score and individual signals.
        """
        from datetime import datetime

        signals = []

        # 1. Momentum dispersion
        signals.append(self._check_momentum_dispersion(momentum_values))

        # 2. Value trap
        signals.append(self._check_value_trap(signal_stock_returns))

        # 3. Volatility regime
        signals.append(self._check_volatility_regime(vix_series))

        # 4. Cross-market contagion
        signals.append(self._check_cross_market_contagion(us_returns, kr_returns))

        # 5. Volume anomaly
        signals.append(self._check_volume_anomaly(signal_volumes, signal_avg_volumes))

        # 6. Sector concentration
        signals.append(self._check_sector_concentration(signal_sectors))

        # Compute weighted score
        warning_score = sum(s.score * s.weight for s in signals)
        warning_score = max(0.0, min(1.0, warning_score))

        # Map to level and exposure
        exposure_multiplier, level = self._score_to_exposure(warning_score)

        return EarlyWarningReport(
            warning_score=warning_score,
            level=level,
            exposure_multiplier=exposure_multiplier,
            signals=signals,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            strategy_id="US_63d_mom_60d_decile_d4_AND_high_52w_pct_decile_d0",
        )

    def _check_momentum_dispersion(
        self, momentum_values: Optional[Any]
    ) -> WarningSignal:
        """Check if cross-sectional momentum dispersion is collapsing.

        When all stocks move together, decile-based selection fails.
        """
        weight = self.WEIGHTS["momentum_dispersion"]

        if momentum_values is None:
            return WarningSignal(
                name="momentum_dispersion",
                score=0.3,  # mild uncertainty penalty
                weight=weight,
                detail="Momentum data unavailable; assuming mild caution.",
                raw_value=0.0,
            )

        arr = np.asarray(momentum_values, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 20:
            return WarningSignal(
                name="momentum_dispersion",
                score=0.3,
                weight=weight,
                detail=f"Insufficient momentum data ({len(arr)} stocks).",
                raw_value=0.0,
            )

        dispersion = float(np.std(arr))
        mean_dispersion = float(np.mean(np.abs(arr)))

        # Compare current dispersion to what we'd expect
        # Low dispersion = stocks moving together = bad for decile strategies
        # Use a simple heuristic: if dispersion < 0.05, high warning
        if dispersion < 0.02:
            score = 1.0
            detail = f"Momentum dispersion collapsed ({dispersion:.4f}). Decile selection unreliable."
        elif dispersion < 0.05:
            score = 0.6
            detail = f"Momentum dispersion low ({dispersion:.4f}). Decile spread narrowing."
        elif dispersion < 0.08:
            score = 0.2
            detail = f"Momentum dispersion moderate ({dispersion:.4f}). Normal range."
        else:
            score = 0.0
            detail = f"Momentum dispersion healthy ({dispersion:.4f}). Good decile separation."

        return WarningSignal(
            name="momentum_dispersion",
            score=score,
            weight=weight,
            detail=detail,
            raw_value=dispersion,
        )

    def _check_value_trap(
        self, signal_stock_returns: Optional[Any]
    ) -> WarningSignal:
        """Check if stocks near 52-week lows keep declining (value trap)."""
        weight = self.WEIGHTS["value_trap"]

        if signal_stock_returns is None:
            return WarningSignal(
                name="value_trap",
                score=0.3,
                weight=weight,
                detail="Signal stock return data unavailable.",
                raw_value=0.0,
            )

        arr = np.asarray(signal_stock_returns, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 1:
            return WarningSignal(
                name="value_trap",
                score=0.3,
                weight=weight,
                detail="No signal stock returns available.",
                raw_value=0.0,
            )

        mean_return = float(np.mean(arr))
        negative_pct = float(np.mean(arr < 0))

        if mean_return < -0.05 and negative_pct > 0.7:
            score = 1.0
            detail = (
                f"Value trap active: mean return {mean_return:.2%}, "
                f"{negative_pct:.0%} negative. D0 stocks still falling."
            )
        elif mean_return < -0.02 and negative_pct > 0.6:
            score = 0.6
            detail = (
                f"Value trap emerging: mean return {mean_return:.2%}, "
                f"{negative_pct:.0%} negative."
            )
        elif mean_return < 0:
            score = 0.3
            detail = f"Signal returns mildly negative ({mean_return:.2%})."
        else:
            score = 0.0
            detail = f"Signal returns positive ({mean_return:.2%}). Mean reversion working."

        return WarningSignal(
            name="value_trap",
            score=score,
            weight=weight,
            detail=detail,
            raw_value=mean_return,
        )

    def _check_volatility_regime(
        self, vix_series: Optional[Any]
    ) -> WarningSignal:
        """Check if VIX indicates extreme volatility regime."""
        weight = self.WEIGHTS["volatility_regime"]

        if vix_series is None:
            return WarningSignal(
                name="volatility_regime",
                score=0.3,
                weight=weight,
                detail="VIX data unavailable.",
                raw_value=0.0,
            )

        arr = np.asarray(vix_series, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 30:
            return WarningSignal(
                name="volatility_regime",
                score=0.3,
                weight=weight,
                detail=f"Insufficient VIX data ({len(arr)} days).",
                raw_value=0.0,
            )

        current_vix = float(arr[-1])
        window = min(len(arr), self.config.vix_ma_window)
        vix_ma = float(np.mean(arr[-window:]))
        vix_ratio = current_vix / vix_ma if vix_ma > 0 else 1.0

        if vix_ratio >= self.config.vix_danger_multiple:
            score = 1.0
            detail = (
                f"VIX at {current_vix:.1f} ({vix_ratio:.2f}x MA). "
                f"Extreme volatility regime. 63-day horizon unreliable."
            )
        elif vix_ratio >= self.config.vix_caution_multiple:
            score = 0.6
            detail = (
                f"VIX elevated at {current_vix:.1f} ({vix_ratio:.2f}x MA). "
                f"Approaching danger zone."
            )
        elif vix_ratio >= 1.2:
            score = 0.2
            detail = f"VIX slightly elevated ({current_vix:.1f}, {vix_ratio:.2f}x MA)."
        else:
            score = 0.0
            detail = f"VIX normal ({current_vix:.1f}, {vix_ratio:.2f}x MA)."

        return WarningSignal(
            name="volatility_regime",
            score=score,
            weight=weight,
            detail=detail,
            raw_value=vix_ratio,
        )

    def _check_cross_market_contagion(
        self,
        us_returns: Optional[Any],
        kr_returns: Optional[Any],
    ) -> WarningSignal:
        """Check for cross-market contagion from Korean markets."""
        weight = self.WEIGHTS["cross_market_contagion"]

        if us_returns is None or kr_returns is None:
            return WarningSignal(
                name="cross_market_contagion",
                score=0.0,
                weight=weight,
                detail="Cross-market data unavailable. No contagion signal.",
                raw_value=0.0,
            )

        us = np.asarray(us_returns, dtype=float)
        kr = np.asarray(kr_returns, dtype=float)

        min_len = min(len(us), len(kr))
        if min_len < 5:
            return WarningSignal(
                name="cross_market_contagion",
                score=0.0,
                weight=weight,
                detail="Insufficient cross-market data.",
                raw_value=0.0,
            )

        us_recent = us[-min_len:]
        kr_recent = kr[-min_len:]

        # Check correlation spike
        window = min(min_len, self.config.contagion_correlation_window)
        us_w = us_recent[-window:]
        kr_w = kr_recent[-window:]

        if np.std(us_w) < 1e-10 or np.std(kr_w) < 1e-10:
            corr = 0.0
        else:
            corr = float(np.corrcoef(us_w, kr_w)[0, 1])

        # Both markets declining simultaneously
        us_cum = float(np.sum(us_w))
        kr_cum = float(np.sum(kr_w))

        both_negative = us_cum < self.config.contagion_return_threshold and \
                        kr_cum < self.config.contagion_return_threshold
        high_corr = corr > 0.7

        if both_negative and high_corr:
            score = 1.0
            detail = (
                f"Contagion active: US {us_cum:.2%}, KR {kr_cum:.2%}, "
                f"corr={corr:.2f}. Synchronized sell-off."
            )
        elif both_negative:
            score = 0.5
            detail = (
                f"Both markets declining: US {us_cum:.2%}, KR {kr_cum:.2%}. "
                f"Low correlation ({corr:.2f}) limits contagion risk."
            )
        elif high_corr and us_cum < 0:
            score = 0.3
            detail = f"High correlation ({corr:.2f}) with US declining ({us_cum:.2%})."
        else:
            score = 0.0
            detail = f"No contagion signal. US {us_cum:.2%}, corr={corr:.2f}."

        return WarningSignal(
            name="cross_market_contagion",
            score=score,
            weight=weight,
            detail=detail,
            raw_value=corr,
        )

    def _check_volume_anomaly(
        self,
        signal_volumes: Optional[Any],
        signal_avg_volumes: Optional[Any],
    ) -> WarningSignal:
        """Check for volume dry-up in signal stocks."""
        weight = self.WEIGHTS["volume_anomaly"]

        if signal_volumes is None or signal_avg_volumes is None:
            return WarningSignal(
                name="volume_anomaly",
                score=0.0,
                weight=weight,
                detail="Volume data unavailable. No anomaly signal.",
                raw_value=0.0,
            )

        vols = np.asarray(signal_volumes, dtype=float)
        avg_vols = np.asarray(signal_avg_volumes, dtype=float)

        # Filter valid entries
        valid = (avg_vols > 0) & ~np.isnan(vols) & ~np.isnan(avg_vols)
        if valid.sum() < 1:
            return WarningSignal(
                name="volume_anomaly",
                score=0.0,
                weight=weight,
                detail="No valid volume data.",
                raw_value=0.0,
            )

        ratios = vols[valid] / avg_vols[valid]
        mean_ratio = float(np.mean(ratios))
        dry_pct = float(np.mean(ratios < self.config.volume_dry_threshold))

        if mean_ratio < 0.3:
            score = 1.0
            detail = f"Severe volume dry-up: avg ratio {mean_ratio:.2f}x. Illiquidity risk."
        elif mean_ratio < self.config.volume_dry_threshold:
            score = 0.6
            detail = f"Volume below normal: avg ratio {mean_ratio:.2f}x. {dry_pct:.0%} stocks affected."
        elif mean_ratio < 0.8:
            score = 0.2
            detail = f"Volume slightly below normal ({mean_ratio:.2f}x)."
        else:
            score = 0.0
            detail = f"Volume normal ({mean_ratio:.2f}x)."

        return WarningSignal(
            name="volume_anomaly",
            score=score,
            weight=weight,
            detail=detail,
            raw_value=mean_ratio,
        )

    def _check_sector_concentration(
        self, signal_sectors: Optional[Dict[str, str]]
    ) -> WarningSignal:
        """Check if signal stocks are over-concentrated in one sector."""
        weight = self.WEIGHTS["sector_concentration"]

        if signal_sectors is None or len(signal_sectors) == 0:
            return WarningSignal(
                name="sector_concentration",
                score=0.0,
                weight=weight,
                detail="Sector data unavailable.",
                raw_value=0.0,
            )

        # Count sector distribution
        sector_counts: Dict[str, int] = {}
        for ticker, sector in signal_sectors.items():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        total = sum(sector_counts.values())
        if total == 0:
            return WarningSignal(
                name="sector_concentration",
                score=0.0,
                weight=weight,
                detail="No stocks in signal universe.",
                raw_value=0.0,
            )

        max_sector = max(sector_counts, key=sector_counts.get)
        max_pct = sector_counts[max_sector] / total

        if max_pct > 0.8:
            score = 1.0
            detail = (
                f"Extreme sector concentration: {max_pct:.0%} in {max_sector}. "
                f"Cross-sector diversification lost."
            )
        elif max_pct > self.config.max_sector_pct:
            score = 0.6
            detail = f"High concentration: {max_pct:.0%} in {max_sector}."
        elif max_pct > 0.4:
            score = 0.2
            detail = f"Moderate concentration: {max_pct:.0%} in {max_sector}."
        else:
            score = 0.0
            detail = f"Good diversification. Max sector: {max_pct:.0%} in {max_sector}."

        return WarningSignal(
            name="sector_concentration",
            score=score,
            weight=weight,
            detail=detail,
            raw_value=max_pct,
        )

    @staticmethod
    def _score_to_exposure(warning_score: float) -> Tuple[float, str]:
        """Map warning score to exposure multiplier and level name.

        Returns:
            (exposure_multiplier, level_name)
        """
        score = max(0.0, min(1.0, warning_score))

        if score < 0.2:
            return 1.0, "HEALTHY"
        elif score < 0.4:
            # Linear interpolation: 1.0 at 0.2, 0.75 at 0.4
            mult = 1.0 - (score - 0.2) * (0.25 / 0.2)
            return round(mult, 4), "CAUTION"
        elif score < 0.6:
            mult = 0.75 - (score - 0.4) * (0.25 / 0.2)
            return round(mult, 4), "WARNING"
        elif score < 0.8:
            mult = 0.50 - (score - 0.6) * (0.25 / 0.2)
            return round(mult, 4), "DANGER"
        else:
            mult = 0.25 - (score - 0.8) * (0.25 / 0.2)
            return round(max(mult, 0.0), 4), "CRITICAL"


def quick_evaluate(
    vix_current: Optional[float] = None,
    vix_ma: Optional[float] = None,
    us_21d_return: Optional[float] = None,
    kr_21d_return: Optional[float] = None,
    signal_stock_mean_return: Optional[float] = None,
    momentum_dispersion: Optional[float] = None,
    max_sector_pct: Optional[float] = None,
    volume_ratio: Optional[float] = None,
) -> EarlyWarningReport:
    """Convenience function for quick evaluation with scalar inputs.

    Use this when you have pre-computed summary statistics rather than
    raw time series data.
    """
    from datetime import datetime

    engine = EarlyWarningEngine()
    signals = []
    weights = engine.WEIGHTS

    # Momentum dispersion
    if momentum_dispersion is not None:
        if momentum_dispersion < 0.02:
            s = 1.0
        elif momentum_dispersion < 0.05:
            s = 0.6
        elif momentum_dispersion < 0.08:
            s = 0.2
        else:
            s = 0.0
        signals.append(WarningSignal(
            "momentum_dispersion", s, weights["momentum_dispersion"],
            f"Dispersion: {momentum_dispersion:.4f}", momentum_dispersion
        ))
    else:
        signals.append(WarningSignal(
            "momentum_dispersion", 0.3, weights["momentum_dispersion"],
            "Data unavailable", 0.0
        ))

    # Value trap
    if signal_stock_mean_return is not None:
        r = signal_stock_mean_return
        if r < -0.05:
            s = 1.0
        elif r < -0.02:
            s = 0.6
        elif r < 0:
            s = 0.3
        else:
            s = 0.0
        signals.append(WarningSignal(
            "value_trap", s, weights["value_trap"],
            f"Signal mean return: {r:.2%}", r
        ))
    else:
        signals.append(WarningSignal(
            "value_trap", 0.3, weights["value_trap"],
            "Data unavailable", 0.0
        ))

    # Volatility regime
    if vix_current is not None and vix_ma is not None and vix_ma > 0:
        ratio = vix_current / vix_ma
        if ratio >= 2.0:
            s = 1.0
        elif ratio >= 1.5:
            s = 0.6
        elif ratio >= 1.2:
            s = 0.2
        else:
            s = 0.0
        signals.append(WarningSignal(
            "volatility_regime", s, weights["volatility_regime"],
            f"VIX: {vix_current:.1f} ({ratio:.2f}x MA)", ratio
        ))
    else:
        signals.append(WarningSignal(
            "volatility_regime", 0.3, weights["volatility_regime"],
            "VIX data unavailable", 0.0
        ))

    # Cross-market contagion
    if us_21d_return is not None and kr_21d_return is not None:
        both_neg = us_21d_return < -0.03 and kr_21d_return < -0.03
        if both_neg:
            s = 0.8
        elif us_21d_return < -0.03:
            s = 0.3
        else:
            s = 0.0
        signals.append(WarningSignal(
            "cross_market_contagion", s, weights["cross_market_contagion"],
            f"US: {us_21d_return:.2%}, KR: {kr_21d_return:.2%}", us_21d_return
        ))
    else:
        signals.append(WarningSignal(
            "cross_market_contagion", 0.0, weights["cross_market_contagion"],
            "Data unavailable", 0.0
        ))

    # Volume anomaly
    if volume_ratio is not None:
        if volume_ratio < 0.3:
            s = 1.0
        elif volume_ratio < 0.5:
            s = 0.6
        elif volume_ratio < 0.8:
            s = 0.2
        else:
            s = 0.0
        signals.append(WarningSignal(
            "volume_anomaly", s, weights["volume_anomaly"],
            f"Volume ratio: {volume_ratio:.2f}x", volume_ratio
        ))
    else:
        signals.append(WarningSignal(
            "volume_anomaly", 0.0, weights["volume_anomaly"],
            "Data unavailable", 0.0
        ))

    # Sector concentration
    if max_sector_pct is not None:
        if max_sector_pct > 0.8:
            s = 1.0
        elif max_sector_pct > 0.6:
            s = 0.6
        elif max_sector_pct > 0.4:
            s = 0.2
        else:
            s = 0.0
        signals.append(WarningSignal(
            "sector_concentration", s, weights["sector_concentration"],
            f"Max sector: {max_sector_pct:.0%}", max_sector_pct
        ))
    else:
        signals.append(WarningSignal(
            "sector_concentration", 0.0, weights["sector_concentration"],
            "Data unavailable", 0.0
        ))

    warning_score = sum(s.score * s.weight for s in signals)
    warning_score = max(0.0, min(1.0, warning_score))
    exposure, level = EarlyWarningEngine._score_to_exposure(warning_score)

    return EarlyWarningReport(
        warning_score=warning_score,
        level=level,
        exposure_multiplier=exposure,
        signals=signals,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        strategy_id="US_63d_mom_60d_decile_d4_AND_high_52w_pct_decile_d0",
    )
