"""
Data Validation Layer
---------------------
Validates OHLCV data quality before it enters the ML pipeline.

Checks for:
  - Missing trading days
  - Duplicate dates
  - Price spikes (|daily_return| > 0.50)
  - Zero-volume streaks (> 3 consecutive)
  - Stale prices (same close > 5 consecutive days)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report from validating a single ticker's OHLCV data."""
    ticker: str
    total_bars: int
    missing_bars: int          # expected trading days vs actual
    duplicate_dates: int
    price_spike_count: int     # |daily_return| > 0.50
    zero_volume_runs: int      # consecutive zero-volume streaks > 3
    stale_price_days: int      # same close for > 5 consecutive days
    passed: bool
    warnings: List[str] = field(default_factory=list)


def validate_ohlcv(
    df: pd.DataFrame,
    ticker: str = "",
    spike_threshold: float = 0.50,
    zero_vol_streak: int = 3,
    stale_streak: int = 5,
) -> ValidationReport:
    """Validate OHLCV DataFrame for data quality issues.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex.
        ticker: Ticker symbol for reporting.
        spike_threshold: Flag daily returns exceeding this absolute value.
        zero_vol_streak: Flag consecutive zero-volume runs longer than this.
        stale_streak: Flag same-close streaks longer than this.

    Returns:
        ValidationReport with all findings.
    """
    warnings = []  # type: List[str]

    if df.empty:
        return ValidationReport(
            ticker=ticker, total_bars=0, missing_bars=0,
            duplicate_dates=0, price_spike_count=0,
            zero_volume_runs=0, stale_price_days=0,
            passed=False, warnings=["Empty DataFrame"],
        )

    total_bars = len(df)

    # --- Duplicate dates ---
    dup_count = int(df.index.duplicated().sum())
    if dup_count > 0:
        warnings.append("Found {} duplicate dates".format(dup_count))

    # --- Missing bars ---
    if hasattr(df.index, 'freq') or len(df) > 1:
        idx = pd.DatetimeIndex(df.index)
        if len(idx) >= 2:
            expected = pd.bdate_range(idx.min(), idx.max())
            missing = len(expected) - len(idx.unique())
            missing = max(missing, 0)
        else:
            missing = 0
    else:
        missing = 0

    if missing > 0:
        missing_pct = missing / max(len(pd.bdate_range(df.index.min(), df.index.max())), 1)
        if missing_pct > 0.05:
            warnings.append("Missing {:.1%} of expected trading days ({} bars)".format(
                missing_pct, missing))

    # --- Price spikes ---
    close = df["Close"].values.astype(float)
    daily_returns = np.diff(close) / (np.abs(close[:-1]) + 1e-10)
    spike_count = int(np.sum(np.abs(daily_returns) > spike_threshold))
    if spike_count > 0:
        warnings.append("Found {} price spikes (|ret| > {:.0%})".format(
            spike_count, spike_threshold))

    # --- Zero-volume streaks ---
    vol = df["Volume"].values.astype(float) if "Volume" in df.columns else np.ones(total_bars)
    zero_vol = (vol == 0).astype(int)
    zero_runs = 0
    current_run = 0
    for v in zero_vol:
        if v:
            current_run += 1
        else:
            if current_run > zero_vol_streak:
                zero_runs += 1
            current_run = 0
    if current_run > zero_vol_streak:
        zero_runs += 1
    if zero_runs > 0:
        warnings.append("Found {} zero-volume streaks (> {} consecutive)".format(
            zero_runs, zero_vol_streak))

    # --- Stale prices ---
    stale_days = 0
    current_stale = 1
    for i in range(1, len(close)):
        if abs(close[i] - close[i - 1]) < 1e-8:
            current_stale += 1
        else:
            if current_stale > stale_streak:
                stale_days += current_stale
            current_stale = 1
    if current_stale > stale_streak:
        stale_days += current_stale

    if stale_days > 0:
        warnings.append("Found {} stale-price days (same close > {} consecutive)".format(
            stale_days, stale_streak))

    # --- Pass/Fail ---
    passed = (
        spike_count == 0
        and zero_runs == 0
        and stale_days == 0
        and dup_count == 0
    )

    report = ValidationReport(
        ticker=ticker,
        total_bars=total_bars,
        missing_bars=missing,
        duplicate_dates=dup_count,
        price_spike_count=spike_count,
        zero_volume_runs=zero_runs,
        stale_price_days=stale_days,
        passed=passed,
        warnings=warnings,
    )

    if not passed:
        logger.warning("Validation failed for %s: %s", ticker, "; ".join(warnings))

    return report


def validate_panel(
    panel: pd.DataFrame,
    ticker_col: str = "ticker",
) -> Dict[str, ValidationReport]:
    """Validate all tickers in a panel DataFrame.

    Args:
        panel: DataFrame with MultiIndex (date, ticker) or a 'ticker' column.
        ticker_col: Column name containing ticker symbols (if not MultiIndex).

    Returns:
        Dict of ticker -> ValidationReport.
    """
    reports = {}  # type: Dict[str, ValidationReport]

    if isinstance(panel.index, pd.MultiIndex):
        tickers = panel.index.get_level_values("ticker").unique()
        for ticker in tickers:
            sub = panel.xs(ticker, level="ticker")
            # Try to reconstruct OHLCV-like data for validation
            if "Close" in sub.columns or "_close" in sub.columns:
                close_col = "Close" if "Close" in sub.columns else "_close"
                mock_df = pd.DataFrame({
                    "Open": sub[close_col] if close_col in sub.columns else 0,
                    "High": sub[close_col] if close_col in sub.columns else 0,
                    "Low": sub[close_col] if close_col in sub.columns else 0,
                    "Close": sub[close_col],
                    "Volume": sub["Volume"] if "Volume" in sub.columns else 1,
                }, index=sub.index)
                reports[ticker] = validate_ohlcv(mock_df, ticker)
            else:
                reports[ticker] = ValidationReport(
                    ticker=ticker, total_bars=len(sub),
                    missing_bars=0, duplicate_dates=0,
                    price_spike_count=0, zero_volume_runs=0,
                    stale_price_days=0, passed=True,
                    warnings=["No Close column available for validation"],
                )
    else:
        if ticker_col in panel.columns:
            for ticker in panel[ticker_col].unique():
                sub = panel[panel[ticker_col] == ticker]
                reports[ticker] = validate_ohlcv(sub, ticker)

    return reports


def filter_invalid_tickers(
    reports: Dict[str, ValidationReport],
    max_missing_pct: float = 0.05,
) -> List[str]:
    """Filter tickers that pass validation.

    Args:
        reports: Dict of ticker -> ValidationReport.
        max_missing_pct: Maximum fraction of missing bars allowed.

    Returns:
        List of valid ticker symbols.
    """
    valid = []
    for ticker, report in reports.items():
        if report.total_bars == 0:
            continue

        missing_pct = report.missing_bars / max(report.total_bars, 1)
        if missing_pct > max_missing_pct:
            logger.info("Filtering %s: %.1f%% missing bars", ticker, missing_pct * 100)
            continue

        if report.price_spike_count > 5:
            logger.info("Filtering %s: %d price spikes", ticker, report.price_spike_count)
            continue

        if report.zero_volume_runs > 3:
            logger.info("Filtering %s: %d zero-volume runs", ticker, report.zero_volume_runs)
            continue

        valid.append(ticker)

    return valid
