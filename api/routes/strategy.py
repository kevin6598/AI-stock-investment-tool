"""
Strategy Governance Routes
---------------------------
GET  /api/v1/strategy/status          - Strategy identity & backtest metrics
GET  /api/v1/strategy/early-warning   - Live early warning assessment
GET  /api/v1/strategy/exposure-guidance - Capital exposure recommendation
"""

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter

from api.schemas import (
    StrategyStatusResponse,
    StrategySignalSchema,
    StrategyBacktestSchema,
    StrategyGovernanceSchema,
    EarlyWarningResponse,
    WarningSignalSchema,
    ExposureGuidanceResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/strategy", tags=["strategy"])


def _get_live_market_data():
    """Fetch live market data for early warning computation.

    Returns dict with keys: vix_current, vix_ma, us_returns, kr_returns, etc.
    Returns None values for data that cannot be fetched.
    """
    result = {
        "vix_current": None,
        "vix_ma": None,
        "us_21d_return": None,
        "kr_21d_return": None,
        "momentum_dispersion": None,
        "signal_stock_mean_return": None,
        "max_sector_pct": None,
        "volume_ratio": None,
    }

    try:
        from data.stock_api import get_historical_data

        # VIX data
        try:
            vix_df = get_historical_data("^VIX", period="2y")
            if not vix_df.empty and "Close" in vix_df.columns:
                vix_close = vix_df["Close"].dropna()
                if len(vix_close) > 252:
                    result["vix_current"] = float(vix_close.iloc[-1])
                    result["vix_ma"] = float(vix_close.iloc[-252:].mean())
                elif len(vix_close) > 30:
                    result["vix_current"] = float(vix_close.iloc[-1])
                    result["vix_ma"] = float(vix_close.mean())
        except Exception as e:
            logger.debug("VIX fetch failed: %s", e)

        # US market returns (SPY)
        try:
            spy_df = get_historical_data("SPY", period="1y")
            if not spy_df.empty and "Close" in spy_df.columns:
                spy_close = spy_df["Close"].dropna()
                if len(spy_close) > 21:
                    ret_21d = float(
                        spy_close.iloc[-1] / spy_close.iloc[-22] - 1
                    )
                    result["us_21d_return"] = ret_21d
        except Exception as e:
            logger.debug("SPY fetch failed: %s", e)

        # Korean market returns (EWY as proxy)
        try:
            ewy_df = get_historical_data("EWY", period="1y")
            if not ewy_df.empty and "Close" in ewy_df.columns:
                ewy_close = ewy_df["Close"].dropna()
                if len(ewy_close) > 21:
                    ret_21d = float(
                        ewy_close.iloc[-1] / ewy_close.iloc[-22] - 1
                    )
                    result["kr_21d_return"] = ret_21d
        except Exception as e:
            logger.debug("EWY fetch failed: %s", e)

        # Momentum dispersion (from SPY components or broad market)
        try:
            broad_tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                "META", "TSLA", "JPM", "V", "JNJ",
                "WMT", "PG", "XOM", "HD", "BAC",
                "MA", "DIS", "CSCO", "PFE", "KO",
            ]
            mom_values = []
            for t in broad_tickers:
                try:
                    df = get_historical_data(t, period="6mo")
                    if not df.empty and "Close" in df.columns and len(df) > 60:
                        close = df["Close"].dropna()
                        mom_60d = float(
                            np.log(close.iloc[-1] / close.iloc[-61])
                        )
                        mom_values.append(mom_60d)
                except Exception:
                    continue

            if len(mom_values) >= 10:
                result["momentum_dispersion"] = float(np.std(mom_values))
        except Exception as e:
            logger.debug("Momentum dispersion computation failed: %s", e)

    except ImportError:
        logger.warning("data.stock_api not available; using defaults")

    return result


@router.get("/status", response_model=StrategyStatusResponse)
async def strategy_status():
    """Return the canonical identity and backtest metrics of the surviving strategy."""
    from strategy_definition.us_63d_mom_60d_d4_52w_d0 import get_strategy_summary

    summary = get_strategy_summary()

    return StrategyStatusResponse(
        strategy_id=summary["strategy_id"],
        version=summary["version"],
        market=summary["market"],
        horizon_days=summary["horizon_days"],
        type=summary["type"],
        thesis=summary["thesis"],
        signal=StrategySignalSchema(**summary["signal"]),
        backtest=StrategyBacktestSchema(**summary["backtest"]),
        governance=StrategyGovernanceSchema(**summary["governance"]),
    )


@router.get("/early-warning", response_model=EarlyWarningResponse)
async def early_warning():
    """Compute live early warning assessment using current market data."""
    from engine.early_warning import quick_evaluate

    market_data = _get_live_market_data()

    report = quick_evaluate(
        vix_current=market_data["vix_current"],
        vix_ma=market_data["vix_ma"],
        us_21d_return=market_data["us_21d_return"],
        kr_21d_return=market_data["kr_21d_return"],
        signal_stock_mean_return=market_data["signal_stock_mean_return"],
        momentum_dispersion=market_data["momentum_dispersion"],
        max_sector_pct=market_data["max_sector_pct"],
        volume_ratio=market_data["volume_ratio"],
    )

    report_dict = report.to_dict()

    return EarlyWarningResponse(
        warning_score=report_dict["warning_score"],
        level=report_dict["level"],
        exposure_multiplier=report_dict["exposure_multiplier"],
        signals=[WarningSignalSchema(**s) for s in report_dict["signals"]],
        timestamp=report_dict["timestamp"],
        strategy_id=report_dict["strategy_id"],
    )


@router.get("/exposure-guidance", response_model=ExposureGuidanceResponse)
async def exposure_guidance():
    """Return capital exposure recommendation based on early warning state."""
    from engine.early_warning import quick_evaluate

    market_data = _get_live_market_data()

    report = quick_evaluate(
        vix_current=market_data["vix_current"],
        vix_ma=market_data["vix_ma"],
        us_21d_return=market_data["us_21d_return"],
        kr_21d_return=market_data["kr_21d_return"],
        signal_stock_mean_return=market_data["signal_stock_mean_return"],
        momentum_dispersion=market_data["momentum_dispersion"],
        max_sector_pct=market_data["max_sector_pct"],
        volume_ratio=market_data["volume_ratio"],
    )

    # Generate human-readable guidance
    level = report.level
    mult = report.exposure_multiplier

    if level == "HEALTHY":
        action = "MAINTAIN FULL POSITION"
        guidance = (
            f"Strategy structure is healthy. Maintain {mult:.0%} exposure. "
            f"All 6 structural indicators within normal bounds."
        )
    elif level == "CAUTION":
        action = "REDUCE TO 75%"
        guidance = (
            f"Minor structural concerns detected. Reduce exposure to ~{mult:.0%}. "
            f"Monitor active warning signals for escalation."
        )
    elif level == "WARNING":
        action = "REDUCE TO 50%"
        guidance = (
            f"Multiple warning signals active. Reduce exposure to ~{mult:.0%}. "
            f"Strategy's structural assumptions are being tested."
        )
    elif level == "DANGER":
        action = "REDUCE TO 25%"
        guidance = (
            f"Significant structural degradation. Reduce exposure to ~{mult:.0%}. "
            f"Consider exiting positions that can be closed at reasonable cost."
        )
    else:  # CRITICAL
        action = "EXIT ALL POSITIONS"
        guidance = (
            "Strategy structure has failed. Exit all positions. "
            "Do not re-enter until warning score drops below 0.4."
        )

    return ExposureGuidanceResponse(
        strategy_id=report.strategy_id,
        warning_score=round(report.warning_score, 4),
        warning_level=level,
        exposure_multiplier=round(mult, 4),
        recommended_action=action,
        position_guidance=guidance,
        timestamp=report.timestamp,
    )
