"""
Backtesting Engine
-------------------
Walk-forward portfolio simulation with realistic transaction costs and slippage.

Usage:
    config = BacktestConfig(initial_capital=100_000)
    bt = Backtester(config)
    result = bt.run(predictions, realized_returns, weights)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rebalance_frequency: int = 21  # trading days between rebalances
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 10.0  # basis points per trade
    slippage_bps: float = 5.0  # basis points slippage


@dataclass
class BacktestResult:
    """Backtesting output."""
    equity_curve: np.ndarray
    daily_returns: np.ndarray
    rolling_sharpe: np.ndarray  # 63-day rolling
    drawdown_series: np.ndarray
    weight_history: List[np.ndarray]
    turnover: float
    hit_ratio_rolling: np.ndarray  # 63-day rolling
    summary: Dict[str, float]


class Backtester:
    """Walk-forward backtesting engine with transaction costs."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        predictions: np.ndarray,
        realized_returns: np.ndarray,
        weights: np.ndarray,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            predictions: predicted returns (n_periods,) or (n_periods, n_assets).
            realized_returns: actual returns matching predictions shape.
            weights: portfolio weights (n_periods,) or (n_periods, n_assets).
                     For single-asset: signal strength; for multi-asset: allocation.

        Returns:
            BacktestResult with equity curve, metrics, and diagnostics.
        """
        # Normalize inputs to 1D
        if predictions.ndim == 2:
            predictions = predictions.mean(axis=1)
        if realized_returns.ndim == 2:
            realized_returns = realized_returns.mean(axis=1)
        if weights.ndim == 2:
            weights = weights.mean(axis=1)

        n = len(realized_returns)
        cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000.0

        # Build position series (rebalance every N days)
        positions = np.zeros(n)
        rebal = self.config.rebalance_frequency
        for i in range(0, n, rebal):
            end = min(i + rebal, n)
            # Position = sign of prediction * weight magnitude
            pos = np.sign(predictions[i]) * abs(weights[i]) if abs(predictions[i]) > 1e-8 else 0.0
            positions[i:end] = pos

        # Compute portfolio returns with transaction costs
        daily_returns = np.zeros(n)
        turnover_sum = 0.0
        weight_history = []

        prev_pos = 0.0
        for i in range(n):
            # Transaction cost from position change
            trade = abs(positions[i] - prev_pos)
            cost = trade * cost_rate
            turnover_sum += trade

            daily_returns[i] = positions[i] * realized_returns[i] - cost
            prev_pos = positions[i]
            weight_history.append(np.array([positions[i]]))

        # Equity curve
        equity = np.zeros(n + 1)
        equity[0] = self.config.initial_capital
        for i in range(n):
            equity[i + 1] = equity[i] * (1 + daily_returns[i])
        equity_curve = equity[1:]  # drop initial

        # Drawdown series
        running_max = np.maximum.accumulate(equity_curve)
        drawdown_series = equity_curve / running_max - 1.0

        # Rolling Sharpe (63-day)
        rolling_sharpe = np.full(n, np.nan)
        window = 63
        for i in range(window, n):
            chunk = daily_returns[i - window:i]
            mean_r = np.mean(chunk)
            std_r = np.std(chunk)
            if std_r > 1e-10:
                rolling_sharpe[i] = mean_r / std_r * np.sqrt(252)

        # Rolling hit ratio (63-day)
        hit_ratio_rolling = np.full(n, np.nan)
        for i in range(window, n):
            chunk_pred = predictions[i - window:i]
            chunk_real = realized_returns[i - window:i]
            correct = np.sum(np.sign(chunk_pred) == np.sign(chunk_real))
            hit_ratio_rolling[i] = correct / window

        # Summary statistics
        total_return = (equity_curve[-1] / self.config.initial_capital - 1) if n > 0 else 0.0
        annual_return = (1 + total_return) ** (252 / max(n, 1)) - 1
        annual_vol = np.std(daily_returns) * np.sqrt(252) if n > 1 else 0.0
        sharpe = annual_return / max(annual_vol, 1e-8)
        max_dd = float(np.min(drawdown_series)) if n > 0 else 0.0
        calmar = annual_return / max(abs(max_dd), 1e-8) if max_dd != 0 else 0.0
        hit_ratio = float(np.nanmean(np.sign(predictions) == np.sign(realized_returns)))
        avg_turnover = turnover_sum / max(n / rebal, 1)

        summary = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "hit_ratio": float(hit_ratio),
            "avg_turnover_per_rebalance": float(avg_turnover),
            "n_periods": n,
        }

        return BacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            rolling_sharpe=rolling_sharpe,
            drawdown_series=drawdown_series,
            weight_history=weight_history,
            turnover=float(turnover_sum),
            hit_ratio_rolling=hit_ratio_rolling,
            summary=summary,
        )
