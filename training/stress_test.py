"""
Stress Testing Module
----------------------
Scenario-based stress testing for ML portfolio models.

Tests model and portfolio performance under historical crisis scenarios
by applying return shocks, volatility multipliers, and correlation shifts
to the return series.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    vol_multiplier: float
    return_shock: float
    duration_days: int
    correlation_boost: float


# Default scenarios calibrated to historical events
SCENARIOS = {
    "financial_crisis_2008": StressScenario(
        "2008 Crisis",
        vol_multiplier=3.0,
        return_shock=-0.40,
        duration_days=252,
        correlation_boost=0.3,
    ),
    "covid_crash_2020": StressScenario(
        "COVID Crash",
        vol_multiplier=4.0,
        return_shock=-0.35,
        duration_days=30,
        correlation_boost=0.4,
    ),
    "high_volatility": StressScenario(
        "High Vol",
        vol_multiplier=2.0,
        return_shock=-0.10,
        duration_days=63,
        correlation_boost=0.1,
    ),
    "low_liquidity": StressScenario(
        "Low Liquidity",
        vol_multiplier=1.5,
        return_shock=-0.05,
        duration_days=63,
        correlation_boost=0.2,
    ),
}


@dataclass
class StressResult:
    """Result from a single stress test scenario."""
    scenario_name: str
    stress_sharpe: float
    stress_max_drawdown: float
    tail_loss_p5: float
    survival: bool  # did portfolio survive without breaching risk limits?


class StressTester:
    """Scenario-based stress testing engine.

    Simulates portfolio performance under stressed market conditions
    by modifying the return distribution.

    Usage:
        tester = StressTester()
        results = tester.run_all(model, features, returns)
    """

    def __init__(
        self,
        scenarios: Optional[Dict[str, StressScenario]] = None,
        max_drawdown_limit: float = 0.50,
        random_seed: int = 42,
    ):
        """
        Args:
            scenarios: Dict of scenario_key -> StressScenario.
                If None, uses default SCENARIOS.
            max_drawdown_limit: Drawdown threshold for survival flag.
            random_seed: Random seed for reproducibility.
        """
        self.scenarios = scenarios or SCENARIOS
        self.max_drawdown_limit = max_drawdown_limit
        self.random_seed = random_seed

    def simulate_scenario(
        self,
        returns: np.ndarray,
        scenario: StressScenario,
    ) -> np.ndarray:
        """Apply stress scenario to a return series.

        Transforms returns by:
          1. Scaling volatility by vol_multiplier
          2. Adding return shock spread over duration
          3. Boosting correlation (for multi-asset: shift towards common factor)

        Args:
            returns: Array of daily returns (n_days,) or (n_days, n_assets).
            scenario: The stress scenario to apply.

        Returns:
            Stressed return array of the same shape.
        """
        rng = np.random.RandomState(self.random_seed)
        stressed = returns.copy().astype(np.float64)

        is_panel = stressed.ndim == 2

        # Determine stress duration (cap at available data)
        n_days = stressed.shape[0]
        stress_days = min(scenario.duration_days, n_days)

        if stress_days <= 0:
            return stressed

        # Apply to the last `stress_days` of data
        start = n_days - stress_days

        if is_panel:
            # Multi-asset case
            stress_slice = stressed[start:]

            # 1. Scale volatility
            stress_slice *= scenario.vol_multiplier

            # 2. Add return shock (spread evenly)
            daily_shock = scenario.return_shock / stress_days
            stress_slice += daily_shock

            # 3. Correlation boost: add common factor
            if scenario.correlation_boost > 0:
                common_factor = rng.randn(stress_days) * scenario.correlation_boost
                stress_slice += common_factor[:, np.newaxis]

            stressed[start:] = stress_slice
        else:
            # Single-asset case
            stress_slice = stressed[start:]

            # 1. Scale volatility
            stress_slice *= scenario.vol_multiplier

            # 2. Add return shock
            daily_shock = scenario.return_shock / stress_days
            stress_slice += daily_shock

            stressed[start:] = stress_slice

        return stressed

    def evaluate_under_stress(
        self,
        returns: np.ndarray,
        weights: Optional[np.ndarray],
        scenario: StressScenario,
    ) -> StressResult:
        """Evaluate portfolio performance under a stress scenario.

        Args:
            returns: Daily returns array (n_days,) or (n_days, n_assets).
            weights: Portfolio weights (n_assets,). Required for multi-asset.
            scenario: The stress scenario.

        Returns:
            StressResult with performance metrics under stress.
        """
        stressed_returns = self.simulate_scenario(returns, scenario)

        # Compute portfolio returns
        if stressed_returns.ndim == 2 and weights is not None:
            port_returns = stressed_returns @ weights
        else:
            port_returns = stressed_returns.flatten()

        # Sharpe ratio (annualized)
        mean_ret = np.nanmean(port_returns)
        std_ret = np.nanstd(port_returns)
        sharpe = (mean_ret * 252) / max(std_ret * np.sqrt(252), 1e-8)

        # Max drawdown
        cum_returns = np.cumprod(1 + port_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / np.maximum(peak, 1e-8)
        max_dd = float(np.min(drawdown))

        # Tail loss (5th percentile)
        tail_p5 = float(np.percentile(port_returns, 5))

        # Survival check
        survival = abs(max_dd) < self.max_drawdown_limit

        return StressResult(
            scenario_name=scenario.name,
            stress_sharpe=float(sharpe),
            stress_max_drawdown=float(max_dd),
            tail_loss_p5=tail_p5,
            survival=survival,
        )

    def run_all(
        self,
        returns: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, StressResult]:
        """Run all stress scenarios.

        Args:
            returns: Daily returns array.
            weights: Portfolio weights (for multi-asset).

        Returns:
            Dict of scenario_key -> StressResult.
        """
        results = {}
        for key, scenario in self.scenarios.items():
            try:
                result = self.evaluate_under_stress(returns, weights, scenario)
                results[key] = result
                logger.info("Stress test '%s': Sharpe=%.2f, MaxDD=%.2f, Survival=%s",
                           scenario.name, result.stress_sharpe,
                           result.stress_max_drawdown, result.survival)
            except Exception as e:
                logger.warning("Stress test '%s' failed: %s", key, e)
                results[key] = StressResult(
                    scenario_name=scenario.name,
                    stress_sharpe=0.0,
                    stress_max_drawdown=-1.0,
                    tail_loss_p5=-1.0,
                    survival=False,
                )
        return results
