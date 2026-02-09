"""
CVaR Portfolio Optimizer
-------------------------
Conditional Value-at-Risk (CVaR) based portfolio optimization.

CVaR measures the expected loss in the worst (1-alpha) tail of the
distribution, making it a coherent risk measure (unlike VaR).

Components:
  - PortfolioConstraints: investment policy limits
  - PortfolioResult: optimized portfolio output
  - CVaROptimizer: Monte Carlo CVaR minimization with SLSQP
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize


@dataclass
class PortfolioConstraints:
    """Investment policy constraints for portfolio construction."""
    max_weight: float = 0.30
    min_weight: float = 0.0
    max_sector_weight: float = 0.40
    target_volatility: Optional[float] = None
    max_leverage: float = 1.0


@dataclass
class PortfolioResult:
    """Optimized portfolio output."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    cvar_95: float
    cvar_99: float
    sharpe: float


class CVaROptimizer:
    """CVaR-minimizing portfolio optimizer.

    Uses Monte Carlo simulation to estimate the return distribution,
    then minimizes CVaR (expected shortfall) subject to constraints.
    Multi-start optimization avoids local minima.
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def simulate_returns(
        self,
        means: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """Generate Monte Carlo return scenarios.

        Args:
            means: expected returns per asset (n_assets,).
            covariance: covariance matrix (n_assets, n_assets).

        Returns:
            Simulated returns (n_simulations, n_assets).
        """
        rng = np.random.RandomState(self.random_seed)
        return rng.multivariate_normal(means, covariance, size=self.n_simulations)

    @staticmethod
    def compute_cvar(
        portfolio_returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Compute Conditional Value-at-Risk.

        CVaR = mean of worst (1-alpha) tail returns.

        Args:
            portfolio_returns: array of portfolio return scenarios.
            confidence: confidence level (e.g. 0.95 for CVaR_95).

        Returns:
            CVaR value (negative number = loss).
        """
        sorted_returns = np.sort(portfolio_returns)
        cutoff = int(len(sorted_returns) * (1 - confidence))
        cutoff = max(cutoff, 1)
        return float(np.mean(sorted_returns[:cutoff]))

    def _build_covariance_from_predictions(
        self,
        variances: np.ndarray,
        correlation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Combine model-predicted variances with historical correlations.

        Args:
            variances: per-asset variance estimates (n_assets,).
            correlation_matrix: historical correlation matrix (n_assets, n_assets).

        Returns:
            Covariance matrix (n_assets, n_assets).
        """
        stds = np.sqrt(np.maximum(variances, 1e-10))
        D = np.diag(stds)
        return D @ correlation_matrix @ D

    def optimize(
        self,
        tickers: List[str],
        means: np.ndarray,
        covariance: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> PortfolioResult:
        """Optimize portfolio weights to minimize CVaR.

        Args:
            tickers: list of ticker symbols.
            means: expected returns per asset.
            covariance: covariance matrix.
            constraints: portfolio constraints.
            sector_map: ticker -> sector mapping for sector caps.

        Returns:
            PortfolioResult with optimal weights and risk metrics.
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        n = len(tickers)
        simulated = self.simulate_returns(means, covariance)

        # Build scipy constraints
        scipy_constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        # Sector constraints
        if sector_map is not None:
            sectors = {}  # type: Dict[str, List[int]]
            for i, t in enumerate(tickers):
                sec = sector_map.get(t, "Other")
                sectors.setdefault(sec, []).append(i)
            for sec, indices in sectors.items():
                scipy_constraints.append({
                    "type": "ineq",
                    "fun": lambda w, idx=indices: constraints.max_sector_weight - sum(w[j] for j in idx),
                })

        # Volatility target constraint
        if constraints.target_volatility is not None:
            target_vol = constraints.target_volatility
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda w: target_vol - np.sqrt(w @ covariance @ w) + 0.01,
            })

        bounds = [(constraints.min_weight, constraints.max_weight)] * n

        def neg_cvar_objective(w):
            port_returns = simulated @ w
            cvar = self.compute_cvar(port_returns, 0.95)
            return -cvar  # minimize negative CVaR = maximize CVaR (least negative)

        # Multi-start optimization (5 random starts)
        best_result = None
        best_val = float("inf")

        for seed in range(5):
            rng = np.random.RandomState(seed + self.random_seed)
            x0 = rng.dirichlet(np.ones(n))
            # Clip to bounds
            x0 = np.clip(x0, constraints.min_weight, constraints.max_weight)
            x0 = x0 / x0.sum()

            try:
                result = minimize(
                    neg_cvar_objective, x0=x0, method="SLSQP",
                    bounds=bounds, constraints=scipy_constraints,
                    options={"maxiter": 500, "ftol": 1e-8},
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None and best_result.success:
            w = best_result.x
        else:
            w = np.ones(n) / n  # equal weight fallback

        w = np.maximum(w, 0.0)
        total = w.sum()
        if total > 0:
            w = w / total

        # Compute portfolio metrics
        port_returns = simulated @ w
        exp_ret = float(means @ w)
        exp_vol = float(np.sqrt(w @ covariance @ w))
        cvar_95 = self.compute_cvar(port_returns, 0.95)
        cvar_99 = self.compute_cvar(port_returns, 0.99)
        sharpe = exp_ret / max(exp_vol, 1e-8)

        return PortfolioResult(
            weights={t: float(wi) for t, wi in zip(tickers, w)},
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe=sharpe,
        )


class HybridOptimizer:
    """HRP + CVaR hybrid portfolio optimizer.

    Blends HRP base allocation with CVaR-optimized weights for a portfolio
    that is both diversified (HRP) and tail-risk-aware (CVaR).

    Usage:
        hybrid = HybridOptimizer(hrp_blend=0.5)
        result = hybrid.optimize(tickers, means, covariance)
    """

    def __init__(
        self,
        hrp_blend: float = 0.5,
        max_turnover: Optional[float] = None,
        n_simulations: int = 10000,
        random_seed: int = 42,
    ):
        """
        Args:
            hrp_blend: blend factor (0 = pure CVaR, 1 = pure HRP).
            max_turnover: if set, penalize deviation from previous weights.
            n_simulations: Monte Carlo simulations for CVaR.
            random_seed: random seed.
        """
        self.hrp_blend = hrp_blend
        self.max_turnover = max_turnover
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self._previous_weights = None  # type: Optional[Dict[str, float]]

    def optimize(
        self,
        tickers: List[str],
        means: np.ndarray,
        covariance: np.ndarray,
        constraints: Optional[PortfolioConstraints] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> PortfolioResult:
        """Run HRP + CVaR hybrid optimization.

        Steps:
          1. Get HRP base weights
          2. Get CVaR weights
          3. Blend
          4. Apply constraints (volatility target, sector caps, turnover)

        Args:
            tickers: list of ticker symbols.
            means: expected returns per asset.
            covariance: covariance matrix.
            constraints: portfolio constraints.
            sector_map: ticker -> sector mapping.

        Returns:
            PortfolioResult with blended weights and risk metrics.
        """
        from engine.hrp_optimizer import HRPOptimizer

        if constraints is None:
            constraints = PortfolioConstraints()

        n = len(tickers)

        # Step 1: HRP weights
        # Derive correlation from covariance
        stds = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        correlation = covariance / np.outer(stds, stds)
        np.fill_diagonal(correlation, 1.0)
        # Clip to valid range
        correlation = np.clip(correlation, -1.0, 1.0)

        hrp = HRPOptimizer()
        hrp_weights = hrp.optimize(tickers, covariance, correlation)

        # Step 2: CVaR weights
        cvar = CVaROptimizer(
            n_simulations=self.n_simulations,
            random_seed=self.random_seed,
        )
        cvar_result = cvar.optimize(
            tickers, means, covariance, constraints, sector_map,
        )

        # Step 3: Blend
        blend = self.hrp_blend
        blended = {}
        for t in tickers:
            h_w = hrp_weights.get(t, 1.0 / n)
            c_w = cvar_result.weights.get(t, 1.0 / n)
            blended[t] = blend * h_w + (1.0 - blend) * c_w

        # Step 4: Apply turnover constraint
        if self.max_turnover is not None and self._previous_weights is not None:
            total_turnover = sum(
                abs(blended.get(t, 0.0) - self._previous_weights.get(t, 0.0))
                for t in tickers
            )
            if total_turnover > self.max_turnover:
                # Scale down changes to meet turnover budget
                scale = self.max_turnover / max(total_turnover, 1e-10)
                for t in tickers:
                    prev = self._previous_weights.get(t, 0.0)
                    delta = blended[t] - prev
                    blended[t] = prev + delta * scale

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {t: v / total for t, v in blended.items()}

        # Apply sector caps
        if sector_map is not None:
            sector_totals = {}  # type: Dict[str, float]
            for t, w in blended.items():
                sec = sector_map.get(t, "Other")
                sector_totals[sec] = sector_totals.get(sec, 0.0) + w
            for sec, sw in sector_totals.items():
                if sw > constraints.max_sector_weight:
                    scale_factor = constraints.max_sector_weight / sw
                    for t in blended:
                        if sector_map.get(t, "Other") == sec:
                            blended[t] *= scale_factor
            total = sum(blended.values())
            if total > 0:
                blended = {t: v / total for t, v in blended.items()}

        # Apply volatility target
        if constraints.target_volatility is not None:
            w_arr = np.array([blended.get(t, 0.0) for t in tickers])
            port_vol = np.sqrt(max(w_arr @ covariance @ w_arr, 1e-12))
            if port_vol > constraints.target_volatility:
                vol_scale = constraints.target_volatility / port_vol
                blended = {t: v * vol_scale for t, v in blended.items()}
                total = sum(blended.values())
                if total > 0:
                    blended = {t: v / total for t, v in blended.items()}

        # Save for next turnover check
        self._previous_weights = blended.copy()

        # Compute portfolio metrics
        w_arr = np.array([blended.get(t, 0.0) for t in tickers])
        exp_ret = float(means @ w_arr)
        exp_vol = float(np.sqrt(max(w_arr @ covariance @ w_arr, 0.0)))

        rng = np.random.RandomState(self.random_seed)
        sim_returns = rng.multivariate_normal(means, covariance, size=self.n_simulations)
        port_returns = sim_returns @ w_arr
        cvar_95 = CVaROptimizer.compute_cvar(port_returns, 0.95)
        cvar_99 = CVaROptimizer.compute_cvar(port_returns, 0.99)
        sharpe = exp_ret / max(exp_vol, 1e-8)

        return PortfolioResult(
            weights=blended,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe=sharpe,
        )
