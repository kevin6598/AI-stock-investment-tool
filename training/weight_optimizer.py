"""
Weight Optimizer
----------------
Learns optimal weighting ratios for combining prediction components:
  Final = w1*Sentiment + w2*Technical + w3*Fundamental + w4*Macro + w5*Risk

Implements:
  1. InverseICWeighter   - Static weights proportional to trailing IC
  2. RidgeMetaWeighter   - Ridge regression meta-model
  3. BayesianWeighter    - Bayesian model averaging
  4. SharpeOptimizer     - Direct Sharpe ratio maximization via scipy
  5. RegimeClassifier    - Market regime detection (rule-based + HMM-like)
  6. WeightSmoother      - Exponential smoothing with change caps
  7. DynamicWeightEngine - Full pipeline combining all components
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge
from dataclasses import dataclass


COMPONENT_NAMES = ["sentiment", "technical", "fundamental", "macro", "risk"]


# ---------------------------------------------------------------------------
# 1. Inverse-IC Weighter (Static Baseline)
# ---------------------------------------------------------------------------

class InverseICWeighter:
    """Weight each component proportional to its trailing Information Coefficient.

    The simplest robust approach: better-predicting components get more weight.
    """

    def __init__(self, lookback_periods: int = 252):
        self.lookback = lookback_periods
        self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

    def fit(self, component_predictions: Dict[str, np.ndarray],
            realized_returns: np.ndarray) -> np.ndarray:
        """Compute IC-based weights.

        Args:
            component_predictions: dict of component_name → prediction array.
            realized_returns: actual forward returns.

        Returns:
            Weight array [w1, ..., w5], normalized to sum to 1.
        """
        ics = []
        for name in COMPONENT_NAMES:
            preds = component_predictions.get(name)
            if preds is None or len(preds) == 0:
                ics.append(0.0)
                continue

            # Use only trailing lookback
            n = min(self.lookback, len(preds), len(realized_returns))
            p = preds[-n:]
            r = realized_returns[-n:]

            mask = ~(np.isnan(p) | np.isnan(r))
            if mask.sum() < 10:
                ics.append(0.0)
                continue

            ic, _ = sp_stats.spearmanr(p[mask], r[mask])
            ics.append(max(ic, 0.0))  # floor at 0 — don't reward negative IC

        ics = np.array(ics)
        total = ics.sum()
        if total > 0:
            self.weights = ics / total
        else:
            self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

        return self.weights

    def get_weights(self) -> Dict[str, float]:
        return {n: float(w) for n, w in zip(COMPONENT_NAMES, self.weights)}


# ---------------------------------------------------------------------------
# 2. Ridge Meta-Model Weighter
# ---------------------------------------------------------------------------

class RidgeMetaWeighter:
    """Learn weights via Ridge regression on component predictions.

    The L2 penalty prevents any single component from dominating,
    especially important with small training samples.
    """

    def __init__(self, alpha: float = 1.0, lookback_periods: int = 504):
        self.alpha = alpha
        self.lookback = lookback_periods
        self.model = Ridge(alpha=alpha, fit_intercept=False)
        self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

    def fit(self, component_predictions: Dict[str, np.ndarray],
            realized_returns: np.ndarray,
            sample_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit Ridge meta-model.

        Args:
            component_predictions: dict of component_name → prediction array.
            realized_returns: actual forward returns.
            sample_weights: optional time-decay weights.

        Returns:
            Normalized weight array.
        """
        n = min(self.lookback, len(realized_returns))

        # Build feature matrix from component predictions
        X_cols = []
        for name in COMPONENT_NAMES:
            preds = component_predictions.get(name)
            if preds is not None:
                X_cols.append(preds[-n:])
            else:
                X_cols.append(np.zeros(n))

        X = np.column_stack(X_cols)
        y = realized_returns[-n:]

        # Handle NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < 20:
            return self.weights

        X_clean = X[mask]
        y_clean = y[mask]
        sw = sample_weights[-n:][mask] if sample_weights is not None else None

        self.model.fit(X_clean, y_clean, sample_weight=sw)

        # Normalize coefficients to weights
        coefs = self.model.coef_
        # Project to non-negative space (Approach: clip then normalize)
        coefs = np.maximum(coefs, 0.0)
        total = coefs.sum()
        if total > 0:
            self.weights = coefs / total
        else:
            self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

        return self.weights

    def get_weights(self) -> Dict[str, float]:
        return {n: float(w) for n, w in zip(COMPONENT_NAMES, self.weights)}


# ---------------------------------------------------------------------------
# 3. Bayesian Model Averaging
# ---------------------------------------------------------------------------

class BayesianWeighter:
    """Bayesian model averaging using predictive likelihood.

    Each component is treated as a separate 'model'. Weights are posterior
    model probabilities computed from rolling prediction accuracy.
    """

    def __init__(self, lookback_periods: int = 126):
        self.lookback = lookback_periods
        self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

    def fit(self, component_predictions: Dict[str, np.ndarray],
            realized_returns: np.ndarray) -> np.ndarray:
        """Compute Bayesian weights via predictive log-likelihood."""
        n = min(self.lookback, len(realized_returns))
        log_liks = []

        for name in COMPONENT_NAMES:
            preds = component_predictions.get(name)
            if preds is None:
                log_liks.append(-1e10)
                continue

            p = preds[-n:]
            r = realized_returns[-n:]
            mask = ~(np.isnan(p) | np.isnan(r))

            if mask.sum() < 10:
                log_liks.append(-1e10)
                continue

            residuals = r[mask] - p[mask]
            sigma = max(np.std(residuals), 1e-8)

            # Gaussian log-likelihood
            ll = -0.5 * np.sum((residuals / sigma) ** 2) - mask.sum() * np.log(sigma)
            log_liks.append(ll)

        # Softmax to convert to weights
        log_liks = np.array(log_liks, dtype=np.float64)
        log_liks -= np.max(log_liks)  # numerical stability
        probs = np.exp(log_liks)
        total = probs.sum()
        if total > 0:
            self.weights = probs / total
        else:
            self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

        return self.weights

    def get_weights(self) -> Dict[str, float]:
        return {n: float(w) for n, w in zip(COMPONENT_NAMES, self.weights)}


# ---------------------------------------------------------------------------
# 4. Sharpe Ratio Optimizer
# ---------------------------------------------------------------------------

class SharpeOptimizer:
    """Direct optimization of portfolio Sharpe ratio.

    Uses scipy.optimize.minimize with SLSQP to maximize:
      Sharpe = E(Rp) / σ(Rp)
    where Rp = Σ wᵢ × componentᵢ predictions → ranked portfolio returns.

    Constraints: Σwᵢ = 1, wᵢ ≥ 0.
    """

    def __init__(self, allow_short_bias: bool = False):
        self.allow_short = allow_short_bias
        self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

    def fit(self, component_predictions: Dict[str, np.ndarray],
            realized_returns: np.ndarray) -> np.ndarray:
        """Optimize weights to maximize Sharpe ratio.

        The objective constructs a combined signal, ranks it, and computes
        the long-short return series.
        """
        n = len(realized_returns)
        X_cols = []
        for name in COMPONENT_NAMES:
            preds = component_predictions.get(name)
            if preds is not None:
                X_cols.append(preds[-n:])
            else:
                X_cols.append(np.zeros(n))

        X = np.column_stack(X_cols)
        y = realized_returns[-n:]

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(y_clean) < 30:
            return self.weights

        def neg_sharpe(w):
            # Combined signal
            combined = X_clean @ w

            # Rank-based portfolio: long top 20%, short bottom 20%
            ranks = sp_stats.rankdata(combined)
            n_obs = len(ranks)
            q_size = max(n_obs // 5, 1)

            long_mask = ranks > (n_obs - q_size)
            short_mask = ranks <= q_size

            long_ret = np.mean(y_clean[long_mask]) if long_mask.sum() > 0 else 0
            short_ret = np.mean(y_clean[short_mask]) if short_mask.sum() > 0 else 0

            port_ret = long_ret - short_ret
            port_vol = max(np.std(y_clean), 1e-8)

            sharpe = port_ret / port_vol
            return -sharpe  # minimize negative Sharpe

        # Constraints and bounds
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        lb = -0.2 if self.allow_short else 0.0
        bounds = [(lb, 1.0)] * len(COMPONENT_NAMES)

        # Multiple random starting points (avoid local minima)
        best_result = None
        best_val = float("inf")

        for seed in range(5):
            rng = np.random.RandomState(seed)
            x0 = rng.dirichlet(np.ones(len(COMPONENT_NAMES)))

            try:
                result = minimize(
                    neg_sharpe, x0=x0, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-8},
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None and best_result.success:
            w = best_result.x
            w = np.maximum(w, 0.0)
            total = w.sum()
            self.weights = w / total if total > 0 else np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

        return self.weights

    def get_weights(self) -> Dict[str, float]:
        return {n: float(w) for n, w in zip(COMPONENT_NAMES, self.weights)}


# ---------------------------------------------------------------------------
# 5. Regime Classifier
# ---------------------------------------------------------------------------

@dataclass
class RegimeInfo:
    regime: str                 # "strong_bull", "normal", "bear", "crisis"
    confidence: float           # 0-1
    market_return_21d: float
    market_vol_21d: float
    sma_200_ratio: float


class RegimeClassifier:
    """Market regime detection using rule-based thresholds.

    Regimes:
      - crisis: volatility > crisis_vol_threshold
      - strong_bull: price > SMA200 * (1 + bull_threshold)
      - bear: price < SMA200 * (1 - bear_threshold)
      - normal: everything else
    """

    def __init__(
        self,
        crisis_vol_threshold: float = 0.30,
        bull_threshold: float = 0.05,
        bear_threshold: float = 0.05,
    ):
        self.crisis_vol = crisis_vol_threshold
        self.bull_thresh = bull_threshold
        self.bear_thresh = bear_threshold

    def classify(self, market_df: pd.DataFrame) -> RegimeInfo:
        """Classify current market regime from market index data.

        Args:
            market_df: Market index OHLCV DataFrame.

        Returns:
            RegimeInfo with regime classification and supporting data.
        """
        c = market_df["Close"]

        # Current indicators
        sma_200 = c.rolling(200).mean().iloc[-1]
        current = c.iloc[-1]
        sma_ratio = current / sma_200 - 1.0 if sma_200 > 0 else 0.0

        log_ret = np.log(c / c.shift(1)).dropna()
        vol_21d = float(log_ret.tail(21).std() * np.sqrt(252))
        ret_21d = float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else 0.0

        # Classification
        if vol_21d > self.crisis_vol:
            regime = "crisis"
            confidence = min(vol_21d / self.crisis_vol - 1.0, 1.0) * 0.5 + 0.5
        elif sma_ratio > self.bull_thresh:
            regime = "strong_bull"
            confidence = min(sma_ratio / self.bull_thresh, 2.0) * 0.5
        elif sma_ratio < -self.bear_thresh:
            regime = "bear"
            confidence = min(abs(sma_ratio) / self.bear_thresh, 2.0) * 0.5
        else:
            regime = "normal"
            confidence = 1.0 - abs(sma_ratio) / max(self.bull_thresh, 0.01)

        return RegimeInfo(
            regime=regime,
            confidence=max(0.0, min(1.0, confidence)),
            market_return_21d=ret_21d,
            market_vol_21d=vol_21d,
            sma_200_ratio=sma_ratio,
        )


# Suggested regime-specific weight priors
REGIME_WEIGHT_PRIORS = {
    "strong_bull": {"sentiment": 0.20, "technical": 0.30, "fundamental": 0.20, "macro": 0.15, "risk": 0.15},
    "normal":      {"sentiment": 0.15, "technical": 0.20, "fundamental": 0.30, "macro": 0.15, "risk": 0.20},
    "bear":        {"sentiment": 0.10, "technical": 0.15, "fundamental": 0.25, "macro": 0.20, "risk": 0.30},
    "crisis":      {"sentiment": 0.05, "technical": 0.10, "fundamental": 0.15, "macro": 0.25, "risk": 0.45},
}


# ---------------------------------------------------------------------------
# 6. Weight Smoother
# ---------------------------------------------------------------------------

class WeightSmoother:
    """Exponential smoothing with per-component change caps.

    Prevents drastic weight shifts that may be optimization artifacts.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        max_change_per_period: float = 0.10,
    ):
        """
        Args:
            smoothing_alpha: exponential smoothing factor (0 = no change, 1 = full update).
            max_change_per_period: maximum absolute weight change per component per rebalance.
        """
        self.alpha = smoothing_alpha
        self.max_change = max_change_per_period
        self.previous_weights = None

    def smooth(self, new_weights: np.ndarray) -> np.ndarray:
        """Apply smoothing and change caps to new weights."""
        if self.previous_weights is None:
            self.previous_weights = new_weights.copy()
            return new_weights

        # Exponential smoothing
        smoothed = self.alpha * new_weights + (1 - self.alpha) * self.previous_weights

        # Apply change caps
        delta = smoothed - self.previous_weights
        capped_delta = np.clip(delta, -self.max_change, self.max_change)
        result = self.previous_weights + capped_delta

        # Re-normalize
        result = np.maximum(result, 0.0)
        total = result.sum()
        if total > 0:
            result = result / total

        self.previous_weights = result.copy()
        return result


# ---------------------------------------------------------------------------
# 7. Time-Decay Sample Weights
# ---------------------------------------------------------------------------

def compute_time_decay_weights(n_samples: int, half_life: int = 126) -> np.ndarray:
    """Compute exponential time-decay weights.

    Args:
        n_samples: number of observations.
        half_life: half-life in trading days (default 126 ≈ 6 months).

    Returns:
        Array of sample weights (most recent = 1.0, decaying backward).
    """
    decay_lambda = np.log(2) / half_life
    t = np.arange(n_samples)[::-1]  # most recent = 0
    weights = np.exp(-decay_lambda * t)
    return weights


# ---------------------------------------------------------------------------
# 8. Dynamic Weight Engine (Full Pipeline)
# ---------------------------------------------------------------------------

class DynamicWeightEngine:
    """Full weight optimization pipeline.

    Combines multiple weighting methods with regime awareness,
    time decay, and weight smoothing.

    Usage:
        engine = DynamicWeightEngine()
        weights = engine.optimize(component_preds, returns, market_df)
    """

    def __init__(
        self,
        primary_method: str = "inverse_ic",
        secondary_method: str = "ridge",
        use_regime: bool = True,
        regime_blend_alpha: float = 0.3,
        smoothing_alpha: float = 0.3,
        max_weight_change: float = 0.10,
        time_decay_half_life: int = 126,
        turnover_penalty: float = 0.0,
        volatility_scaling: bool = False,
    ):
        """
        Args:
            primary_method: "inverse_ic", "ridge", "bayesian", "sharpe",
                            "performance_decay", or "volatility_aware".
            secondary_method: fallback method if primary fails.
            use_regime: whether to blend regime priors into weights.
            regime_blend_alpha: how much to blend regime priors (0 = ignore, 1 = all regime).
            smoothing_alpha: exponential smoothing factor.
            max_weight_change: maximum weight change per rebalance.
            time_decay_half_life: half-life for sample weighting.
            turnover_penalty: penalty for weight changes (0 = no penalty).
            volatility_scaling: apply inverse-volatility scaling to final weights.
        """
        self.primary_method = primary_method
        self.secondary_method = secondary_method
        self.use_regime = use_regime
        self.regime_blend = regime_blend_alpha
        self.turnover_penalty = turnover_penalty
        self.volatility_scaling = volatility_scaling

        # Instantiate all weighters
        self.weighters = {
            "inverse_ic": InverseICWeighter(),
            "ridge": RidgeMetaWeighter(),
            "bayesian": BayesianWeighter(),
            "sharpe": SharpeOptimizer(),
            "performance_decay": PerformanceDecayWeighter(),
            "volatility_aware": VolatilityAwareWeighter(),
        }

        self.regime_classifier = RegimeClassifier()
        self.smoother = WeightSmoother(smoothing_alpha, max_weight_change)
        self.half_life = time_decay_half_life

        self.current_regime = None
        self.weight_history = []

    def optimize(
        self,
        component_predictions: Dict[str, np.ndarray],
        realized_returns: np.ndarray,
        market_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Run full weight optimization pipeline.

        Args:
            component_predictions: dict of component_name → prediction array.
            realized_returns: realized forward returns.
            market_df: market index OHLCV for regime detection.

        Returns:
            Dict of component_name → optimized weight.
        """
        # Step 1: Classify regime
        if self.use_regime and market_df is not None:
            self.current_regime = self.regime_classifier.classify(market_df)

        # Step 2: Compute data-driven weights
        try:
            weighter = self.weighters[self.primary_method]
            if self.primary_method == "ridge":
                decay = compute_time_decay_weights(len(realized_returns), self.half_life)
                weights = weighter.fit(component_predictions, realized_returns,
                                       sample_weights=decay)
            else:
                weights = weighter.fit(component_predictions, realized_returns)
        except Exception:
            weighter = self.weighters[self.secondary_method]
            weights = weighter.fit(component_predictions, realized_returns)

        # Step 3: Blend with regime priors
        if self.use_regime and self.current_regime is not None:
            regime_name = self.current_regime.regime
            if regime_name in REGIME_WEIGHT_PRIORS:
                prior = np.array([
                    REGIME_WEIGHT_PRIORS[regime_name][n] for n in COMPONENT_NAMES
                ])
                # Blend: more regime influence when regime confidence is high
                blend_factor = self.regime_blend * self.current_regime.confidence
                weights = (1 - blend_factor) * weights + blend_factor * prior

        # Step 4: Smooth
        weights = self.smoother.smooth(weights)

        # Step 4.5: Turnover penalty
        if self.turnover_penalty > 0 and self.weight_history:
            prev_weights = np.array([
                self.weight_history[-1]["weights"].get(n, 0.0)
                for n in COMPONENT_NAMES
            ])
            # Penalize deviation from previous weights
            penalty = self.turnover_penalty
            weights = (1.0 - penalty) * weights + penalty * prev_weights

        # Step 4.6: Volatility scaling
        if self.volatility_scaling:
            vol_weighter = VolatilityAwareWeighter()
            vol_weights = vol_weighter.fit(component_predictions, realized_returns)
            # Blend: use volatility scaling as a multiplier
            weights = weights * vol_weights
            vol_total = weights.sum()
            if vol_total > 0:
                weights = weights / vol_total

        # Step 5: Final normalization
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total > 0:
            weights = weights / total

        # Record history
        result = {n: float(w) for n, w in zip(COMPONENT_NAMES, weights)}
        self.weight_history.append({
            "weights": result.copy(),
            "regime": self.current_regime.regime if self.current_regime else "unknown",
            "method": self.primary_method,
        })

        return result

    def get_weight_history(self) -> List[Dict]:
        return self.weight_history

    def get_current_regime(self) -> Optional[RegimeInfo]:
        return self.current_regime


# ---------------------------------------------------------------------------
# 9. Performance Decay Weighter
# ---------------------------------------------------------------------------

class PerformanceDecayWeighter:
    """Weights decay based on time since last good performance.

    Formula: w_i = base_w_i * exp(-lambda * days_since_peak_ic_i)

    Components that had strong recent IC keep their weight;
    those with declining performance are gradually down-weighted.
    """

    def __init__(
        self,
        decay_lambda: float = 0.01,
        ic_peak_window: int = 252,
    ):
        """
        Args:
            decay_lambda: exponential decay rate.
            ic_peak_window: lookback window for computing peak IC.
        """
        self.decay_lambda = decay_lambda
        self.ic_peak_window = ic_peak_window
        self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

    def fit(
        self,
        component_predictions: Dict[str, np.ndarray],
        realized_returns: np.ndarray,
        days_since_peak_ic: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """Compute decay-adjusted weights.

        Args:
            component_predictions: dict of component_name -> prediction array.
            realized_returns: actual forward returns.
            days_since_peak_ic: optional precomputed days since peak IC per component.
                If None, computed from the data.

        Returns:
            Weight array normalized to sum to 1.
        """
        n = min(self.ic_peak_window, len(realized_returns))

        if days_since_peak_ic is None:
            days_since_peak_ic = {}
            for name in COMPONENT_NAMES:
                preds = component_predictions.get(name)
                if preds is None or len(preds) < 20:
                    days_since_peak_ic[name] = n
                    continue

                # Compute rolling IC and find peak
                window = min(63, n)
                best_ic = -1.0
                best_pos = n
                for i in range(window, n):
                    p = preds[-(n - i):-(n - i) + window] if (n - i) + window <= len(preds) else None
                    r = realized_returns[i - window:i]
                    if p is None or len(p) != len(r):
                        continue
                    mask = ~(np.isnan(p) | np.isnan(r))
                    if mask.sum() < 10:
                        continue
                    ic, _ = sp_stats.spearmanr(p[mask], r[mask])
                    if ic > best_ic:
                        best_ic = ic
                        best_pos = n - i

                days_since_peak_ic[name] = max(best_pos, 0)

        # Base weights from IC
        ic_weighter = InverseICWeighter(lookback_periods=n)
        base_weights = ic_weighter.fit(component_predictions, realized_returns)

        # Apply decay
        decay_factors = np.array([
            np.exp(-self.decay_lambda * days_since_peak_ic.get(name, n))
            for name in COMPONENT_NAMES
        ])

        adjusted = base_weights * decay_factors
        total = adjusted.sum()
        if total > 0:
            self.weights = adjusted / total
        else:
            self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

        return self.weights

    def get_weights(self) -> Dict[str, float]:
        return {n: float(w) for n, w in zip(COMPONENT_NAMES, self.weights)}


# ---------------------------------------------------------------------------
# 10. Volatility-Aware Weighter
# ---------------------------------------------------------------------------

class VolatilityAwareWeighter:
    """Reduce weight of components with high recent prediction volatility.

    Formula: w_i = base_w_i / vol_i, then normalize.

    A component whose predictions are erratic (high vol) gets down-weighted
    relative to one with stable predictions.
    """

    def __init__(self, vol_window: int = 63):
        """
        Args:
            vol_window: window for computing prediction volatility.
        """
        self.vol_window = vol_window
        self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

    def fit(
        self,
        component_predictions: Dict[str, np.ndarray],
        realized_returns: np.ndarray,
    ) -> np.ndarray:
        """Compute volatility-adjusted weights.

        Args:
            component_predictions: dict of component_name -> prediction array.
            realized_returns: actual forward returns.

        Returns:
            Weight array normalized to sum to 1.
        """
        # Base weights from IC
        ic_weighter = InverseICWeighter()
        base_weights = ic_weighter.fit(component_predictions, realized_returns)

        # Compute prediction volatility per component
        vols = []
        for name in COMPONENT_NAMES:
            preds = component_predictions.get(name)
            if preds is None or len(preds) < self.vol_window:
                vols.append(1.0)
                continue
            recent = preds[-self.vol_window:]
            recent = recent[~np.isnan(recent)]
            vol = np.std(recent) if len(recent) > 5 else 1.0
            vols.append(max(vol, 1e-8))

        vols = np.array(vols)

        # Inverse vol weighting
        inv_vol = 1.0 / vols
        adjusted = base_weights * inv_vol
        total = adjusted.sum()
        if total > 0:
            self.weights = adjusted / total
        else:
            self.weights = np.ones(len(COMPONENT_NAMES)) / len(COMPONENT_NAMES)

        return self.weights

    def get_weights(self) -> Dict[str, float]:
        return {n: float(w) for n, w in zip(COMPONENT_NAMES, self.weights)}
