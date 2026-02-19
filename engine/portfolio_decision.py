"""
Portfolio Decision Engine
--------------------------
Multi-condition filtering and risk-aware allocation for stock selection.

Combines model predictions with risk metrics, liquidity constraints,
and multiple allocation strategies (equal weight, risk parity, CVaR).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging

from engine.liquidity import LiquidityConfig, LiquidityFilter
from engine.early_warning import EarlyWarningEngine, EarlyWarningReport

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Thresholds for multi-condition stock filtering."""
    min_p_up: float = 0.55
    min_expected_return: float = 0.005
    min_q10: float = -0.05          # tail risk floor
    min_confidence: float = 0.3
    max_risk_score: float = 0.8
    min_meta_trade_prob: float = 0.4
    min_conditions: int = 4          # of 6 conditions required to pass


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    allocation_mode: str = "risk_parity"  # "equal", "risk_parity", "cvar"
    max_positions: int = 20
    max_single_weight: float = 0.10
    max_sector_weight: float = 0.30
    rebalance_frequency_days: int = 21
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    liquidity_config: LiquidityConfig = field(default_factory=LiquidityConfig)


@dataclass
class PortfolioDecision:
    """Output of the portfolio decision engine."""
    selected_stocks: List[str]
    weights: Dict[str, float]
    expected_portfolio_return: float
    expected_vol: float
    expected_sharpe: float
    rejected: List[Dict[str, Any]]     # [{ticker, reasons}, ...]
    filter_pass_rate: float
    allocation_mode: str
    early_warning: Optional[EarlyWarningReport] = None
    exposure_multiplier: float = 1.0


class PortfolioDecisionEngine:
    """Core decision engine for stock selection and allocation.

    Pipeline:
      1. Filter candidates using multi-condition scoring
      2. Rank by composite score
      3. Allocate weights using selected strategy
      4. Apply position limits and normalization

    Usage:
        engine = PortfolioDecisionEngine()
        decision = engine.run(predictions, returns_df)
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()

    def filter_candidates(
        self,
        predictions: List[Any],
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Filter predictions using multi-condition scoring.

        For each prediction, counts how many of 6 conditions are met.
        Accepts if count >= min_conditions.

        Conditions:
          1. p_up >= min_p_up
          2. point_estimate >= min_expected_return
          3. q10 >= min_q10 (tail risk floor)
          4. confidence >= min_confidence
          5. risk_score <= max_risk_score (inverted: low risk is good)
          6. meta_trade_probability >= min_meta_trade_prob

        Args:
            predictions: List of FullInferenceResult-like objects.

        Returns:
            (passed, rejected) tuple.
        """
        fc = self.config.filter_config
        passed = []
        rejected = []

        for pred in predictions:
            conditions_met = 0
            reasons = []

            # Extract values (support both object and dict)
            p_up = _get_field(pred, "probability_up", 0.5)
            point = _get_field(pred, "point_estimate", 0.0)
            quantiles = _get_field(pred, "quantiles", {})
            q10 = quantiles.get("q10", quantiles.get("p10", -999.0))
            confidence = _get_field(pred, "confidence", 0.0)
            # Compute risk_score from uncertainty if not directly available
            uncertainty = _get_field(pred, "uncertainty", 0.5)
            risk_score = uncertainty  # higher uncertainty = higher risk
            meta_prob = _get_field(pred, "meta_trade_probability", 0.0)
            ticker = _get_field(pred, "ticker", "UNKNOWN")

            # Condition 1: Direction probability
            if p_up >= fc.min_p_up:
                conditions_met += 1
            else:
                reasons.append("p_up=%.3f < %.3f" % (p_up, fc.min_p_up))

            # Condition 2: Expected return
            if point >= fc.min_expected_return:
                conditions_met += 1
            else:
                reasons.append("return=%.4f < %.4f" % (point, fc.min_expected_return))

            # Condition 3: Tail risk
            if q10 >= fc.min_q10:
                conditions_met += 1
            else:
                reasons.append("q10=%.4f < %.4f" % (q10, fc.min_q10))

            # Condition 4: Confidence
            if confidence >= fc.min_confidence:
                conditions_met += 1
            else:
                reasons.append("conf=%.3f < %.3f" % (confidence, fc.min_confidence))

            # Condition 5: Risk score (inverted - lower is better)
            if risk_score <= fc.max_risk_score:
                conditions_met += 1
            else:
                reasons.append("risk=%.3f > %.3f" % (risk_score, fc.max_risk_score))

            # Condition 6: Meta trade probability
            if meta_prob >= fc.min_meta_trade_prob:
                conditions_met += 1
            else:
                reasons.append("meta=%.3f < %.3f" % (meta_prob, fc.min_meta_trade_prob))

            if conditions_met >= fc.min_conditions:
                passed.append(pred)
            else:
                rejected.append({
                    "ticker": ticker,
                    "conditions_met": conditions_met,
                    "reasons": reasons,
                })

        total = len(predictions)
        pass_rate = len(passed) / total if total > 0 else 0.0
        logger.info("Filter: %d/%d passed (%.1f%%), min_conditions=%d",
                     len(passed), total, pass_rate * 100, fc.min_conditions)

        return passed, rejected

    def rank_candidates(
        self,
        candidates: List[Any],
    ) -> List[Any]:
        """Rank candidates by composite attractiveness score.

        Score = 0.4 * p_up + 0.3 * normalized_return + 0.2 * confidence + 0.1 * meta_prob

        Args:
            candidates: List of passed predictions.

        Returns:
            Sorted list (best first), limited to max_positions.
        """
        scored = []
        for pred in candidates:
            p_up = _get_field(pred, "probability_up", 0.5)
            point = _get_field(pred, "point_estimate", 0.0)
            confidence = _get_field(pred, "confidence", 0.0)
            meta_prob = _get_field(pred, "meta_trade_probability", 0.0)

            # Normalize return to [0, 1] range using sigmoid
            norm_return = 1.0 / (1.0 + np.exp(-point * 100))

            score = (
                0.4 * p_up +
                0.3 * norm_return +
                0.2 * confidence +
                0.1 * meta_prob
            )
            scored.append((score, pred))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [pred for _, pred in scored[:self.config.max_positions]]
        return ranked

    def rank_candidates_v2(
        self,
        candidates: List[Any],
    ) -> List[Any]:
        """Enhanced ranking with risk and sentiment integration.

        Score = 0.30 * p_up
             + 0.25 * sigmoid(expected_return * 100)
             + 0.15 * confidence
             + 0.10 * meta_trade_probability
             + 0.10 * (1 - risk_score)
             + 0.10 * sentiment_score

        Args:
            candidates: List of passed predictions.

        Returns:
            Sorted list (best first), limited to max_positions.
        """
        scored = []
        for pred in candidates:
            p_up = _get_field(pred, "probability_up", 0.5)
            point = _get_field(pred, "point_estimate", 0.0)
            confidence = _get_field(pred, "confidence", 0.0)
            meta_prob = _get_field(pred, "meta_trade_probability", 0.0) or 0.0
            uncertainty = _get_field(pred, "uncertainty", 0.5) or 0.5
            risk_score = _get_field(pred, "risk_score", uncertainty)

            # Get sentiment score
            sentiment_score = 0.0
            ticker = _get_field(pred, "ticker", "")
            if ticker:
                try:
                    from models.sentiment import get_extended_sentiment_features
                    feat = get_extended_sentiment_features(ticker)
                    sentiment_score = feat.get("sentiment_weighted", 0.0)
                except Exception:
                    pass

            # Normalize return to [0, 1] using sigmoid
            norm_return = 1.0 / (1.0 + np.exp(-point * 100))
            # Normalize sentiment to [0, 1]
            norm_sentiment = max(0.0, min(1.0, (sentiment_score + 1.0) / 2.0))

            score = (
                0.30 * p_up
                + 0.25 * norm_return
                + 0.15 * confidence
                + 0.10 * meta_prob
                + 0.10 * (1.0 - risk_score)
                + 0.10 * norm_sentiment
            )
            scored.append((score, pred))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [pred for _, pred in scored[:self.config.max_positions]]
        return ranked

    def allocate(
        self,
        candidates: List[Any],
        returns_df: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Allocate portfolio weights using selected strategy.

        Args:
            candidates: Ranked list of predictions to allocate.
            returns_df: Historical returns DataFrame (tickers as columns).
                Required for risk_parity and cvar modes.

        Returns:
            Dict of ticker -> weight (sums to 1.0).
        """
        if not candidates:
            return {}

        tickers = [_get_field(p, "ticker", "UNKNOWN") for p in candidates]
        n = len(tickers)

        mode = self.config.allocation_mode

        if mode == "equal":
            raw_weights = {t: 1.0 / n for t in tickers}

        elif mode == "risk_parity":
            raw_weights = self._risk_parity_weights(tickers, returns_df)

        elif mode == "cvar":
            raw_weights = self._cvar_weights(tickers, candidates, returns_df)

        else:
            logger.warning("Unknown allocation mode '%s', using equal weight", mode)
            raw_weights = {t: 1.0 / n for t in tickers}

        # Apply position cap
        capped = self._apply_caps(raw_weights)
        return capped

    def _risk_parity_weights(
        self,
        tickers: List[str],
        returns_df: Optional[Any],
    ) -> Dict[str, float]:
        """Inverse-volatility weighting (risk parity approximation).

        Uses HRP approach from engine/hrp_optimizer.py if returns available,
        otherwise falls back to equal weight.
        """
        import pandas as pd

        n = len(tickers)
        if returns_df is None or not isinstance(returns_df, pd.DataFrame):
            return {t: 1.0 / n for t in tickers}

        # Compute per-asset volatility from returns
        available = [t for t in tickers if t in returns_df.columns]
        if len(available) < 2:
            return {t: 1.0 / n for t in tickers}

        vols = returns_df[available].std().replace(0, np.nan).dropna()
        if vols.empty:
            return {t: 1.0 / n for t in tickers}

        # Inverse-vol weights
        inv_vol = 1.0 / vols
        total = inv_vol.sum()
        if total <= 0:
            return {t: 1.0 / n for t in tickers}

        weights = (inv_vol / total).to_dict()

        # Fill missing tickers with equal share of remainder
        assigned = sum(weights.values())
        missing = [t for t in tickers if t not in weights]
        if missing:
            remaining = max(1.0 - assigned, 0.0)
            per_missing = remaining / len(missing) if remaining > 0 else 1.0 / n
            for t in missing:
                weights[t] = per_missing

        return weights

    def _cvar_weights(
        self,
        tickers: List[str],
        candidates: List[Any],
        returns_df: Optional[Any],
    ) -> Dict[str, float]:
        """Delegate to CVaROptimizer from engine/cvar_optimizer.py."""
        import pandas as pd

        n = len(tickers)
        if returns_df is None or not isinstance(returns_df, pd.DataFrame):
            return {t: 1.0 / n for t in tickers}

        available = [t for t in tickers if t in returns_df.columns]
        if len(available) < 2:
            return {t: 1.0 / n for t in tickers}

        try:
            from engine.cvar_optimizer import CVaROptimizer, PortfolioConstraints

            rets = returns_df[available].dropna()
            if len(rets) < 30:
                return {t: 1.0 / n for t in tickers}

            means = rets.mean().values
            cov = rets.cov().values

            constraints = PortfolioConstraints(
                max_weight=self.config.max_single_weight,
                max_sector_weight=self.config.max_sector_weight,
            )

            optimizer = CVaROptimizer(n_simulations=5000, random_seed=42)
            result = optimizer.optimize(available, means, cov, constraints)

            weights = result.weights

            # Fill missing tickers
            assigned = sum(weights.values())
            missing = [t for t in tickers if t not in weights]
            if missing:
                remaining = max(1.0 - assigned, 0.0)
                per_missing = remaining / len(missing) if remaining > 0 else 0.0
                for t in missing:
                    weights[t] = per_missing

            return weights

        except Exception as e:
            logger.warning("CVaR optimization failed, falling back to equal: %s", e)
            return {t: 1.0 / n for t in tickers}

    def _apply_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position cap and re-normalize."""
        cap = self.config.max_single_weight
        capped = {}
        for t, w in weights.items():
            capped[t] = min(w, cap)

        total = sum(capped.values())
        if total > 0:
            capped = {t: v / total for t, v in capped.items()}
        return capped

    def apply_early_warning_adjustment(
        self,
        weights: Dict[str, float],
        early_warning: Optional[EarlyWarningReport] = None,
    ) -> tuple:
        """Scale portfolio weights by early warning exposure multiplier.

        When the early warning system detects structural degradation,
        weights are uniformly scaled down. The freed capital is implicitly
        allocated to cash.

        Args:
            weights: current portfolio weights (sum to 1.0).
            early_warning: EarlyWarningReport from the engine. If None,
                weights are returned unchanged.

        Returns:
            (adjusted_weights, exposure_multiplier) tuple.
        """
        if early_warning is None:
            return weights, 1.0

        multiplier = early_warning.exposure_multiplier

        if multiplier >= 1.0:
            return weights, 1.0

        if multiplier <= 0.0:
            logger.warning(
                "Early warning CRITICAL (score=%.3f). Zeroing all positions.",
                early_warning.warning_score,
            )
            return {t: 0.0 for t in weights}, 0.0

        adjusted = {t: w * multiplier for t, w in weights.items()}
        logger.info(
            "Early warning %s (score=%.3f): scaling exposure to %.0f%%",
            early_warning.level,
            early_warning.warning_score,
            multiplier * 100,
        )
        return adjusted, multiplier

    def run(
        self,
        predictions: List[Any],
        returns_df: Optional[Any] = None,
        early_warning: Optional[EarlyWarningReport] = None,
    ) -> PortfolioDecision:
        """Execute the full portfolio decision pipeline.

        Args:
            predictions: List of FullInferenceResult-like objects (or dicts).
            returns_df: Historical returns DataFrame for allocation.
            early_warning: Optional EarlyWarningReport to adjust exposure.

        Returns:
            PortfolioDecision with selected stocks, weights, and metrics.
        """
        # Step 1: Filter
        passed, rejected = self.filter_candidates(predictions)

        if not passed:
            return PortfolioDecision(
                selected_stocks=[],
                weights={},
                expected_portfolio_return=0.0,
                expected_vol=0.0,
                expected_sharpe=0.0,
                rejected=rejected,
                filter_pass_rate=0.0,
                allocation_mode=self.config.allocation_mode,
                early_warning=early_warning,
                exposure_multiplier=early_warning.exposure_multiplier if early_warning else 1.0,
            )

        # Step 2: Rank
        ranked = self.rank_candidates(passed)

        # Step 3: Allocate
        weights = self.allocate(ranked, returns_df)

        # Step 4: Apply early warning exposure adjustment
        weights, exposure_mult = self.apply_early_warning_adjustment(
            weights, early_warning
        )

        # Step 5: Compute expected portfolio metrics
        selected = list(weights.keys())
        exp_ret = 0.0
        for pred in ranked:
            ticker = _get_field(pred, "ticker", "UNKNOWN")
            if ticker in weights:
                exp_ret += weights[ticker] * _get_field(pred, "point_estimate", 0.0)

        # Estimate portfolio vol from returns if available
        import pandas as pd
        exp_vol = 0.0
        if returns_df is not None and isinstance(returns_df, pd.DataFrame):
            avail = [t for t in selected if t in returns_df.columns]
            if len(avail) >= 2:
                rets = returns_df[avail].dropna()
                if len(rets) > 30:
                    w_arr = np.array([weights.get(t, 0.0) for t in avail])
                    cov = rets.cov().values
                    port_var = float(w_arr @ cov @ w_arr)
                    exp_vol = np.sqrt(max(port_var, 0.0)) * np.sqrt(252)

        exp_sharpe = exp_ret / max(exp_vol, 1e-8) if exp_vol > 0 else 0.0

        total = len(predictions)
        pass_rate = len(passed) / total if total > 0 else 0.0

        return PortfolioDecision(
            selected_stocks=selected,
            weights=weights,
            expected_portfolio_return=float(exp_ret),
            expected_vol=float(exp_vol),
            expected_sharpe=float(exp_sharpe),
            rejected=rejected,
            filter_pass_rate=float(pass_rate),
            allocation_mode=self.config.allocation_mode,
            early_warning=early_warning,
            exposure_multiplier=exposure_mult,
        )


def load_config(path: Optional[str] = None) -> PortfolioConfig:
    """Load PortfolioConfig from a JSON file.

    Args:
        path: Path to config JSON. If None, uses engine/portfolio_config.json.

    Returns:
        PortfolioConfig instance.
    """
    import json
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_config.json")

    if not os.path.exists(path):
        logger.info("No config file found at %s, using defaults", path)
        return PortfolioConfig()

    with open(path, "r") as f:
        data = json.load(f)

    filt = data.get("filter", {})
    liq = data.get("liquidity", {})

    filter_config = FilterConfig(
        min_p_up=filt.get("min_p_up", 0.55),
        min_expected_return=filt.get("min_expected_return", 0.005),
        min_q10=filt.get("min_q10", -0.05),
        min_confidence=filt.get("min_confidence", 0.3),
        max_risk_score=filt.get("max_risk_score", 0.8),
        min_meta_trade_prob=filt.get("min_meta_trade_prob", 0.4),
        min_conditions=filt.get("min_conditions", 4),
    )

    liquidity_config = LiquidityConfig(
        min_adv_usd=liq.get("min_adv_usd", 1_000_000),
        max_position_pct_adv=liq.get("max_position_pct_adv", 0.05),
        min_adv_percentile=liq.get("min_adv_percentile", 0.10),
    )

    return PortfolioConfig(
        allocation_mode=data.get("allocation_mode", "risk_parity"),
        max_positions=data.get("max_positions", 20),
        max_single_weight=data.get("max_single_weight", 0.10),
        max_sector_weight=data.get("max_sector_weight", 0.30),
        filter_config=filter_config,
        liquidity_config=liquidity_config,
    )


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    """Extract field from object or dict."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)
