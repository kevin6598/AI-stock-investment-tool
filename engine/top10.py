"""
Top 10 Stock Selection Engine
------------------------------
Market-specific top-10 stock picker using model predictions,
risk scoring, sentiment analysis, and liquidity filtering.

Usage:
    engine = Top10Engine()
    result = engine.select(market="US", horizon="1M")
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Top10Stock:
    """Single stock entry in the Top 10 result."""
    rank: int
    ticker: str
    score: float
    direction: str
    p_up: float
    expected_return: float
    confidence: float
    risk_score: float
    sentiment_score: float
    allocation_weight: float
    reasons: List[str]
    # Optional strategy signal fields (populated when strategy_filter is used)
    strategy_mom_60d: Optional[float] = None
    strategy_high_52w_pct: Optional[float] = None
    strategy_mom_60d_decile: Optional[int] = None
    strategy_high_52w_pct_decile: Optional[int] = None


@dataclass
class Top10Result:
    """Full Top 10 selection result."""
    market: str
    horizon: str
    stocks: List[Top10Stock]
    generated_at: str
    model_version: Optional[str]
    total_candidates: int
    pass_rate: float


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Extract field from object or dict."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


class Top10Engine:
    """Market-specific Top 10 stock selection engine.

    Pipeline:
      1. Get universe for market
      2. Batch predict all tickers
      3. Filter using PortfolioDecisionEngine
      4. Rank using enhanced scoring formula
      5. Select top 10
      6. Allocate weights

    Scoring formula:
      FinalScore = 0.30 * p_up
                 + 0.25 * sigmoid(expected_return * 100)
                 + 0.15 * confidence
                 + 0.10 * meta_trade_probability
                 + 0.10 * (1 - risk_score)
                 + 0.10 * sentiment_score
    """

    def __init__(
        self,
        predict_fn: Optional[Any] = None,
        model_version: Optional[str] = None,
    ):
        """
        Args:
            predict_fn: Callable(ticker, horizon) -> prediction dict/object.
                If None, uses api.routes.predict._predict_single.
            model_version: Version string for the active model.
        """
        self._predict_fn = predict_fn
        self._model_version = model_version
        self._sentiment_model = None
        self._sentiment_aggregator = None

    def _get_predict_fn(self):
        """Lazily resolve prediction function."""
        if self._predict_fn is not None:
            return self._predict_fn
        try:
            from api.routes.predict import _predict_single
            return _predict_single
        except ImportError:
            raise RuntimeError("No predict function available")

    def _compute_score(self, pred: Any) -> float:
        """Compute enhanced ranking score for a prediction."""
        p_up = _get_field(pred, "p_up", _get_field(pred, "probability_up", 0.5))
        point = _get_field(pred, "point_estimate", 0.0)
        confidence = _get_field(pred, "confidence", 0.0)
        meta_prob = _get_field(pred, "meta_trade_probability", 0.0) or 0.0
        uncertainty = _get_field(pred, "uncertainty", 0.5) or 0.5
        risk_score = _get_field(pred, "risk_score", uncertainty)

        # Get sentiment score from extended features if available
        sentiment_score = self._extract_sentiment_score(pred)

        score = (
            0.30 * p_up
            + 0.25 * _sigmoid(point * 100)
            + 0.15 * confidence
            + 0.10 * meta_prob
            + 0.10 * (1.0 - risk_score)
            + 0.10 * max(0.0, min(1.0, (sentiment_score + 1.0) / 2.0))
        )
        return float(score)

    def _extract_sentiment_score(self, pred: Any) -> float:
        """Extract sentiment score from prediction or fetch it."""
        # Check if prediction already has sentiment
        score = _get_field(pred, "sentiment_score", None)
        if score is not None:
            return float(score)

        # Try to get from the ticker's extended sentiment features
        # Cache the model/aggregator to avoid reloading per ticker
        ticker = _get_field(pred, "ticker", "")
        if ticker:
            try:
                from models.sentiment import (
                    get_extended_sentiment_features,
                    SentimentModel, SentimentAggregator,
                )
                if self._sentiment_model is None:
                    self._sentiment_model = SentimentModel(use_finbert=False)
                    self._sentiment_aggregator = SentimentAggregator()
                feat = get_extended_sentiment_features(
                    ticker,
                    model=self._sentiment_model,
                    aggregator=self._sentiment_aggregator,
                )
                return feat.get("sentiment_weighted", 0.0)
            except Exception:
                pass
        return 0.0

    def _generate_reasons(
        self, pred: Any, score: float, strategy_candidate: Any = None,
    ) -> List[str]:
        """Generate human-readable reasons for selection."""
        reasons = []

        # Strategy signal reason (first, if available)
        if strategy_candidate is not None:
            reasons.append(
                "Strategy signal: mom_60d d%d (%.2f%%), high_52w_pct d%d (%.1f%%)"
                % (
                    strategy_candidate.mom_60d_decile,
                    strategy_candidate.mom_60d * 100,
                    strategy_candidate.high_52w_pct_decile,
                    strategy_candidate.high_52w_pct * 100,
                )
            )

        p_up = _get_field(pred, "p_up", _get_field(pred, "probability_up", 0.5))
        point = _get_field(pred, "point_estimate", 0.0)
        confidence = _get_field(pred, "confidence", 0.0)
        risk_score = _get_field(pred, "risk_score", 0.5)

        if p_up >= 0.65:
            reasons.append("Strong directional signal (P_up=%.1f%%)" % (p_up * 100))
        elif p_up >= 0.55:
            reasons.append("Moderate upside probability (P_up=%.1f%%)" % (p_up * 100))

        if point >= 0.02:
            reasons.append("High expected return (%.1f%%)" % (point * 100))

        if confidence >= 0.6:
            reasons.append("High model confidence (%.1f%%)" % (confidence * 100))

        if risk_score <= 0.3:
            reasons.append("Low risk profile")

        if not reasons:
            reasons.append("Composite score=%.3f" % score)

        return reasons

    def select(
        self,
        market: str = "US",
        horizon: str = "1M",
        allocation_mode: str = "risk_parity",
        max_stocks: int = 10,
        strategy_filter: Optional[Any] = None,
    ) -> Top10Result:
        """Run the full Top 10 selection pipeline.

        Args:
            market: "US", "KR", or "ALL".
            horizon: Forecast horizon ("1M", "3M", "6M", "1Y").
            allocation_mode: Weight allocation strategy.
            max_stocks: Number of stocks to select (default 10).
            strategy_filter: Optional StrategyFilterResult to restrict universe
                to strategy-matched tickers only.

        Returns:
            Top10Result with ranked stock picks.
        """
        from data.universe_manager import UniverseManager

        # Step 1: Get universe (possibly narrowed by strategy filter)
        if strategy_filter is not None and strategy_filter.candidates:
            tickers = [c.ticker for c in strategy_filter.candidates]
            logger.info(
                "Top10: universe narrowed to %d strategy-matched tickers",
                len(tickers),
            )
        else:
            um = UniverseManager()
            market_upper = market.upper()
            tickers = um.get_universe_by_market(
                "all" if market_upper == "ALL" else market_upper,
            )

        total_candidates = len(tickers)
        logger.info("Top10: %d candidates for market=%s", total_candidates, market)

        # Step 2: Batch predict
        predict_fn = self._get_predict_fn()
        predictions = []
        for ticker in tickers:
            try:
                pred = predict_fn(ticker, horizon)
                predictions.append(pred)
            except Exception as e:
                logger.debug("Prediction failed for %s: %s", ticker, e)

        if not predictions:
            return Top10Result(
                market=market,
                horizon=horizon,
                stocks=[],
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_version=self._model_version,
                total_candidates=total_candidates,
                pass_rate=0.0,
            )

        # Step 3: Filter using PortfolioDecisionEngine
        try:
            from engine.portfolio_decision import PortfolioDecisionEngine
            pde = PortfolioDecisionEngine()
            passed, rejected = pde.filter_candidates(predictions)
        except Exception:
            passed = predictions
            rejected = []

        if not passed:
            return Top10Result(
                market=market,
                horizon=horizon,
                stocks=[],
                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_version=self._model_version,
                total_candidates=total_candidates,
                pass_rate=0.0,
            )

        pass_rate = len(passed) / total_candidates if total_candidates > 0 else 0.0

        # Step 4: Score and rank
        scored = []
        for pred in passed:
            score = self._compute_score(pred)
            scored.append((score, pred))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_stocks]

        # Step 5: Allocate weights
        weights = self._allocate_weights(
            [pred for _, pred in top], allocation_mode,
        )

        # Build strategy feature lookup
        strategy_lookup = {}
        if strategy_filter is not None:
            for c in strategy_filter.candidates:
                strategy_lookup[c.ticker] = c

        # Step 6: Build result
        stocks = []
        for rank_idx, (score, pred) in enumerate(top):
            ticker = _get_field(pred, "ticker", "UNKNOWN")
            p_up = _get_field(pred, "p_up", _get_field(pred, "probability_up", 0.5))
            point = _get_field(pred, "point_estimate", 0.0)
            confidence = _get_field(pred, "confidence", 0.0)
            risk_score = _get_field(pred, "risk_score", 0.5)
            sentiment = self._extract_sentiment_score(pred)
            direction = _get_field(pred, "direction", "UP" if point > 0 else "DOWN")

            # Strategy signal fields
            sc = strategy_lookup.get(ticker)
            strat_kwargs = {}
            if sc is not None:
                strat_kwargs = dict(
                    strategy_mom_60d=round(sc.mom_60d, 6),
                    strategy_high_52w_pct=round(sc.high_52w_pct, 6),
                    strategy_mom_60d_decile=sc.mom_60d_decile,
                    strategy_high_52w_pct_decile=sc.high_52w_pct_decile,
                )

            stocks.append(Top10Stock(
                rank=rank_idx + 1,
                ticker=ticker,
                score=round(score, 4),
                direction=direction,
                p_up=round(float(p_up), 4),
                expected_return=round(float(point), 6),
                confidence=round(float(confidence), 4),
                risk_score=round(float(risk_score), 4),
                sentiment_score=round(float(sentiment), 4),
                allocation_weight=round(weights.get(ticker, 1.0 / max_stocks), 4),
                reasons=self._generate_reasons(pred, score, sc),
                **strat_kwargs,
            ))

        return Top10Result(
            market=market,
            horizon=horizon,
            stocks=stocks,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_version=self._model_version,
            total_candidates=total_candidates,
            pass_rate=round(pass_rate, 4),
        )

    def _allocate_weights(
        self,
        candidates: List[Any],
        mode: str,
    ) -> Dict[str, float]:
        """Allocate portfolio weights to top picks."""
        n = len(candidates)
        if n == 0:
            return {}

        tickers = [_get_field(p, "ticker", "UNKNOWN") for p in candidates]

        if mode == "equal":
            return {t: 1.0 / n for t in tickers}

        if mode == "risk_parity":
            # Inverse-risk weighting
            inv_risk = []
            for pred in candidates:
                risk = _get_field(pred, "risk_score", 0.5) or 0.5
                inv_risk.append(1.0 / max(risk, 0.05))
            total = sum(inv_risk)
            if total > 0:
                return {t: ir / total for t, ir in zip(tickers, inv_risk)}

        # Fallback: equal weight
        return {t: 1.0 / n for t in tickers}
