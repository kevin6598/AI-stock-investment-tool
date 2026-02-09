"""
Sentiment Engine
----------------
FinBERT-based financial sentiment scoring with time-decay aggregation.

Architecture:
  - SentimentModel: wraps FinBERT (or keyword fallback) for per-headline scoring
  - SentimentAggregator: time-decay weighted aggregation into daily scores
  - Output: numerical features usable by ML models

FinBERT outputs: P(positive), P(negative), P(neutral)
We convert to a single sentiment score: sentiment = P(pos) - P(neg)
Range: [-1, +1]
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try to import transformers; fall back to keyword-based approach
_HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    logger.info("transformers not installed; using keyword-based sentiment fallback")


# ---------------------------------------------------------------------------
# FinBERT Sentiment Model
# ---------------------------------------------------------------------------

class FinBERTSentimentModel:
    """FinBERT-based financial sentiment scoring.

    Uses ProsusAI/finbert, a BERT model fine-tuned on financial text.
    Labels: positive, negative, neutral.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers library required for FinBERT. "
                              "Install with: pip install transformers")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        # FinBERT label order: positive=0, negative=1, neutral=2
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}

    def score_texts(self, texts: List[str], batch_size: int = 16
                    ) -> List[Dict[str, float]]:
        """Score a batch of texts.

        Args:
            texts: list of headline/article strings.

        Returns:
            List of dicts with keys: positive, negative, neutral, sentiment_score.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for p in probs:
                results.append({
                    "positive": float(p[0]),
                    "negative": float(p[1]),
                    "neutral": float(p[2]),
                    "sentiment_score": float(p[0] - p[1]),  # range [-1, +1]
                })

        return results


# ---------------------------------------------------------------------------
# Keyword-based Fallback
# ---------------------------------------------------------------------------

# Loughran-McDonald financial sentiment lexicon (abbreviated)
_POSITIVE_WORDS = {
    "surge", "gain", "rally", "profit", "growth", "upgrade", "beat",
    "exceed", "outperform", "strong", "bullish", "record", "high",
    "jump", "soar", "boost", "optimistic", "positive", "recovery",
    "momentum", "breakthrough", "innovative", "dividend", "expand",
    "buy", "overweight", "upside", "revenue", "earnings",
}

_NEGATIVE_WORDS = {
    "fall", "drop", "decline", "loss", "miss", "downgrade", "weak",
    "bearish", "crash", "plunge", "risk", "warning", "concern",
    "uncertainty", "debt", "default", "bankruptcy", "recession",
    "sell", "underweight", "downside", "layoff", "cut", "lawsuit",
    "investigation", "fraud", "penalty", "volatile", "inflation",
}


class KeywordSentimentModel:
    """Simple keyword-based sentiment scoring (fallback when no FinBERT)."""

    def score_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        results = []
        for text in texts:
            words = set(text.lower().split())
            pos_count = len(words & _POSITIVE_WORDS)
            neg_count = len(words & _NEGATIVE_WORDS)
            total = pos_count + neg_count

            if total == 0:
                score = 0.0
                pos_prob = 0.33
                neg_prob = 0.33
            else:
                score = (pos_count - neg_count) / total
                pos_prob = pos_count / total
                neg_prob = neg_count / total

            results.append({
                "positive": pos_prob,
                "negative": neg_prob,
                "neutral": 1.0 - pos_prob - neg_prob,
                "sentiment_score": score,
            })
        return results


# ---------------------------------------------------------------------------
# Unified Sentiment Model (auto-selects FinBERT or keyword)
# ---------------------------------------------------------------------------

class SentimentModel:
    """Unified interface. Uses FinBERT if available, keyword fallback otherwise."""

    def __init__(self, use_finbert: bool = True):
        self._model = None
        self._is_finbert = False

        if use_finbert and _HAS_TRANSFORMERS:
            try:
                self._model = FinBERTSentimentModel()
                self._is_finbert = True
                logger.info("Loaded FinBERT sentiment model")
            except Exception as e:
                logger.warning(f"FinBERT load failed ({e}), using keyword fallback")
                self._model = KeywordSentimentModel()
        else:
            self._model = KeywordSentimentModel()
            logger.info("Using keyword-based sentiment model")

    @property
    def is_finbert(self) -> bool:
        return self._is_finbert

    def score(self, texts: List[str]) -> List[Dict[str, float]]:
        """Score a list of text strings."""
        if not texts:
            return []
        return self._model.score_texts(texts)

    def score_single(self, text: str) -> Dict[str, float]:
        """Score a single text string."""
        results = self.score([text])
        return results[0] if results else {"positive": 0.33, "negative": 0.33,
                                            "neutral": 0.34, "sentiment_score": 0.0}


# ---------------------------------------------------------------------------
# Sentiment Aggregator (time-decay weighted daily score)
# ---------------------------------------------------------------------------

class SentimentAggregator:
    """Aggregates per-headline sentiment into daily features with time decay.

    Time decay: recent headlines matter more.
    Formula: aggregated_score = sum(score_i * decay_i) / sum(decay_i)
    where decay_i = exp(-lambda * age_in_hours_i)
    """

    def __init__(self, half_life_hours: float = 48.0):
        """
        Args:
            half_life_hours: headlines older than this contribute half as much.
        """
        self.decay_lambda = np.log(2) / half_life_hours

    def aggregate(
        self,
        scored_headlines: List[Dict],
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Aggregate scored headlines into a single feature set.

        Args:
            scored_headlines: List of dicts with keys:
                sentiment_score, positive, negative, neutral, timestamp.
            reference_time: The "current" time for decay calculation.

        Returns:
            Dict with aggregated features:
                sentiment_mean, sentiment_weighted, sentiment_std,
                positive_ratio, negative_ratio, news_volume,
                sentiment_momentum (recent vs older).
        """
        if not scored_headlines:
            return {
                "sentiment_mean": 0.0,
                "sentiment_weighted": 0.0,
                "sentiment_std": 0.0,
                "positive_ratio": 0.33,
                "negative_ratio": 0.33,
                "news_volume": 0,
                "sentiment_momentum": 0.0,
            }

        if reference_time is None:
            reference_time = datetime.now()

        scores = []
        weights = []
        pos_count = 0
        neg_count = 0

        for item in scored_headlines:
            s = item.get("sentiment_score", 0.0)
            scores.append(s)

            # Parse timestamp for decay
            ts = item.get("timestamp", "")
            try:
                if isinstance(ts, str) and ts:
                    # Try multiple formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                                "%a, %d %b %Y %H:%M:%S"]:
                        try:
                            dt = datetime.strptime(ts[:19], fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        dt = reference_time
                elif isinstance(ts, datetime):
                    dt = ts
                else:
                    dt = reference_time
            except Exception:
                dt = reference_time

            age_hours = max((reference_time - dt).total_seconds() / 3600, 0)
            decay = np.exp(-self.decay_lambda * age_hours)
            weights.append(decay)

            if s > 0.1:
                pos_count += 1
            elif s < -0.1:
                neg_count += 1

        scores = np.array(scores)
        weights = np.array(weights)
        n = len(scores)

        # Weighted mean
        w_sum = weights.sum()
        weighted_mean = (scores * weights).sum() / w_sum if w_sum > 0 else 0.0

        # Momentum: recent half vs older half
        mid = n // 2
        if mid > 0:
            recent_mean = scores[:mid].mean()
            older_mean = scores[mid:].mean()
            momentum = recent_mean - older_mean
        else:
            momentum = 0.0

        return {
            "sentiment_mean": float(scores.mean()),
            "sentiment_weighted": float(weighted_mean),
            "sentiment_std": float(scores.std()) if n > 1 else 0.0,
            "positive_ratio": pos_count / n if n > 0 else 0.33,
            "negative_ratio": neg_count / n if n > 0 else 0.33,
            "news_volume": n,
            "sentiment_momentum": float(momentum),
        }


# ---------------------------------------------------------------------------
# End-to-end: fetch + score + aggregate
# ---------------------------------------------------------------------------

def get_sentiment_features(
    ticker: str,
    model: Optional[SentimentModel] = None,
    aggregator: Optional[SentimentAggregator] = None,
) -> Dict[str, float]:
    """One-call function: fetch news, score sentiment, aggregate.

    Args:
        ticker: Stock ticker.
        model: SentimentModel instance (reuse to avoid reloading).
        aggregator: SentimentAggregator instance.

    Returns:
        Dict of aggregated sentiment features.
    """
    from data.news_api import fetch_news

    if model is None:
        model = SentimentModel(use_finbert=False)  # default to keyword for speed
    if aggregator is None:
        aggregator = SentimentAggregator()

    # Fetch
    news_items = fetch_news(ticker)

    if not news_items:
        return aggregator.aggregate([])

    # Score
    titles = [item["title"] for item in news_items if item.get("title")]
    if not titles:
        return aggregator.aggregate([])

    scores = model.score(titles)

    # Merge scores with timestamps
    scored = []
    for item, score in zip(news_items, scores):
        scored.append({**score, "timestamp": item.get("timestamp", "")})

    # Aggregate
    return aggregator.aggregate(scored)
