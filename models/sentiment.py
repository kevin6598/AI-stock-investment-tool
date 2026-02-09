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


# ---------------------------------------------------------------------------
# Extended Sentiment Components
# ---------------------------------------------------------------------------

# Event categories with expected market impact direction and magnitude
_EVENT_CATEGORIES = {
    "earnings_beat": {"direction": 1, "magnitude": 0.8},
    "earnings_miss": {"direction": -1, "magnitude": 0.8},
    "fda_approval": {"direction": 1, "magnitude": 0.9},
    "merger_acquisition": {"direction": 1, "magnitude": 0.7},
    "lawsuit": {"direction": -1, "magnitude": 0.5},
    "ceo_change": {"direction": 0, "magnitude": 0.4},
    "dividend_increase": {"direction": 1, "magnitude": 0.3},
    "dividend_cut": {"direction": -1, "magnitude": 0.6},
    "stock_split": {"direction": 1, "magnitude": 0.2},
    "guidance_raise": {"direction": 1, "magnitude": 0.6},
    "guidance_lower": {"direction": -1, "magnitude": 0.7},
    "buyback": {"direction": 1, "magnitude": 0.4},
    "regulatory_action": {"direction": -1, "magnitude": 0.6},
}

_EVENT_KEYWORDS = {
    "earnings_beat": ["beat", "exceeded", "surpassed", "topped", "above expectations"],
    "earnings_miss": ["missed", "fell short", "below expectations", "disappointed"],
    "fda_approval": ["fda approved", "fda approval", "regulatory approval"],
    "merger_acquisition": ["merger", "acquisition", "acquire", "buyout", "takeover"],
    "lawsuit": ["lawsuit", "sued", "litigation", "legal action", "class action"],
    "ceo_change": ["ceo resign", "new ceo", "ceo appointed", "ceo departure"],
    "dividend_increase": ["dividend increase", "raised dividend", "hiked dividend"],
    "dividend_cut": ["dividend cut", "slashed dividend", "suspended dividend"],
    "stock_split": ["stock split", "share split"],
    "guidance_raise": ["raised guidance", "increased outlook", "raised forecast"],
    "guidance_lower": ["lowered guidance", "cut outlook", "reduced forecast"],
    "buyback": ["buyback", "share repurchase", "stock repurchase"],
    "regulatory_action": ["sec investigation", "regulatory probe", "antitrust"],
}

# Macro event keywords and their typical market impact
_MACRO_EVENTS = {
    "rate_hike": {"keywords": ["rate hike", "raised rates", "tightening"], "impact": -0.5},
    "rate_cut": {"keywords": ["rate cut", "lowered rates", "easing"], "impact": 0.5},
    "inflation_high": {"keywords": ["inflation rose", "cpi higher", "prices surged"], "impact": -0.3},
    "inflation_low": {"keywords": ["inflation fell", "cpi lower", "disinflation"], "impact": 0.3},
    "jobs_strong": {"keywords": ["jobs beat", "employment surged", "payrolls exceeded"], "impact": 0.2},
    "jobs_weak": {"keywords": ["jobs miss", "unemployment rose", "layoffs"], "impact": -0.4},
    "gdp_growth": {"keywords": ["gdp grew", "economic growth", "expansion"], "impact": 0.3},
    "recession_risk": {"keywords": ["recession", "contraction", "economic slowdown"], "impact": -0.6},
}


class EventClassifier:
    """Classifies news headlines into event categories with impact scores.

    Uses keyword matching (no ML model required) to detect corporate events
    and estimate their directional impact.
    """

    def classify(self, text: str) -> Dict[str, float]:
        """Classify a headline and return event scores.

        Returns:
            Dict with keys: event_type, event_direction, event_magnitude,
                           event_confidence
        """
        text_lower = text.lower()
        best_event = None
        best_match_count = 0

        for event_type, keywords in _EVENT_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            if match_count > best_match_count:
                best_match_count = match_count
                best_event = event_type

        if best_event is not None:
            info = _EVENT_CATEGORIES[best_event]
            confidence = min(best_match_count / 3.0, 1.0)
            return {
                "event_type": best_event,
                "event_direction": float(info["direction"]),
                "event_magnitude": float(info["magnitude"]),
                "event_confidence": confidence,
            }

        return {
            "event_type": "none",
            "event_direction": 0.0,
            "event_magnitude": 0.0,
            "event_confidence": 0.0,
        }

    def classify_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Classify multiple headlines."""
        return [self.classify(t) for t in texts]


class MacroImpactScorer:
    """Scores macro-economic news for market impact.

    Detects macro events (rate decisions, inflation, jobs, GDP) and
    estimates their directional impact on the broad market.
    """

    def score(self, text: str) -> Dict[str, float]:
        """Score a single headline for macro impact.

        Returns:
            Dict with macro_event, macro_impact, macro_confidence
        """
        text_lower = text.lower()
        best_event = None
        best_score = 0.0

        for event_name, info in _MACRO_EVENTS.items():
            match_count = sum(1 for kw in info["keywords"] if kw in text_lower)
            if match_count > 0:
                score = match_count * abs(info["impact"])
                if score > best_score:
                    best_score = score
                    best_event = event_name

        if best_event is not None:
            impact = _MACRO_EVENTS[best_event]["impact"]
            confidence = min(best_score / 2.0, 1.0)
            return {
                "macro_event": best_event,
                "macro_impact": float(impact),
                "macro_confidence": confidence,
            }

        return {
            "macro_event": "none",
            "macro_impact": 0.0,
            "macro_confidence": 0.0,
        }

    def score_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self.score(t) for t in texts]


class KeywordEmbedder:
    """Creates numerical embedding from keyword frequency analysis.

    Produces a fixed-length vector of keyword category scores that can
    be used as features in the ML pipeline.
    """

    def __init__(self):
        self._categories = {
            "growth": ["growth", "expand", "increase", "surge", "soar", "rally",
                       "bullish", "momentum", "breakout", "innovation"],
            "risk": ["risk", "volatile", "uncertainty", "concern", "warning",
                     "threat", "downturn", "crisis", "plunge", "crash"],
            "value": ["undervalued", "cheap", "discount", "bargain", "value",
                      "dividend", "yield", "income", "buyback", "book value"],
            "quality": ["strong", "solid", "stable", "consistent", "reliable",
                        "profitable", "margin", "efficiency", "competitive"],
            "momentum": ["beat", "exceeded", "outperform", "upgrade", "record",
                         "high", "acceleration", "strength", "surprise"],
        }

    def embed(self, text: str) -> Dict[str, float]:
        """Embed a single text into category scores.

        Returns:
            Dict mapping category -> normalized score (0 to 1).
        """
        words = set(text.lower().split())
        result = {}
        for cat, keywords in self._categories.items():
            matches = sum(1 for kw in keywords if kw in words)
            result[f"kw_{cat}"] = min(matches / 3.0, 1.0)
        return result

    def embed_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self.embed(t) for t in texts]

    @property
    def feature_names(self) -> List[str]:
        return [f"kw_{cat}" for cat in self._categories]


def get_extended_sentiment_features(
    ticker: str,
    model: Optional[SentimentModel] = None,
    aggregator: Optional[SentimentAggregator] = None,
) -> Dict[str, float]:
    """Extended sentiment features including events, macro, and keywords.

    Falls back gracefully if news fetching fails.
    """
    base_features = get_sentiment_features(ticker, model, aggregator)

    # Add event and keyword features with defaults
    extended = {
        **base_features,
        "event_direction": 0.0,
        "event_magnitude": 0.0,
        "macro_impact": 0.0,
        "kw_growth": 0.0,
        "kw_risk": 0.0,
        "kw_value": 0.0,
        "kw_quality": 0.0,
        "kw_momentum": 0.0,
    }

    try:
        from data.news_api import fetch_news
        news_items = fetch_news(ticker)
        if not news_items:
            return extended

        titles = [item["title"] for item in news_items if item.get("title")]
        if not titles:
            return extended

        # Event classification
        ec = EventClassifier()
        events = ec.classify_batch(titles)
        if events:
            # Average across headlines, weighted by confidence
            dirs = [e["event_direction"] * e["event_confidence"] for e in events]
            mags = [e["event_magnitude"] * e["event_confidence"] for e in events]
            total_conf = sum(e["event_confidence"] for e in events)
            if total_conf > 0:
                extended["event_direction"] = sum(dirs) / total_conf
                extended["event_magnitude"] = sum(mags) / total_conf

        # Macro impact
        ms = MacroImpactScorer()
        macro_scores = ms.score_batch(titles)
        impacts = [m["macro_impact"] * m["macro_confidence"] for m in macro_scores]
        total_conf = sum(m["macro_confidence"] for m in macro_scores)
        if total_conf > 0:
            extended["macro_impact"] = sum(impacts) / total_conf

        # Keyword embedding
        ke = KeywordEmbedder()
        kw_scores = ke.embed_batch(titles)
        for cat in ke.feature_names:
            vals = [kw[cat] for kw in kw_scores]
            extended[cat] = float(np.mean(vals)) if vals else 0.0

    except Exception as e:
        logger.debug(f"Extended sentiment features unavailable for {ticker}: {e}")

    return extended
