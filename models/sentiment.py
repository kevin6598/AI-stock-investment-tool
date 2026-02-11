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
            magnitude = float(info["magnitude"])
            return {
                "event_type": best_event,
                "event_direction": float(info["direction"]),
                "event_magnitude": magnitude,
                "event_confidence": confidence,
                "surprise_magnitude": magnitude * confidence,
            }

        return {
            "event_type": "none",
            "event_direction": 0.0,
            "event_magnitude": 0.0,
            "event_confidence": 0.0,
            "surprise_magnitude": 0.0,
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


class ConfidenceGatedEventEncoder:
    """Wraps EventClassifier, MacroImpactScorer, and KeywordEmbedder
    with confidence-based gating.

    Produces 10 gated numeric features from a single text:
      - event_direction, event_magnitude, surprise_magnitude (from EventClassifier)
      - macro_impact (from MacroImpactScorer)
      - kw_growth, kw_risk, kw_value, kw_quality, kw_momentum (from KeywordEmbedder)
      - event_confidence (from EventClassifier)

    Sparsity rule: if confidence < threshold, all event features are zeroed.
    """

    def __init__(self, confidence_threshold: float = 0.2):
        self.confidence_threshold = confidence_threshold
        self.event_classifier = EventClassifier()
        self.macro_scorer = MacroImpactScorer()
        self.keyword_embedder = KeywordEmbedder()

    @property
    def feature_names(self) -> List[str]:
        return [
            "event_direction", "event_magnitude", "surprise_magnitude",
            "event_confidence", "macro_impact",
            "kw_growth", "kw_risk", "kw_value", "kw_quality", "kw_momentum",
        ]

    def encode(self, text: str) -> Dict[str, float]:
        """Encode a single text into gated numeric features.

        Args:
            text: Input headline/article.

        Returns:
            Dict of 10 feature name -> value.
        """
        event = self.event_classifier.classify(text)
        macro = self.macro_scorer.score(text)
        kw = self.keyword_embedder.embed(text)

        confidence = event.get("event_confidence", 0.0)

        # Gating: zero out if confidence below threshold
        if confidence < self.confidence_threshold:
            return {name: 0.0 for name in self.feature_names}

        return {
            "event_direction": event.get("event_direction", 0.0),
            "event_magnitude": event.get("event_magnitude", 0.0),
            "surprise_magnitude": event.get("surprise_magnitude", 0.0),
            "event_confidence": confidence,
            "macro_impact": macro.get("macro_impact", 0.0),
            "kw_growth": kw.get("kw_growth", 0.0),
            "kw_risk": kw.get("kw_risk", 0.0),
            "kw_value": kw.get("kw_value", 0.0),
            "kw_quality": kw.get("kw_quality", 0.0),
            "kw_momentum": kw.get("kw_momentum", 0.0),
        }

    def encode_batch(self, texts: List[str]) -> Dict[str, float]:
        """Encode multiple texts with confidence-weighted aggregation.

        Args:
            texts: List of headlines.

        Returns:
            Dict of aggregated feature name -> value.
        """
        if not texts:
            return {name: 0.0 for name in self.feature_names}

        all_features = [self.encode(t) for t in texts]

        # Confidence-weighted aggregation
        confidences = [f.get("event_confidence", 0.0) for f in all_features]
        total_conf = sum(confidences)

        if total_conf < 1e-8:
            return {name: 0.0 for name in self.feature_names}

        result = {}
        for name in self.feature_names:
            vals = [f[name] * c for f, c in zip(all_features, confidences)]
            result[name] = sum(vals) / total_conf

        return result


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
        "event_confidence": 0.0,
        "surprise_magnitude": 0.0,
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

        # Use ConfidenceGatedEventEncoder for all event/keyword features
        encoder = ConfidenceGatedEventEncoder()
        gated = encoder.encode_batch(titles)
        for key, val in gated.items():
            extended[key] = val

    except Exception as e:
        logger.debug(f"Extended sentiment features unavailable for {ticker}: {e}")

    return extended


# ---------------------------------------------------------------------------
# Sentence Embedding Sentiment Model
# ---------------------------------------------------------------------------

_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass


class SentenceEmbeddingSentimentModel:
    """Sentence-embedding-based sentiment scoring using all-MiniLM-L6-v2.

    Encodes headlines into 384-dim embeddings, applies PCA to 8 dimensions,
    and computes cosine similarity against positive/negative reference centroids.

    Output: sentence_sentiment_score (1 dim) + sentence_pca_0..7 (8 dims) = 9 features.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pca_dims: int = 8,
    ):
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )
        self.pca_dims = pca_dims
        self._model = SentenceTransformer(model_name)
        self._pca = None  # fitted lazily
        # Reference texts for centroid computation
        self._pos_refs = [
            "strong earnings beat expectations revenue growth",
            "stock surges on positive outlook upgrade",
            "record profit margins innovative breakthrough",
            "bullish momentum dividend increase expansion",
        ]
        self._neg_refs = [
            "earnings miss revenue decline loss warning",
            "stock plunges on weak guidance downgrade",
            "bankruptcy risk debt default recession",
            "bearish selloff layoffs fraud investigation",
        ]
        self._pos_centroid = None  # type: Optional[np.ndarray]
        self._neg_centroid = None  # type: Optional[np.ndarray]
        self._init_centroids()

    def _init_centroids(self):
        """Compute reference centroids for positive/negative sentiment."""
        pos_emb = self._model.encode(self._pos_refs, show_progress_bar=False)
        neg_emb = self._model.encode(self._neg_refs, show_progress_bar=False)
        self._pos_centroid = np.mean(pos_emb, axis=0)
        self._neg_centroid = np.mean(neg_emb, axis=0)
        # Normalize
        self._pos_centroid = self._pos_centroid / (np.linalg.norm(self._pos_centroid) + 1e-8)
        self._neg_centroid = self._neg_centroid / (np.linalg.norm(self._neg_centroid) + 1e-8)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def score_texts(self, texts: List[str]) -> Dict[str, float]:
        """Score a batch of texts and return aggregated features.

        Returns:
            Dict with sentence_sentiment_score and sentence_pca_0..7.
        """
        if not texts:
            result = {"sentence_sentiment_score": 0.0, "sentence_similarity": 0.0}
            for i in range(self.pca_dims):
                result["sentence_pca_%d" % i] = 0.0
            return result

        embeddings = self._model.encode(texts, show_progress_bar=False)

        # Cosine similarity to positive/negative centroids
        scores = []
        for emb in embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            pos_sim = self._cosine_sim(emb_norm, self._pos_centroid)
            neg_sim = self._cosine_sim(emb_norm, self._neg_centroid)
            scores.append(pos_sim - neg_sim)

        avg_score = float(np.mean(scores))
        avg_similarity = float(np.mean(np.abs(scores)))

        # PCA reduction
        from sklearn.decomposition import PCA
        if self._pca is None:
            n_components = min(self.pca_dims, embeddings.shape[0], embeddings.shape[1])
            self._pca = PCA(n_components=n_components)
            pca_result = self._pca.fit_transform(embeddings)
        else:
            pca_result = self._pca.transform(embeddings)

        # Average PCA components across all headlines
        avg_pca = np.mean(pca_result, axis=0)

        result = {
            "sentence_sentiment_score": avg_score,
            "sentence_similarity": avg_similarity,
        }
        for i in range(self.pca_dims):
            if i < len(avg_pca):
                result["sentence_pca_%d" % i] = float(avg_pca[i])
            else:
                result["sentence_pca_%d" % i] = 0.0

        return result


class DualSentimentEngine:
    """Dual sentiment engine combining FinBERT + Sentence Embedding models.

    Weighted average: final_score = 0.6 * finbert_score + 0.4 * sentence_score
    Falls back to single-engine if one model is unavailable.
    """

    def __init__(
        self,
        finbert_weight: float = 0.6,
        sentence_weight: float = 0.4,
        use_finbert: bool = True,
    ):
        self.finbert_weight = finbert_weight
        self.sentence_weight = sentence_weight
        self._finbert_model = None  # type: Optional[SentimentModel]
        self._sentence_model = None  # type: Optional[SentenceEmbeddingSentimentModel]
        self._aggregator = SentimentAggregator()

        # Initialize FinBERT
        try:
            self._finbert_model = SentimentModel(use_finbert=use_finbert)
        except Exception as e:
            logger.warning("FinBERT initialization failed: %s", e)

        # Initialize Sentence Embedding model
        try:
            self._sentence_model = SentenceEmbeddingSentimentModel()
            logger.info("Loaded sentence embedding sentiment model")
        except Exception as e:
            logger.info("Sentence embedding model unavailable: %s", e)

    @property
    def has_finbert(self) -> bool:
        return self._finbert_model is not None

    @property
    def has_sentence_model(self) -> bool:
        return self._sentence_model is not None

    def score(self, ticker: str) -> Dict[str, float]:
        """Compute dual-engine sentiment features for a ticker.

        Returns:
            Dict with all base features plus dual-engine additions:
              dual_sentiment_score, sentence_similarity, sentence_pca_0..7
        """
        # Get base features from FinBERT path
        base = get_extended_sentiment_features(
            ticker, self._finbert_model, self._aggregator,
        )

        # Defaults for dual-engine features
        dual_features = {
            "dual_sentiment_score": base.get("sentiment_weighted", 0.0),
            "sentence_similarity": 0.0,
        }
        for i in range(8):
            dual_features["sentence_pca_%d" % i] = 0.0

        # Sentence embedding scoring
        try:
            if self._sentence_model is not None:
                from data.news_api import fetch_news
                news_items = fetch_news(ticker)
                if news_items:
                    titles = [item["title"] for item in news_items
                              if item.get("title")]
                    if titles:
                        sent_features = self._sentence_model.score_texts(titles)
                        finbert_score = base.get("sentiment_weighted", 0.0)
                        sentence_score = sent_features.get(
                            "sentence_sentiment_score", 0.0,
                        )
                        # Weighted combination
                        dual_features["dual_sentiment_score"] = (
                            self.finbert_weight * finbert_score
                            + self.sentence_weight * sentence_score
                        )
                        dual_features["sentence_similarity"] = sent_features.get(
                            "sentence_similarity", 0.0,
                        )
                        for i in range(8):
                            key = "sentence_pca_%d" % i
                            dual_features[key] = sent_features.get(key, 0.0)
        except Exception as e:
            logger.debug("Sentence embedding scoring failed for %s: %s", ticker, e)

        return {**base, **dual_features}
