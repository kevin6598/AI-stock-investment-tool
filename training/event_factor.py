"""
Event Embedding Generator
--------------------------
Generates PCA-reduced sentence embeddings from news headlines as alpha factors.

Design decisions:
  - Frozen embeddings: No gradient flow back to sentence-transformers
  - PCA to 3 components: Prevents curse of dimensionality
  - shift(1): News at close of day t informs predictions for t+1
  - Graceful fallback: Returns zeros if sentence-transformers unavailable
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os
import hashlib
import logging

logger = logging.getLogger(__name__)


class EventEmbeddingGenerator:
    """Generate PCA-reduced embeddings from news text for use as features.

    Usage:
        gen = EventEmbeddingGenerator()
        features = gen.compute_event_features("AAPL", dates, news_df)
        # Returns DataFrame with event_pc1, event_pc2, event_pc3
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        n_components: int = 3,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: Sentence-transformer model name.
            n_components: Number of PCA components to reduce to.
            cache_dir: Directory for caching embeddings. None = no caching.
        """
        self.model_name = model_name
        self.n_components = n_components
        self.cache_dir = cache_dir
        self._model = None
        self._pca = None
        self._available = None  # lazy check

    def _check_available(self) -> bool:
        """Check if sentence-transformers is installed."""
        if self._available is not None:
            return self._available
        try:
            import sentence_transformers  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            logger.info("sentence-transformers not installed; event features will be zeros")
        return self._available

    def _load_or_cache_model(self):
        """Load the sentence-transformers model."""
        if self._model is not None:
            return

        if not self._check_available():
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded sentence-transformer: %s", self.model_name)
        except Exception as e:
            logger.warning("Failed to load sentence-transformer: %s", e)
            self._available = False

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Encode texts into PCA-reduced embeddings.

        Args:
            texts: List of news headlines/texts.

        Returns:
            (n, n_components) array of PCA-reduced embeddings.
            Returns zeros if sentence-transformers unavailable.
        """
        if not texts:
            return np.zeros((0, self.n_components))

        if not self._check_available():
            return np.zeros((len(texts), self.n_components))

        self._load_or_cache_model()
        if self._model is None:
            return np.zeros((len(texts), self.n_components))

        # Check cache
        cache_key = None
        if self.cache_dir is not None:
            cache_key = hashlib.md5(
                "|".join(texts).encode("utf-8", errors="replace")
            ).hexdigest()
            cache_path = os.path.join(self.cache_dir, "{}.npy".format(cache_key))
            if os.path.exists(cache_path):
                try:
                    cached = np.load(cache_path)
                    if cached.shape[0] == len(texts):
                        return cached
                except Exception:
                    pass

        try:
            # Encode: (n, embedding_dim) e.g. (n, 384)
            raw_embeddings = self._model.encode(
                texts, show_progress_bar=False, batch_size=64,
            )

            # PCA reduction
            from sklearn.decomposition import PCA
            n_comp = min(self.n_components, raw_embeddings.shape[1], len(texts))
            if n_comp < 1:
                result = np.zeros((len(texts), self.n_components))
            else:
                self._pca = PCA(n_components=n_comp, random_state=42)
                reduced = self._pca.fit_transform(raw_embeddings)
                # Pad if fewer components than requested
                if reduced.shape[1] < self.n_components:
                    padding = np.zeros((len(texts), self.n_components - reduced.shape[1]))
                    reduced = np.hstack([reduced, padding])
                result = reduced

            # Cache result
            if self.cache_dir is not None and cache_key is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                try:
                    np.save(
                        os.path.join(self.cache_dir, "{}.npy".format(cache_key)),
                        result,
                    )
                except Exception:
                    pass

            return result

        except Exception as e:
            logger.warning("Embedding computation failed: %s", e)
            return np.zeros((len(texts), self.n_components))

    def compute_event_features(
        self,
        ticker: str,
        dates: pd.DatetimeIndex,
        news_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute event embedding features for a ticker across dates.

        Args:
            ticker: Stock ticker symbol.
            dates: DatetimeIndex of the stock DataFrame.
            news_df: Optional DataFrame with columns ['date', 'text'].
                If None, attempts to fetch via data.news_api.

        Returns:
            DataFrame indexed by dates with columns:
                event_pc1, event_pc2, event_pc3
            All zeros if no news or sentence-transformers unavailable.
        """
        col_names = ["event_pc{}".format(i + 1) for i in range(self.n_components)]
        zeros = pd.DataFrame(
            np.zeros((len(dates), self.n_components)),
            index=dates,
            columns=col_names,
        )

        if not self._check_available():
            return zeros

        # Try to get news data
        if news_df is None:
            try:
                from data.news_api import get_news_for_ticker
                news_df = get_news_for_ticker(ticker)
            except Exception:
                return zeros

        if news_df is None or news_df.empty:
            return zeros

        # Ensure news_df has 'date' and 'text' columns
        if "date" not in news_df.columns or "text" not in news_df.columns:
            if "title" in news_df.columns:
                news_df = news_df.rename(columns={"title": "text"})
            else:
                return zeros

        # Group news by date and concatenate texts
        news_df = news_df.copy()
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()

        date_texts = {}  # type: Dict[pd.Timestamp, str]
        for dt, group in news_df.groupby("date"):
            texts = group["text"].dropna().tolist()
            if texts:
                date_texts[pd.Timestamp(dt)] = " | ".join(texts)

        if not date_texts:
            return zeros

        # Sort dates and compute embeddings
        sorted_dates = sorted(date_texts.keys())
        texts = [date_texts[d] for d in sorted_dates]
        embeddings = self.compute_embeddings(texts)

        # Map embeddings to date index
        emb_df = pd.DataFrame(
            embeddings, index=sorted_dates, columns=col_names,
        )

        # Reindex to match requested dates, fill missing with 0
        result = emb_df.reindex(dates, fill_value=0.0)

        # shift(1): news at close of day t -> feature for t+1
        result = result.shift(1).fillna(0.0)

        return result
