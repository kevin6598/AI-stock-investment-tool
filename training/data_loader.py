"""
Panel Data Loader
------------------
Memory-efficient chunked data loader for large stock panels.

Loads data in chunks of N tickers at a time to avoid OOM on
large universes (300+ tickers x 15+ years).

Supports lazy feature computation and disk caching.
"""

from typing import Dict, Iterator, List, Optional
import os
import hashlib
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PanelDataLoader:
    """Memory-efficient chunked data loader for large panels.

    Loads data in chunks of N tickers at a time, optionally caching
    processed chunks to disk for faster re-runs.

    Usage:
        loader = PanelDataLoader(tickers, chunk_size=50)
        for chunk_df in loader.iter_chunks("5y"):
            # process chunk
        full_panel = loader.build_full_panel("5y")
    """

    def __init__(
        self,
        tickers: List[str],
        chunk_size: int = 50,
        cache_dir: str = ".cache/panels",
        random_seed: int = 42,
    ):
        """
        Args:
            tickers: Full list of ticker symbols.
            chunk_size: Number of tickers per chunk.
            cache_dir: Directory for disk caching.
            random_seed: Seed for reproducible shuffling.
        """
        self.tickers = sorted(tickers)
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        self._chunks = self._build_chunks()

    def _build_chunks(self) -> List[List[str]]:
        """Split ticker list into chunks."""
        chunks = []
        for i in range(0, len(self.tickers), self.chunk_size):
            chunks.append(self.tickers[i:i + self.chunk_size])
        return chunks

    def _cache_key(self, chunk_tickers: List[str], period: str) -> str:
        """Generate a cache key for a chunk."""
        raw = "|".join(sorted(chunk_tickers)) + "|" + period
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> str:
        """Get cache file path for a key."""
        return os.path.join(self.cache_dir, "%s.parquet" % cache_key)

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Try to load a chunk from disk cache."""
        path = self._cache_path(cache_key)
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                logger.debug("Cache hit: %s", cache_key)
                return df
            except Exception as e:
                logger.debug("Cache read failed: %s", e)
        return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Save a chunk to disk cache."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = self._cache_path(cache_key)
            df.to_parquet(path)
            logger.debug("Cached: %s", cache_key)
        except Exception as e:
            logger.debug("Cache write failed: %s", e)

    def load_chunk(
        self,
        chunk_tickers: List[str],
        period: str = "5y",
        market_ticker: str = "SPY",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load and featurize a chunk of tickers.

        Args:
            chunk_tickers: List of tickers in this chunk.
            period: Data lookback period.
            market_ticker: Market index for macro features.
            use_cache: Whether to use disk cache.

        Returns:
            Panel DataFrame with multi-index (date, ticker).
        """
        cache_key = self._cache_key(chunk_tickers, period)

        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        from data.stock_api import get_historical_data, get_stock_info
        from training.feature_engineering import build_panel_dataset, cross_sectional_normalize

        stock_dfs = {}
        stock_infos = {}

        for ticker in chunk_tickers:
            df = get_historical_data(ticker, period=period)
            if df.empty:
                logger.warning("No data for %s, skipping", ticker)
                continue
            stock_dfs[ticker] = df
            info = get_stock_info(ticker) or {}
            stock_infos[ticker] = info

        if not stock_dfs:
            return pd.DataFrame()

        # Market index
        market_df = get_historical_data(market_ticker, period=period)
        if market_df.empty:
            market_df = None

        # Build panel
        horizons = [21, 63, 126]
        panel = build_panel_dataset(stock_dfs, stock_infos, market_df, horizons)
        panel = cross_sectional_normalize(panel)

        if use_cache and not panel.empty:
            self._save_to_cache(cache_key, panel)

        return panel

    def iter_chunks(
        self,
        period: str = "5y",
        market_ticker: str = "SPY",
        use_cache: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """Iterate over chunks, yielding panel DataFrames.

        Args:
            period: Data lookback period.
            market_ticker: Market index ticker.
            use_cache: Whether to use disk cache.

        Yields:
            Panel DataFrame for each chunk.
        """
        for i, chunk_tickers in enumerate(self._chunks):
            logger.info("Loading chunk %d/%d (%d tickers)",
                       i + 1, len(self._chunks), len(chunk_tickers))
            chunk_df = self.load_chunk(chunk_tickers, period, market_ticker, use_cache)
            if not chunk_df.empty:
                yield chunk_df

    def build_full_panel(
        self,
        period: str = "5y",
        market_ticker: str = "SPY",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Build the full panel by concatenating all chunks.

        For small universes (<100 tickers), this is fine.
        For large universes, prefer iter_chunks() to avoid OOM.

        Args:
            period: Data lookback period.
            market_ticker: Market index ticker.
            use_cache: Whether to use disk cache.

        Returns:
            Full panel DataFrame.
        """
        chunks = []
        for chunk_df in self.iter_chunks(period, market_ticker, use_cache):
            chunks.append(chunk_df)

        if not chunks:
            return pd.DataFrame()

        full = pd.concat(chunks)

        # Remove potential duplicate (date, ticker) entries
        if isinstance(full.index, pd.MultiIndex):
            full = full[~full.index.duplicated(keep="last")]

        logger.info("Full panel: %d rows, %d columns", len(full), len(full.columns))
        return full

    def clear_cache(self) -> int:
        """Remove all cached chunk files.

        Returns:
            Number of files removed.
        """
        removed = 0
        if os.path.isdir(self.cache_dir):
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".parquet"):
                    try:
                        os.remove(os.path.join(self.cache_dir, fname))
                        removed += 1
                    except Exception:
                        pass
        logger.info("Cleared %d cached chunks", removed)
        return removed

    @property
    def n_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

    @property
    def n_tickers(self) -> int:
        """Total number of tickers."""
        return len(self.tickers)
