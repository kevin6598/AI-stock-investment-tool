"""
Model Cache
-----------
Loads model artifacts at startup, caches in memory, supports hot-reload.
"""

from typing import Any, Dict, List, Optional
import os
import json
import logging
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

# Default artifact directory
_ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


class ModelCache:
    """In-memory cache for model artifacts.

    Loads at startup from artifact directory or model registry.
    Supports hot-reload for zero-downtime model updates.

    Artifact directory structure:
        artifacts/
            model.pt (or model.pkl)
            config.json
            feature_scaler.pkl
            ticker_list.json
            feature_columns.json
            training_metadata.json
    """

    def __init__(self, artifact_dir: Optional[str] = None):
        self.artifact_dir = artifact_dir or _ARTIFACT_DIR
        self.model = None  # type: Any
        self.meta_model = None  # type: Any
        self.config = {}  # type: Dict
        self.feature_scaler = None  # type: Any
        self.ticker_list = []  # type: List[str]
        self.ticker_to_id = {}  # type: Dict[str, int]
        self.feature_columns = []  # type: List[str]
        self.metadata = {}  # type: Dict
        self._loaded = False
        self._load_time = None  # type: Optional[datetime]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> Optional[str]:
        return self.metadata.get("version")

    @property
    def trained_tickers(self) -> List[str]:
        return self.ticker_list

    def load(self) -> bool:
        """Load model artifacts from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.isdir(self.artifact_dir):
            logger.warning(f"Artifact directory not found: {self.artifact_dir}")
            return self._try_load_from_registry()

        try:
            # Load config
            config_path = os.path.join(self.artifact_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)

            # Load feature columns
            cols_path = os.path.join(self.artifact_dir, "feature_columns.json")
            if os.path.exists(cols_path):
                with open(cols_path, "r") as f:
                    self.feature_columns = json.load(f)

            # Load ticker list
            ticker_path = os.path.join(self.artifact_dir, "ticker_list.json")
            if os.path.exists(ticker_path):
                with open(ticker_path, "r") as f:
                    self.ticker_list = json.load(f)
                    self.ticker_to_id = {t: i for i, t in enumerate(self.ticker_list)}

            # Load feature scaler
            scaler_path = os.path.join(self.artifact_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)

            # Load metadata
            meta_path = os.path.join(self.artifact_dir, "training_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)

            # Load model (support both state_dict and full pickle)
            model_pt = os.path.join(self.artifact_dir, "model.pt")
            model_pkl = os.path.join(self.artifact_dir, "model.pkl")

            if os.path.exists(model_pt):
                import torch
                self.model = torch.load(model_pt, map_location="cpu")
                logger.info("Loaded PyTorch model from artifacts")
            elif os.path.exists(model_pkl):
                with open(model_pkl, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded pickle model from artifacts")
            else:
                logger.warning("No model file found in artifacts")
                return self._try_load_from_registry()

            # Load meta-labeling model if available
            meta_path = os.path.join(self.artifact_dir, "meta_model.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.meta_model = pickle.load(f)
                logger.info("Loaded meta-labeling model from artifacts")

            self._loaded = True
            self._load_time = datetime.now()
            logger.info(f"Model cache loaded: version={self.model_version}, "
                        f"tickers={len(self.ticker_list)}, features={len(self.feature_columns)}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            return self._try_load_from_registry()

    def _try_load_from_registry(self) -> bool:
        """Fallback: try loading from model registry using robust auto-selection."""
        try:
            from training.model_versioning import ModelRegistry
            registry = ModelRegistry()

            # Use composite-score-based selection first
            version = registry.select_best_model(horizon="1M", criteria="robust")

            # Fall back to active model if select_best_model returns None
            if version is None:
                version = registry.get_active_model("hybrid_multimodal", "1M")
            if version is None:
                for model_type in ["lightgbm", "elastic_net", "transformer", "lstm_attention"]:
                    version = registry.get_active_model(model_type, "1M")
                    if version is not None:
                        break

            if version is not None:
                # Checksum validation if available
                artifact_path = version.artifact_path
                expected_hash = version.metrics.get("artifact_checksum")
                if expected_hash and os.path.exists(artifact_path):
                    try:
                        from api.auth import verify_artifact_checksum
                        if not verify_artifact_checksum(artifact_path, expected_hash):
                            logger.warning("Checksum mismatch for %s, loading anyway",
                                          version.version_id)
                    except ImportError:
                        pass

                self.model = registry.load_model(version.version_id)
                self.metadata = {
                    "version": version.version_id,
                    "model_type": version.model_type,
                    "metrics": version.metrics,
                }
                self._loaded = True
                self._load_time = datetime.now()
                logger.info("Loaded model from registry: %s (type=%s)",
                           version.version_id, version.model_type)
                return True
        except Exception as e:
            logger.debug("Registry fallback failed: %s", e)

        logger.info("No pre-trained model available; API will train on-demand")
        return False

    def hot_reload(self, new_artifact_dir: Optional[str] = None) -> bool:
        """Hot-reload model artifacts without downtime.

        Args:
            new_artifact_dir: New artifact directory (optional).

        Returns:
            True if reload succeeded.
        """
        old_model = self.model
        old_config = self.config

        if new_artifact_dir:
            self.artifact_dir = new_artifact_dir

        try:
            success = self.load()
            if not success:
                # Restore previous model
                self.model = old_model
                self.config = old_config
                logger.warning("Hot-reload failed; keeping previous model")
            return success
        except Exception as e:
            self.model = old_model
            self.config = old_config
            logger.error(f"Hot-reload failed: {e}")
            return False

    def get_ticker_id(self, ticker: str) -> int:
        """Get ticker ID for embedding lookup. Returns 0 for unseen tickers."""
        return self.ticker_to_id.get(ticker.upper(), 0)

    def is_trained_ticker(self, ticker: str) -> bool:
        """Check if ticker was in the training set."""
        return ticker.upper() in self.ticker_to_id

    @property
    def uptime_seconds(self) -> float:
        if self._load_time is None:
            return 0.0
        return (datetime.now() - self._load_time).total_seconds()
