"""
Model Versioning
-----------------
SQLite-backed model registry for tracking trained model versions.

Features:
  - Save/load model artifacts (pickle for sklearn/lgbm, torch.save for PyTorch)
  - Track hyperparameters, metrics, and metadata per version
  - Activate/deactivate model versions
  - List and query model history
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os
import sqlite3
import json
import pickle
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "model_store")


@dataclass
class ModelVersion:
    """Metadata for a saved model version."""
    version_id: str
    model_type: str
    horizon: str
    created_at: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifact_path: str
    is_active: bool = False
    config: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """SQLite-backed model registry.

    Usage:
        registry = ModelRegistry()
        vid = registry.save_model(model, "lightgbm", "1M", params, metrics)
        loaded = registry.load_model(vid)
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(MODEL_STORE_DIR, "registry.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the model versions table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                horizon TEXT NOT NULL,
                created_at TEXT NOT NULL,
                params TEXT,
                metrics TEXT,
                artifact_path TEXT NOT NULL,
                is_active INTEGER DEFAULT 0,
                config TEXT
            )
        """)
        # Add config column if upgrading from older schema
        try:
            conn.execute("ALTER TABLE model_versions ADD COLUMN config TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
        conn.close()

    def _generate_version_id(self, model_type: str, horizon: str) -> str:
        """Generate a version ID: {model_type}_{horizon}_{YYYYMMDD_HHMMSS}."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_type}_{horizon}_{ts}"

    def save_model(
        self,
        model: Any,
        model_type: str,
        horizon: str,
        params: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Any] = None,
    ) -> str:
        """Save a model artifact and register it.

        Args:
            model: a BaseModel instance (or any picklable/torch model).
            model_type: e.g. "elastic_net", "lightgbm", "lstm_attention", "transformer".
            horizon: e.g. "1M", "3M", "6M".
            params: hyperparameters dict.
            metrics: evaluation metrics dict.

        Returns:
            version_id string.
        """
        version_id = self._generate_version_id(model_type, horizon)
        artifacts_dir = os.path.join(MODEL_STORE_DIR, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Determine save method
        is_torch_model = model_type in ("lstm_attention", "transformer")
        if is_torch_model:
            artifact_path = os.path.join(artifacts_dir, f"{version_id}.pt")
            self._save_torch(model, artifact_path)
        else:
            artifact_path = os.path.join(artifacts_dir, f"{version_id}.pkl")
            self._save_pickle(model, artifact_path)

        # Serialize config if provided
        config_json = None
        if config is not None:
            if hasattr(config, "to_json"):
                config_json = json.dumps(config.to_json())
            elif isinstance(config, dict):
                config_json = json.dumps(config)

        # Register in database
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO model_versions
               (version_id, model_type, horizon, created_at, params, metrics, artifact_path, is_active, config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                model_type,
                horizon,
                datetime.now().isoformat(),
                json.dumps(params or {}),
                json.dumps(metrics or {}),
                artifact_path,
                0,
                config_json,
            ),
        )
        conn.commit()
        conn.close()

        logger.info(f"Saved model version: {version_id}")
        return version_id

    def load_model(self, version_id: str) -> Any:
        """Load a model artifact by version ID.

        Args:
            version_id: the model version identifier.

        Returns:
            The deserialized model instance.
        """
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT model_type, artifact_path FROM model_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"Version not found: {version_id}")

        model_type, artifact_path = row

        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact missing: {artifact_path}")

        is_torch_model = model_type in ("lstm_attention", "transformer")
        if is_torch_model:
            return self._load_torch(artifact_path)
        else:
            return self._load_pickle(artifact_path)

    def get_active_model(self, model_type: str, horizon: str) -> Optional[ModelVersion]:
        """Get the currently active model version for a given type and horizon."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            """SELECT version_id, model_type, horizon, created_at, params, metrics,
                      artifact_path, is_active, config
               FROM model_versions
               WHERE model_type = ? AND horizon = ? AND is_active = 1
               ORDER BY created_at DESC LIMIT 1""",
            (model_type, horizon),
        ).fetchone()
        conn.close()

        if row is None:
            return None
        return self._row_to_version(row)

    def activate_version(self, version_id: str) -> None:
        """Set a version as the active model (deactivate others of same type/horizon)."""
        conn = sqlite3.connect(self.db_path)

        # Get model_type and horizon for this version
        row = conn.execute(
            "SELECT model_type, horizon FROM model_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"Version not found: {version_id}")

        model_type, horizon = row

        # Deactivate all of same type/horizon
        conn.execute(
            "UPDATE model_versions SET is_active = 0 WHERE model_type = ? AND horizon = ?",
            (model_type, horizon),
        )
        # Activate the requested version
        conn.execute(
            "UPDATE model_versions SET is_active = 1 WHERE version_id = ?",
            (version_id,),
        )
        conn.commit()
        conn.close()
        logger.info(f"Activated model version: {version_id}")

    def list_versions(
        self,
        model_type: Optional[str] = None,
        horizon: Optional[str] = None,
    ) -> List[ModelVersion]:
        """List all model versions, optionally filtered by type and horizon."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT version_id, model_type, horizon, created_at, params, metrics, artifact_path, is_active, config FROM model_versions"
        conditions = []
        params_list = []  # type: List[str]

        if model_type is not None:
            conditions.append("model_type = ?")
            params_list.append(model_type)
        if horizon is not None:
            conditions.append("horizon = ?")
            params_list.append(horizon)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        rows = conn.execute(query, params_list).fetchall()
        conn.close()

        return [self._row_to_version(r) for r in rows]

    def get_best_model(
        self,
        horizon: Optional[str] = None,
        metric: str = "mean_ic",
    ) -> Optional[ModelVersion]:
        """Get the best model version by a specific metric.

        Args:
            horizon: filter by horizon (e.g. "1M"). If None, searches all.
            metric: metric key in the metrics JSON to maximize.

        Returns:
            ModelVersion with the highest metric value, or None.
        """
        versions = self.list_versions(horizon=horizon)
        if not versions:
            return None

        best = None
        best_val = float("-inf")
        for v in versions:
            val = v.metrics.get(metric, float("-inf"))
            if val > best_val:
                best_val = val
                best = v
        return best

    def select_best_model(
        self,
        horizon: Optional[str] = None,
        criteria: str = "robust",
    ) -> Optional[ModelVersion]:
        """Select best model using a composite robustness score.

        Composite score balances predictive power with robustness:
          score = IC_mean + 0.5 * ICIR + 0.3 * Sharpe
                  - 0.5 * overfitting_score - 0.3 * stress_drawdown

        Args:
            horizon: Filter by horizon (e.g. "1M"). If None, searches all.
            criteria: Selection criteria. Currently only "robust" is supported.

        Returns:
            ModelVersion with the highest composite score, or None.
        """
        versions = self.list_versions(horizon=horizon)
        if not versions:
            return None

        best = None
        best_score = float("-inf")

        for v in versions:
            m = v.metrics
            ic_mean = m.get("mean_ic", 0.0)
            icir = m.get("icir", 0.0)
            sharpe = m.get("mean_sharpe", 0.0)
            overfit = m.get("overfitting_score", 0.5)
            stress_dd = abs(m.get("stress_max_drawdown", 0.3))

            score = (
                ic_mean
                + 0.5 * icir
                + 0.3 * sharpe
                - 0.5 * overfit
                - 0.3 * stress_dd
            )

            if score > best_score:
                best_score = score
                best = v

        if best is not None:
            logger.info("Selected best model: %s (composite=%.4f)",
                        best.version_id, best_score)
        return best

    @staticmethod
    def _row_to_version(row: tuple) -> ModelVersion:
        config_data = None
        if len(row) > 8 and row[8]:
            try:
                config_data = json.loads(row[8])
            except (json.JSONDecodeError, TypeError):
                config_data = None
        return ModelVersion(
            version_id=row[0],
            model_type=row[1],
            horizon=row[2],
            created_at=row[3],
            params=json.loads(row[4]) if row[4] else {},
            metrics=json.loads(row[5]) if row[5] else {},
            artifact_path=row[6],
            is_active=bool(row[7]),
            config=config_data,
        )

    @staticmethod
    def _save_pickle(model: Any, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def _load_pickle(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _save_torch(model: Any, path: str) -> None:
        try:
            import torch
            # If model has a .save() method (TransformerGaussianModel), use it
            if hasattr(model, "save"):
                model.save(path)
            elif hasattr(model, "net") and model.net is not None:
                state = {
                    "params": getattr(model, "params", {}),
                    "net_state": model.net.state_dict(),
                    "feature_names": getattr(model, "feature_names", []),
                    "is_fitted": getattr(model, "is_fitted", False),
                }
                # Save scaler if present
                if hasattr(model, "scaler") and hasattr(model.scaler, "mean_"):
                    state["scaler_mean"] = model.scaler.mean_.tolist()
                    state["scaler_scale"] = model.scaler.scale_.tolist()
                torch.save(state, path)
            else:
                torch.save(model, path)
        except ImportError:
            # Fallback to pickle if torch not available
            ModelRegistry._save_pickle(model, path)

    @staticmethod
    def _load_torch(path: str) -> Any:
        try:
            import torch
            return torch.load(path, map_location="cpu")
        except ImportError:
            return ModelRegistry._load_pickle(path)
