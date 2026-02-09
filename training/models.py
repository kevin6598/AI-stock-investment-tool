"""
Model Implementations
---------------------
Four model wrappers with a unified interface:
  1. ElasticNetModel - Regularized linear regression (L1 + L2)
  2. LightGBMModel  - Gradient boosted decision trees
  3. LSTMAttentionModel - LSTM with temporal attention (TFT-lite)
  4. TransformerModel - Encoder-only Transformer with Gaussian output

Each model implements:
  - fit(X_train, y_train, X_val, y_val)
  - predict(X) → point estimates
  - predict_quantiles(X, quantiles) → quantile forecasts
  - get_feature_importance() → dict
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class BaseModel(ABC):
    """Unified interface for all forecasting models."""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_quantiles(self, X: np.ndarray,
                          quantiles: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        pass


# ---------------------------------------------------------------------------
# 1. Elastic Net
# ---------------------------------------------------------------------------

class ElasticNetModel(BaseModel):
    """Regularized linear regression with L1 + L2 penalty.

    Default hyperparameters are conservative for financial data:
      alpha=0.1 (moderate regularization)
      l1_ratio=0.5 (equal L1/L2 mix)
    """

    def __init__(self, params: Optional[Dict] = None):
        defaults = {
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "max_iter": 5000,
            "tol": 1e-4,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(merged)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self._residual_std = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # Standardize features
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = ElasticNet(
            alpha=self.params["alpha"],
            l1_ratio=self.params["l1_ratio"],
            max_iter=self.params["max_iter"],
            tol=self.params["tol"],
            random_state=42,
        )
        self.model.fit(X_scaled, y_train)

        # Store residual std for quantile estimation
        preds = self.model.predict(X_scaled)
        residuals = y_train - preds
        self._residual_std = max(np.std(residuals), 1e-8)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_quantiles(self, X: np.ndarray,
                          quantiles: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        point = self.predict(X)
        from scipy.stats import norm
        result = {}
        for q in quantiles:
            z = norm.ppf(q)
            result[q] = point + z * self._residual_std
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        coefs = np.abs(self.model.coef_)
        total = coefs.sum()
        if total == 0:
            return {n: 0.0 for n in self.feature_names}
        return {n: float(c / total) for n, c in zip(self.feature_names, coefs)}


# ---------------------------------------------------------------------------
# 2. LightGBM
# ---------------------------------------------------------------------------

class LightGBMModel(BaseModel):
    """Gradient boosted decision trees via LightGBM.

    Default hyperparameters are conservative for small financial datasets:
      num_leaves=31, max_depth=6, min_child_samples=100
    """

    def __init__(self, params: Optional[Dict] = None):
        defaults = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "verbose": -1,
            "random_state": 42,
            "early_stopping_rounds": 50,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(merged)
        self.model = None
        self.feature_names = []
        self._quantile_models = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        fit_params = {}
        callbacks = []

        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            early = self.params.get("early_stopping_rounds", 50)
            if early:
                callbacks.append(lgb.early_stopping(early, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

        model_params = {k: v for k, v in self.params.items()
                        if k != "early_stopping_rounds"}

        self.model = lgb.LGBMRegressor(**model_params)
        self.model.fit(
            X_train, y_train,
            feature_name=self.feature_names,
            callbacks=callbacks if callbacks else None,
            **fit_params,
        )

        # Train quantile models for distributional output
        self._quantile_models = {}
        for q in [0.10, 0.50, 0.90]:
            q_params = {k: v for k, v in model_params.items()}
            q_params["objective"] = "quantile"
            q_params["alpha"] = q
            q_params["n_estimators"] = min(model_params.get("n_estimators", 500), 200)
            q_model = lgb.LGBMRegressor(**q_params)
            q_model.fit(
                X_train, y_train,
                feature_name=self.feature_names,
                callbacks=callbacks if callbacks else None,
                **fit_params,
            )
            self._quantile_models[q] = q_model

        self.is_fitted = True

    def _as_df(self, X: np.ndarray) -> pd.DataFrame:
        """Wrap numpy array with stored feature names for LightGBM."""
        if self.feature_names:
            return pd.DataFrame(X, columns=self.feature_names)
        return pd.DataFrame(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._as_df(X))

    def predict_quantiles(self, X: np.ndarray,
                          quantiles: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        X_df = self._as_df(X)
        result = {}
        for q in quantiles:
            if q in self._quantile_models:
                result[q] = self._quantile_models[q].predict(X_df)
            else:
                # Interpolate from trained quantile models
                trained_qs = sorted(self._quantile_models.keys())
                if q < trained_qs[0]:
                    low_pred = self._quantile_models[trained_qs[0]].predict(X_df)
                    mid_pred = self._quantile_models[0.50].predict(X_df)
                    result[q] = low_pred - (mid_pred - low_pred) * 0.5
                elif q > trained_qs[-1]:
                    high_pred = self._quantile_models[trained_qs[-1]].predict(X_df)
                    mid_pred = self._quantile_models[0.50].predict(X_df)
                    result[q] = high_pred + (high_pred - mid_pred) * 0.5
                else:
                    # Linear interpolation between nearest trained quantiles
                    lower_q = max(tq for tq in trained_qs if tq <= q)
                    upper_q = min(tq for tq in trained_qs if tq >= q)
                    if lower_q == upper_q:
                        result[q] = self._quantile_models[lower_q].predict(X_df)
                    else:
                        w = (q - lower_q) / (upper_q - lower_q)
                        low_pred = self._quantile_models[lower_q].predict(X_df)
                        high_pred = self._quantile_models[upper_q].predict(X_df)
                        result[q] = low_pred * (1 - w) + high_pred * w
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        total = importance.sum()
        if total == 0:
            return {n: 0.0 for n in self.feature_names}
        return {n: float(v / total) for n, v in zip(self.feature_names, importance)}


# ---------------------------------------------------------------------------
# 3. LSTM with Temporal Attention (TFT-lite)
# ---------------------------------------------------------------------------

class LSTMAttentionModel(BaseModel):
    """LSTM encoder with multi-head temporal attention and quantile output.

    A practical deep learning model for financial time-series that captures:
      - Sequential dependencies via LSTM
      - Temporal attention to learn which past timesteps matter
      - Quantile outputs for distributional forecasting

    Requires data in 3D format: (samples, sequence_length, features).
    The fit() method handles reshaping from 2D panel data.
    """

    def __init__(self, params: Optional[Dict] = None):
        defaults = {
            "hidden_size": 64,
            "num_layers": 2,
            "attention_heads": 2,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 100,
            "patience": 15,
            "sequence_length": 60,
            "quantiles": [0.10, 0.50, 0.90],
        }
        merged = {**defaults, **(params or {})}
        super().__init__(merged)
        self.net = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self._input_size = 0
        self._device = "cpu"

    def _build_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None
                         ) -> Tuple:
        """Convert 2D panel data to 3D sequences for LSTM.

        Assumes X rows are ordered chronologically per entity.
        Creates sliding windows of length sequence_length.
        """
        seq_len = self.params["sequence_length"]
        n_samples = X.shape[0]

        if n_samples <= seq_len:
            # Not enough data for sequences; pad
            pad_len = seq_len - n_samples + 1
            X_padded = np.vstack([np.zeros((pad_len, X.shape[1])), X])
            if y is not None:
                y_padded = np.concatenate([np.zeros(pad_len), y])
            n_samples = X_padded.shape[0]
        else:
            X_padded = X
            y_padded = y

        X_seq = []
        y_seq = []
        for i in range(seq_len, n_samples):
            X_seq.append(X_padded[i - seq_len:i])
            if y_padded is not None:
                y_seq.append(y_padded[i])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32) if y is not None else None
        return X_seq, y_seq

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._input_size = X_train.shape[1]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_seq, y_train_seq = self._build_sequences(X_train_scaled, y_train)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._build_sequences(X_val_scaled, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        # Build network
        quantiles = self.params["quantiles"]
        self.net = _LSTMAttentionNet(
            input_size=self._input_size,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            attention_heads=self.params["attention_heads"],
            dropout=self.params["dropout"],
            num_quantiles=len(quantiles),
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.params["epochs"], eta_min=1e-6,
        )

        # Data loaders
        train_ds = TensorDataset(
            torch.from_numpy(X_train_seq),
            torch.from_numpy(y_train_seq),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.params["batch_size"], shuffle=True,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.params["epochs"]):
            self.net.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                q_preds = self.net(X_batch)  # (batch, num_quantiles)
                loss = _quantile_loss(q_preds, y_batch, quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            scheduler.step()

            # Validation
            if X_val_seq is not None:
                self.net.eval()
                with torch.no_grad():
                    X_v = torch.from_numpy(X_val_seq).to(self._device)
                    y_v = torch.from_numpy(y_val_seq).to(self._device)
                    q_v = self.net(X_v)
                    val_loss = _quantile_loss(q_v, y_v, quantiles).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    self._best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.params["patience"]:
                        break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.net.load_state_dict(self._best_state)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._build_sequences(X_scaled)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self._device)
            q_preds = self.net(X_t).cpu().numpy()

        # Return median (index 1 = 0.50 quantile)
        median_idx = len(self.params["quantiles"]) // 2
        preds = q_preds[:, median_idx]

        # Pad front to match input length
        pad_len = X.shape[0] - len(preds)
        if pad_len > 0:
            preds = np.concatenate([np.full(pad_len, np.nan), preds])
        return preds

    def predict_quantiles(self, X: np.ndarray,
                          quantiles: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        import torch
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._build_sequences(X_scaled)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self._device)
            q_preds = self.net(X_t).cpu().numpy()

        pad_len = X.shape[0] - q_preds.shape[0]
        trained_qs = self.params["quantiles"]

        result = {}
        if quantiles is None:
            quantiles = trained_qs

        for q in quantiles:
            if q in trained_qs:
                idx = trained_qs.index(q)
                vals = q_preds[:, idx]
            else:
                # Interpolate
                lower_qs = [tq for tq in trained_qs if tq <= q]
                upper_qs = [tq for tq in trained_qs if tq >= q]
                if not lower_qs:
                    vals = q_preds[:, 0]
                elif not upper_qs:
                    vals = q_preds[:, -1]
                else:
                    lq = max(lower_qs)
                    uq = min(upper_qs)
                    li = trained_qs.index(lq)
                    ui = trained_qs.index(uq)
                    w = (q - lq) / (uq - lq) if uq != lq else 0.5
                    vals = q_preds[:, li] * (1 - w) + q_preds[:, ui] * w

            if pad_len > 0:
                vals = np.concatenate([np.full(pad_len, np.nan), vals])
            result[q] = vals

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        # For LSTM, return uniform importance (no native feature importance)
        if not self.feature_names:
            return {}
        n = len(self.feature_names)
        return {name: 1.0 / n for name in self.feature_names}


# ---------------------------------------------------------------------------
# PyTorch modules for LSTM-Attention
# ---------------------------------------------------------------------------

def _quantile_loss(predictions, targets, quantiles):
    """Pinball loss for quantile regression."""
    import torch
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - predictions[:, i]
        losses.append(torch.max(q * errors, (q - 1) * errors).mean())
    return sum(losses) / len(losses)


class _LSTMAttentionNet:
    """This is lazily imported to avoid torch dependency at module level."""
    pass


# Rebuild class properly only when torch is available
try:
    import torch
    import torch.nn as nn

    class _TemporalAttention(nn.Module):
        """Multi-head attention over LSTM hidden states."""

        def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            # x: (batch, seq_len, hidden_size)
            attn_out, attn_weights = self.attention(x, x, x)
            return self.norm(x + attn_out), attn_weights

    class _LSTMAttentionNet(nn.Module):
        """LSTM encoder + temporal attention + quantile output heads."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     attention_heads: int, dropout: float, num_quantiles: int):
            super().__init__()

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # LSTM encoder
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # Temporal attention
            self.attention = _TemporalAttention(hidden_size, attention_heads, dropout)

            # Output: one head per quantile
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_quantiles),
            )

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            x = self.input_proj(x)
            lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
            attn_out, _ = self.attention(lstm_out)  # (batch, seq_len, hidden)
            # Use last timestep
            last = attn_out[:, -1, :]           # (batch, hidden)
            return self.output(last)            # (batch, num_quantiles)

except ImportError:
    pass  # torch not installed; LSTMAttentionModel.fit() will fail with clear error


# ---------------------------------------------------------------------------
# 4. Transformer (wraps models.transformer_ts.TransformerGaussianModel)
# ---------------------------------------------------------------------------

class TransformerModel(BaseModel):
    """Encoder-only Transformer with Gaussian output heads.

    Wraps the standalone TransformerGaussianModel to conform to the BaseModel
    interface used by the training pipeline. Gaussian mean is used for point
    estimates; mean +/- z*std for quantile estimates.
    """

    def __init__(self, params: Optional[Dict] = None):
        defaults = {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 256,
            "dropout": 0.2,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "epochs": 100,
            "patience": 15,
            "sequence_length": 60,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(merged)
        self._inner = None  # TransformerGaussianModel instance
        self.feature_names = []  # type: List[str]
        self._residual_std = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        from models.transformer_ts import TransformerGaussianModel

        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # Map BaseModel params to TransformerGaussianModel config
        config = {
            "d_model": self.params["d_model"],
            "n_heads": self.params["n_heads"],
            "n_layers": self.params["n_layers"],
            "d_ff": self.params["d_ff"],
            "dropout": self.params["dropout"],
            "learning_rate": self.params["learning_rate"],
            "batch_size": self.params["batch_size"],
            "epochs": self.params["epochs"],
            "patience": self.params["patience"],
            "sequence_length": self.params["sequence_length"],
            "horizons": [1],  # single horizon -- pipeline handles multi-horizon
        }
        self._inner = TransformerGaussianModel(config)
        self._inner.fit(X_train, y_train, X_val, y_val, feature_names=self.feature_names)

        # Compute residual std from training data for quantile estimation
        if self._inner.is_fitted:
            means, variances = self._inner.predict(X_train)
            if means.shape[0] > 0:
                # means shape: (n_valid, 1) -- take first horizon
                preds = means[:, 0]
                # Align with y_train (predict drops first seq_len samples)
                offset = len(y_train) - len(preds)
                if offset >= 0:
                    residuals = y_train[offset:] - preds
                    self._residual_std = max(float(np.std(residuals)), 1e-8)

        self.is_fitted = self._inner.is_fitted

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._inner is None or not self.is_fitted:
            raise RuntimeError("Model not fitted")

        means, variances = self._inner.predict(X)
        if means.shape[0] == 0:
            return np.full(X.shape[0], np.nan)

        # means shape: (n_valid, n_horizons) -- take first horizon
        preds = means[:, 0]

        # Pad front to match input length (same pattern as LSTMAttentionModel)
        pad_len = X.shape[0] - len(preds)
        if pad_len > 0:
            preds = np.concatenate([np.full(pad_len, np.nan), preds])
        return preds

    def predict_quantiles(self, X: np.ndarray,
                          quantiles: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        if self._inner is None or not self.is_fitted:
            raise RuntimeError("Model not fitted")

        means, variances = self._inner.predict(X)
        if means.shape[0] == 0:
            return {q: np.full(X.shape[0], np.nan) for q in quantiles}

        point = means[:, 0]
        # Use model variance if available, fall back to residual std
        if variances.shape[0] > 0:
            std = np.sqrt(variances[:, 0])
        else:
            std = np.full_like(point, self._residual_std)

        from scipy.stats import norm
        result = {}
        pad_len = X.shape[0] - len(point)
        for q in quantiles:
            z = norm.ppf(q)
            vals = point + z * std
            if pad_len > 0:
                vals = np.concatenate([np.full(pad_len, np.nan), vals])
            result[q] = vals
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        # Transformer has no native feature importance; return uniform weights
        if not self.feature_names:
            return {}
        n = len(self.feature_names)
        return {name: 1.0 / n for name in self.feature_names}


# ---------------------------------------------------------------------------
# 5. Hybrid Multi-Modal Model
# ---------------------------------------------------------------------------

class HybridMultiModalModel(BaseModel):
    """Hybrid multi-modal deep learning model wrapping HybridMultiModalNet.

    Combines temporal encoding (LSTM), technical indicator embeddings,
    VAE latent factors, sentiment encoding, and ticker embeddings
    via a multi-modal fusion engine.

    Requires PyTorch. Data must include sequence context.
    """

    def __init__(self, params: Optional[Dict] = None):
        defaults = {
            "hidden_dim": 128,
            "fusion_dim": 128,
            "vae_latent_dim": 16,
            "n_quantiles": 7,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 100,
            "patience": 15,
            "sequence_length": 60,
            "n_tickers": 50,
            "sentiment_dim": 7,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(merged)
        self.net = None
        self.scaler = StandardScaler()
        self.feature_names = []  # type: List[str]
        self._input_size = 0
        self._device = "cpu"
        self._quantiles_list = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    def _build_sequences(self, X, y=None):
        seq_len = self.params["sequence_length"]
        n = X.shape[0]
        if n <= seq_len:
            pad_len = seq_len - n + 1
            X = np.vstack([np.zeros((pad_len, X.shape[1])), X])
            if y is not None:
                y = np.concatenate([np.zeros(pad_len), y])
            n = X.shape[0]
        X_seq, y_seq = [], []
        for i in range(seq_len, n):
            X_seq.append(X[i - seq_len:i])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32) if y is not None else None
        return X_seq, y_seq

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from models.hybrid_model import HybridMultiModalNet

        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._input_size = X_train.shape[1]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_seq, y_train_seq = self._build_sequences(X_train_scaled, y_train)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._build_sequences(X_val_scaled, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        self.net = HybridMultiModalNet(
            n_features=self._input_size,
            n_tickers=self.params["n_tickers"],
            seq_len=self.params["sequence_length"],
            hidden_dim=self.params["hidden_dim"],
            vae_latent_dim=self.params["vae_latent_dim"],
            sentiment_dim=self.params["sentiment_dim"],
            fusion_dim=self.params["fusion_dim"],
            n_quantiles=self.params["n_quantiles"],
            dropout=self.params["dropout"],
        ).to(self._device)

        from models.losses import MultiTaskLoss
        from models.vae import VAELoss
        criterion = MultiTaskLoss(quantiles=self._quantiles_list)
        vae_criterion = VAELoss(beta_max=0.5, warmup_steps=500)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.params["epochs"], eta_min=1e-6,
        )

        train_ds = TensorDataset(
            torch.from_numpy(X_train_seq),
            torch.from_numpy(y_train_seq),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.params["batch_size"], shuffle=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.params["epochs"]):
            self.net.train()
            for X_b, y_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                batch_size = X_b.size(0)

                x_static = X_b[:, -1, :]
                ticker_ids = torch.zeros(batch_size, dtype=torch.long, device=self._device)

                optimizer.zero_grad()
                out = self.net(X_b, x_static, ticker_ids)

                task_loss = criterion(
                    out["quantiles"], out["p_up"],
                    out["quantiles"][:, 3],  # median as point estimate
                    y_b,
                )
                vae_loss = vae_criterion(
                    out["vae_reconstruction"], x_static,
                    out["vae_mu"], out["vae_log_var"],
                )
                total_loss = task_loss["total"] + 0.1 * vae_loss["total"]
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            if X_val_seq is not None:
                self.net.eval()
                with torch.no_grad():
                    Xv = torch.from_numpy(X_val_seq).to(self._device)
                    yv = torch.from_numpy(y_val_seq).to(self._device)
                    xv_static = Xv[:, -1, :]
                    tv_ids = torch.zeros(Xv.size(0), dtype=torch.long, device=self._device)
                    v_out = self.net(Xv, xv_static, tv_ids)
                    v_task = criterion(
                        v_out["quantiles"], v_out["p_up"],
                        v_out["quantiles"][:, 3], yv,
                    )
                    val_loss = v_task["total"].item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.params["patience"]:
                        break

        if hasattr(self, "_best_state"):
            self.net.load_state_dict(self._best_state)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._build_sequences(X_scaled)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self._device)
            x_static = X_t[:, -1, :]
            ticker_ids = torch.zeros(X_t.size(0), dtype=torch.long, device=self._device)
            out = self.net(X_t, x_static, ticker_ids)
            preds = out["quantiles"][:, 3].cpu().numpy()  # median

        pad_len = X.shape[0] - len(preds)
        if pad_len > 0:
            preds = np.concatenate([np.full(pad_len, np.nan), preds])
        return preds

    def predict_quantiles(self, X: np.ndarray,
                          quantiles: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        import torch
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._build_sequences(X_scaled)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self._device)
            x_static = X_t[:, -1, :]
            ticker_ids = torch.zeros(X_t.size(0), dtype=torch.long, device=self._device)
            out = self.net(X_t, x_static, ticker_ids)
            q_preds = out["quantiles"].cpu().numpy()

        pad_len = X.shape[0] - q_preds.shape[0]
        trained_qs = self._quantiles_list

        if quantiles is None:
            quantiles = trained_qs

        result = {}
        for q in quantiles:
            if q in trained_qs:
                idx = trained_qs.index(q)
                vals = q_preds[:, idx]
            else:
                lower_qs = [tq for tq in trained_qs if tq <= q]
                upper_qs = [tq for tq in trained_qs if tq >= q]
                if not lower_qs:
                    vals = q_preds[:, 0]
                elif not upper_qs:
                    vals = q_preds[:, -1]
                else:
                    lq, uq = max(lower_qs), min(upper_qs)
                    li, ui = trained_qs.index(lq), trained_qs.index(uq)
                    w = (q - lq) / (uq - lq) if uq != lq else 0.5
                    vals = q_preds[:, li] * (1 - w) + q_preds[:, ui] * w

            if pad_len > 0:
                vals = np.concatenate([np.full(pad_len, np.nan), vals])
            result[q] = vals
        return result

    def predict_full(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict with all output heads (retail + auxiliary)."""
        import torch
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._build_sequences(X_scaled)

        self.net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self._device)
            x_static = X_t[:, -1, :]
            ticker_ids = torch.zeros(X_t.size(0), dtype=torch.long, device=self._device)
            out = self.net(X_t, x_static, ticker_ids)

        result = {}
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                result[key] = val.cpu().numpy()
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.feature_names:
            return {}
        n = len(self.feature_names)
        return {name: 1.0 / n for name in self.feature_names}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "elastic_net": ElasticNetModel,
    "lightgbm": LightGBMModel,
    "lstm_attention": LSTMAttentionModel,
    "transformer": TransformerModel,
    "hybrid_multimodal": HybridMultiModalModel,
}


def create_model(model_type: str, params: Optional[Dict] = None) -> BaseModel:
    """Create a model instance by name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](params)
