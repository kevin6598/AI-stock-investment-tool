"""
Multi-Strategy Benchmark Framework
-----------------------------------
Compare alpha extraction strategies under identical walk-forward conditions.

Strategies:
  A - LSTM Baseline (price-only temporal)
  B - NLP Only (sentiment MLP)
  C - Late Ensemble (A + B with ridge)
  D - Residual Sentiment (market-residualized NLP)
  E - Gated Hybrid (gated price-NLP fusion)
  F - Cross-Sectional Attention LSTM (temporal + stock attention)
  G - Short Horizon NLP (5D sentiment)

Usage:
    from training.strategy_benchmark import BenchmarkEvaluator, STRATEGY_REGISTRY
    evaluator = BenchmarkEvaluator(panel, feature_cols, config)
    results = evaluator.run_all(STRATEGY_REGISTRY.values())
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging
import math
import time
import json
import os

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Global benchmark configuration."""
    horizons: List[int] = field(default_factory=lambda: [5, 21])
    train_years: int = 3
    test_months: int = 6
    step_months: int = 6
    val_months: int = 3
    embargo_days: int = 5
    max_epochs: int = 15
    early_stop_patience: int = 3
    ranking_weight: float = 0.5
    max_params: int = 1_500_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    sequence_length: int = 20
    min_stocks_for_ic: int = 10

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkConfig":
        wf = d.get("walk_forward", {})
        return cls(
            horizons=d.get("horizons", [5, 21]),
            train_years=wf.get("train_years", 3),
            test_months=wf.get("test_months", 6),
            step_months=wf.get("step_months", 6),
            embargo_days=wf.get("embargo_days", 5),
            max_epochs=d.get("max_epochs", 15),
            early_stop_patience=d.get("early_stop_patience", 3),
            ranking_weight=d.get("ranking_weight", 0.5),
            max_params=d.get("max_params", 1_500_000),
        )


@dataclass
class StrategyData:
    """Standardized data container for strategy evaluation."""
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    dates: np.ndarray
    tickers: np.ndarray


@dataclass
class FoldMetrics:
    """Metrics from a single walk-forward fold."""
    fold_idx: int
    ic: float
    ic_std: float
    sharpe: float
    max_drawdown: float
    n_samples: int
    n_dates: int
    pred_std: float
    train_ic: float = 0.0


@dataclass
class StrategyResult:
    """Full benchmark result for one strategy-horizon pair."""
    name: str
    horizon: str
    horizon_days: int
    ic_mean: float
    ic_std: float
    icir: float
    sharpe: float
    max_drawdown: float
    overfit_score: float
    composite: float
    fold_metrics: List[FoldMetrics]
    prod_ic: float
    param_count: int
    train_time: float
    status: str = "PENDING"
    gate_stats: Optional[Dict[str, float]] = None
    ensemble_weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        d = {
            "name": self.name,
            "horizon": self.horizon,
            "horizon_days": self.horizon_days,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "overfit_score": self.overfit_score,
            "composite": self.composite,
            "prod_ic": self.prod_ic,
            "param_count": self.param_count,
            "train_time": round(self.train_time, 1),
            "status": self.status,
            "fold_ics": [f.ic for f in self.fold_metrics],
        }
        if self.gate_stats:
            d["gate_stats"] = self.gate_stats
        if self.ensemble_weights:
            d["ensemble_weights"] = self.ensemble_weights
        return d


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_cross_sectional_ic(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: np.ndarray,
    min_stocks: int = 10,
) -> Tuple[float, float]:
    """Cross-sectional Spearman IC: per-date rank correlation, then average.

    Returns:
        (ic_mean, ic_std) tuple.
    """
    from scipy.stats import spearmanr

    unique_dates = np.unique(dates)
    ics = []  # type: List[float]

    for d in unique_dates:
        mask = dates == d
        if mask.sum() < min_stocks:
            continue
        yp = y_pred[mask]
        yt = y_true[mask]

        valid = ~(np.isnan(yp) | np.isnan(yt))
        if valid.sum() < min_stocks:
            continue
        if np.std(yp[valid]) < 1e-8 or np.std(yt[valid]) < 1e-8:
            continue

        ic, _ = spearmanr(yp[valid], yt[valid])
        if not np.isnan(ic):
            ics.append(float(ic))

    if not ics:
        return 0.0, 0.0
    return float(np.mean(ics)), float(np.std(ics))


def compute_simple_investment_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> Tuple[float, float]:
    """Compute annualized Sharpe ratio and max drawdown from long-short signal.

    Returns:
        (sharpe, max_drawdown) tuple.
    """
    valid = ~(np.isnan(predictions) | np.isnan(actuals))
    if valid.sum() < 10:
        return 0.0, 0.0

    preds = predictions[valid]
    actual = actuals[valid]

    signal = np.where(preds > 0, 1.0, -1.0)
    returns = signal * actual

    if len(returns) > 1 and np.std(returns) > 1e-10:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    drawdowns = peak - cum
    mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return sharpe, mdd


def _nan_safe(X: np.ndarray) -> np.ndarray:
    """In-place NaN/Inf cleanup, returns the same array."""
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ---------------------------------------------------------------------------
# PyTorch network definitions (conditional)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _NLPMlpNet(nn.Module):
        """Shallow 2-layer MLP for NLP features."""

        def __init__(self, input_dim: int, hidden: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class _CSAttentionNet(nn.Module):
        """LSTM encoder + single-head cross-sectional self-attention."""

        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers=2,
                batch_first=True, dropout=0.2,
            )
            self.ln = nn.LayerNorm(hidden_dim)
            self.attn = nn.MultiheadAttention(
                hidden_dim, num_heads=1, batch_first=True,
            )
            self.attn_ln = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.lstm(x)
            h = self.ln(out[:, -1, :])  # (batch, hidden)

            # Cross-sectional attention across stocks in the batch
            h_3d = h.unsqueeze(0)  # (1, n_stocks, hidden)
            h_attn, _ = self.attn(h_3d, h_3d, h_3d)
            h_attn = h_attn.squeeze(0)  # (batch, hidden)
            h = self.attn_ln(h + h_attn)

            return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class StrategyModel(ABC):
    """Base class for all benchmark strategies."""

    name = "base"
    supported_horizons = None  # None = all horizons

    @abstractmethod
    def train(self, train_data: StrategyData,
              val_data: Optional[StrategyData],
              config: BenchmarkConfig) -> None:
        ...

    @abstractmethod
    def predict(self, data: StrategyData) -> np.ndarray:
        ...

    @abstractmethod
    def num_parameters(self) -> int:
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Helper: train a simple MLP with early stopping
# ---------------------------------------------------------------------------

def _train_mlp_loop(net, X_train, y_train, X_val, y_val, config):
    """Train a PyTorch MLP with early stopping. Operates in-place on net."""
    if not HAS_TORCH:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    dataset = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    X_val_t, y_val_t = None, None
    if X_val is not None and y_val is not None and len(y_val) > 20:
        X_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf")
    patience_ctr = 0

    net.train()
    for epoch in range(config.max_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        if X_val_t is not None:
            net.eval()
            with torch.no_grad():
                vl = criterion(net(X_val_t), y_val_t).item()
            net.train()
            if vl < best_val:
                best_val = vl
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.early_stop_patience:
                    break

    net.eval()


def _predict_mlp(net, X, nlp_idx):
    """Run MLP prediction."""
    if net is None or not nlp_idx:
        return np.zeros(len(X))

    device = next(net.parameters()).device
    X_nlp = _nan_safe(X[:, nlp_idx].astype(np.float32))

    net.eval()
    with torch.no_grad():
        preds = net(torch.from_numpy(X_nlp).to(device)).cpu().numpy()
    return preds


# ---------------------------------------------------------------------------
# Helper: build sequences for LSTM
# ---------------------------------------------------------------------------

def _build_sequences(X, seq_len):
    """Convert 2D array to 3D sliding-window sequences."""
    n = X.shape[0]
    if n <= seq_len:
        pad_len = seq_len - n + 1
        X = np.vstack([np.zeros((pad_len, X.shape[1]), dtype=np.float32), X])
        n = X.shape[0]

    seqs = []
    for i in range(seq_len, n):
        seqs.append(X[i - seq_len:i])
    return np.array(seqs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Strategy A: LSTM Baseline (price-only)
# ---------------------------------------------------------------------------

class LSTMBaselineStrategy(StrategyModel):
    """Temporal LSTM on price features only."""

    name = "A_LSTM_Baseline"

    def __init__(self):
        self._model = None
        self._price_idx = []  # type: List[int]
        self._price_names = []  # type: List[str]

    def train(self, train_data, val_data, config):
        from training.models import create_model

        self._price_idx = [i for i, n in enumerate(train_data.feature_names)
                           if not n.startswith("nlp_")]
        self._price_names = [train_data.feature_names[i] for i in self._price_idx]

        X_tr = _nan_safe(train_data.X[:, self._price_idx].astype(np.float32))
        y_tr = _nan_safe(train_data.y.astype(np.float32))

        X_v, y_v = None, None
        if val_data is not None:
            X_v = _nan_safe(val_data.X[:, self._price_idx].astype(np.float32))
            y_v = _nan_safe(val_data.y.astype(np.float32))

        self._model = create_model("lstm_attention", {
            "epochs": config.max_epochs,
            "patience": config.early_stop_patience,
            "sequence_length": config.sequence_length,
            "hidden_dim": config.hidden_dim,
            "dropout": 0.2,
            "learning_rate": config.learning_rate,
        })
        self._model.fit(X_tr, y_tr, X_v, y_v, feature_names=self._price_names)

    def predict(self, data):
        if self._model is None:
            return np.zeros(len(data.X))
        X = _nan_safe(data.X[:, self._price_idx].astype(np.float32))
        preds = self._model.predict(X)
        return _nan_safe(preds)

    def num_parameters(self):
        if self._model and hasattr(self._model, "net") and self._model.net:
            return sum(p.numel() for p in self._model.net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy B: NLP Only (shallow MLP)
# ---------------------------------------------------------------------------

class NLPOnlyStrategy(StrategyModel):
    """Shallow MLP on sentiment features only. No temporal structure."""

    name = "B_NLP_Only"

    def __init__(self):
        self._net = None
        self._nlp_idx = []  # type: List[int]

    def train(self, train_data, val_data, config):
        self._nlp_idx = [i for i, n in enumerate(train_data.feature_names)
                         if n.startswith("nlp_")]
        if not self._nlp_idx:
            logger.warning("%s: no nlp_* features found", self.name)
            return

        X_tr = _nan_safe(train_data.X[:, self._nlp_idx].astype(np.float32))
        y_tr = _nan_safe(train_data.y.astype(np.float32))

        X_v, y_v = None, None
        if val_data is not None:
            X_v = _nan_safe(val_data.X[:, self._nlp_idx].astype(np.float32))
            y_v = _nan_safe(val_data.y.astype(np.float32))

        self._net = _NLPMlpNet(len(self._nlp_idx))
        _train_mlp_loop(self._net, X_tr, y_tr, X_v, y_v, config)

    def predict(self, data):
        return _predict_mlp(self._net, data.X, self._nlp_idx)

    def num_parameters(self):
        if self._net:
            return sum(p.numel() for p in self._net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy C: Late Ensemble (LSTM + NLP with ridge)
# ---------------------------------------------------------------------------

class LateEnsembleStrategy(StrategyModel):
    """Train A and B independently, fit ridge on validation predictions."""

    name = "C_Late_Ensemble"

    def __init__(self):
        self._lstm = LSTMBaselineStrategy()
        self._nlp = NLPOnlyStrategy()
        self._weights = {"lstm": 0.5, "nlp": 0.5, "intercept": 0.0}

    def train(self, train_data, val_data, config):
        self._lstm.train(train_data, val_data, config)
        self._nlp.train(train_data, val_data, config)

        if val_data is not None and len(val_data.y) > 30:
            p_lstm = self._lstm.predict(val_data)
            p_nlp = self._nlp.predict(val_data)

            X_ens = np.column_stack([p_lstm, p_nlp])
            valid = ~(np.isnan(X_ens).any(axis=1) | np.isnan(val_data.y))

            if valid.sum() > 30:
                from sklearn.linear_model import Ridge
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_ens[valid], val_data.y[valid])
                self._weights = {
                    "lstm": float(ridge.coef_[0]),
                    "nlp": float(ridge.coef_[1]),
                    "intercept": float(ridge.intercept_),
                }

    def predict(self, data):
        p_l = self._lstm.predict(data)
        p_n = self._nlp.predict(data)
        w = self._weights
        return w["lstm"] * p_l + w["nlp"] * p_n + w["intercept"]

    def num_parameters(self):
        return self._lstm.num_parameters() + self._nlp.num_parameters() + 3

    def get_diagnostics(self):
        return {"ensemble_weights": self._weights}


# ---------------------------------------------------------------------------
# Strategy D: Residual Sentiment
# ---------------------------------------------------------------------------

class ResidualSentimentStrategy(StrategyModel):
    """NLP model on market-residualized sentiment features.

    sent_resid = sentiment - beta * market_return
    """

    name = "D_Residual_Sentiment"

    def __init__(self):
        self._net = None
        self._nlp_idx = []  # type: List[int]
        self._market_idx = None  # type: Optional[int]
        self._betas = None  # type: Optional[np.ndarray]

    @staticmethod
    def _find_market_idx(names):
        for i, n in enumerate(names):
            if n in ("market_return", "market_return_21d", "beta_market",
                     "spy_return_21d", "market_momentum"):
                return i
        return None

    def _residualize(self, X_nlp, market_ret):
        if self._betas is None or self._market_idx is None:
            return X_nlp
        out = X_nlp.copy()
        for j in range(out.shape[1]):
            out[:, j] -= self._betas[j] * market_ret
        return out

    def train(self, train_data, val_data, config):
        self._nlp_idx = [i for i, n in enumerate(train_data.feature_names)
                         if n.startswith("nlp_")]
        self._market_idx = self._find_market_idx(train_data.feature_names)

        if not self._nlp_idx:
            return

        X_nlp = _nan_safe(train_data.X[:, self._nlp_idx].astype(np.float32))

        # Compute betas and residualize
        if self._market_idx is not None:
            mr = _nan_safe(train_data.X[:, self._market_idx].astype(np.float32))
            mr_var = float(np.var(mr))
            if mr_var > 1e-10:
                self._betas = np.zeros(len(self._nlp_idx), dtype=np.float32)
                mr_mean = np.mean(mr)
                for j in range(len(self._nlp_idx)):
                    cov = np.mean(X_nlp[:, j] * mr) - np.mean(X_nlp[:, j]) * mr_mean
                    self._betas[j] = cov / mr_var
                X_nlp = self._residualize(X_nlp, mr)

        y_tr = _nan_safe(train_data.y.astype(np.float32))

        X_v, y_v = None, None
        if val_data is not None:
            X_v_nlp = _nan_safe(val_data.X[:, self._nlp_idx].astype(np.float32))
            if self._market_idx is not None and self._betas is not None:
                mr_v = _nan_safe(val_data.X[:, self._market_idx].astype(np.float32))
                X_v_nlp = self._residualize(X_v_nlp, mr_v)
            X_v = X_v_nlp
            y_v = _nan_safe(val_data.y.astype(np.float32))

        self._net = _NLPMlpNet(len(self._nlp_idx))
        _train_mlp_loop(self._net, X_nlp, y_tr, X_v, y_v, config)

    def predict(self, data):
        if self._net is None or not self._nlp_idx:
            return np.zeros(len(data.X))

        X_nlp = _nan_safe(data.X[:, self._nlp_idx].astype(np.float32))
        if self._market_idx is not None and self._betas is not None:
            mr = _nan_safe(data.X[:, self._market_idx].astype(np.float32))
            X_nlp = self._residualize(X_nlp, mr)

        device = next(self._net.parameters()).device
        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(X_nlp).to(device)).cpu().numpy()
        return preds

    def num_parameters(self):
        if self._net:
            return sum(p.numel() for p in self._net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy E: Gated Hybrid
# ---------------------------------------------------------------------------

class GatedHybridStrategy(StrategyModel):
    """Gated fusion of temporal LSTM and NLP features."""

    name = "E_Gated_Hybrid"

    def __init__(self):
        self._model = None

    def train(self, train_data, val_data, config):
        from training.models import create_model

        X_tr = _nan_safe(train_data.X.astype(np.float32))
        y_tr = _nan_safe(train_data.y.astype(np.float32))

        X_v, y_v = None, None
        if val_data is not None:
            X_v = _nan_safe(val_data.X.astype(np.float32))
            y_v = _nan_safe(val_data.y.astype(np.float32))

        self._model = create_model("gated_hybrid", {
            "epochs": config.max_epochs,
            "patience": config.early_stop_patience,
            "sequence_length": config.sequence_length,
            "temporal_hidden": config.hidden_dim,
            "nlp_embed": 32,
            "dropout": 0.2,
            "learning_rate": config.learning_rate,
            "ranking_weight": config.ranking_weight,
            "batch_size": config.batch_size,
        })
        self._model.fit(X_tr, y_tr, X_v, y_v,
                        feature_names=train_data.feature_names)

    def predict(self, data):
        if self._model is None:
            return np.zeros(len(data.X))
        X = _nan_safe(data.X.astype(np.float32))
        return _nan_safe(self._model.predict(X))

    def num_parameters(self):
        if self._model and hasattr(self._model, "net") and self._model.net:
            return sum(p.numel() for p in self._model.net.parameters())
        return 0

    def get_diagnostics(self):
        if self._model and hasattr(self._model, "get_gate_stats"):
            return {"gate_stats": self._model.get_gate_stats()}
        return {}


# ---------------------------------------------------------------------------
# Strategy F: Cross-Sectional Attention LSTM
# ---------------------------------------------------------------------------

class CrossSectionalLSTMStrategy(StrategyModel):
    """LSTM encoder + per-date cross-sectional self-attention.

    Architecture:
      price_features -> LSTM(hidden=64) -> h (per stock)
      h_batch -> SingleHeadAttention(across stocks) -> h2
      h2 -> Linear(64->32) -> ReLU -> Linear(32->1)
    """

    name = "F_CS_Attention_LSTM"

    def __init__(self):
        self._net = None
        self._price_idx = []  # type: List[int]
        self._scaler = None
        self._seq_len = 20

    def train(self, train_data, val_data, config):
        from sklearn.preprocessing import StandardScaler

        self._price_idx = [i for i, n in enumerate(train_data.feature_names)
                           if not n.startswith("nlp_")]
        self._seq_len = config.sequence_length
        n_feat = len(self._price_idx)

        X_price = _nan_safe(train_data.X[:, self._price_idx].astype(np.float32))
        y = _nan_safe(train_data.y.astype(np.float32))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_price).astype(np.float32)

        X_seq = _build_sequences(X_scaled, self._seq_len)
        y_seq = y[self._seq_len:] if len(y) > self._seq_len else y
        min_len = min(len(X_seq), len(y_seq))
        X_seq, y_seq = X_seq[:min_len], y_seq[:min_len]

        self._net = _CSAttentionNet(n_feat, config.hidden_dim)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._net.to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_seq).to(device),
            torch.from_numpy(y_seq).to(device),
        )
        # No shuffle: maintain date ordering for cross-sectional attention
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=config.learning_rate, weight_decay=1e-4,
        )
        criterion = nn.MSELoss()

        self._net.train()
        for epoch in range(config.max_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

        self._net.eval()

    def predict(self, data):
        if self._net is None or not self._price_idx:
            return np.zeros(len(data.X))

        device = next(self._net.parameters()).device
        X_price = _nan_safe(data.X[:, self._price_idx].astype(np.float32))
        X_scaled = self._scaler.transform(X_price).astype(np.float32)
        X_seq = _build_sequences(X_scaled, self._seq_len)

        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(X_seq).to(device)).cpu().numpy()

        # Pad front to match original length
        if len(preds) < len(data.X):
            preds = np.concatenate([np.zeros(len(data.X) - len(preds)), preds])
        return preds

    def num_parameters(self):
        if self._net:
            return sum(p.numel() for p in self._net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy G: Short Horizon NLP (5D)
# ---------------------------------------------------------------------------

class ShortHorizonNLPStrategy(NLPOnlyStrategy):
    """NLP-only model specifically for short (5D) horizon.

    Identical architecture to B, but only evaluated at 5D.
    """

    name = "G_Short_NLP_5D"
    supported_horizons = [5]


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "A": LSTMBaselineStrategy,
    "B": NLPOnlyStrategy,
    "C": LateEnsembleStrategy,
    "D": ResidualSentimentStrategy,
    "E": GatedHybridStrategy,
    "F": CrossSectionalLSTMStrategy,
    "G": ShortHorizonNLPStrategy,
}


# ---------------------------------------------------------------------------
# Benchmark evaluator
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    """Run walk-forward evaluation for all strategies under identical conditions.

    Usage:
        evaluator = BenchmarkEvaluator(panel, feature_cols, config)
        results = evaluator.run_all(STRATEGY_REGISTRY.values())
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str],
        config: BenchmarkConfig,
    ):
        self.panel = panel
        self.feature_cols = feature_cols
        self.config = config

    def _make_strategy_data(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> StrategyData:
        """Convert a panel DataFrame slice to StrategyData."""
        # Ensure feature columns exist
        cols = [c for c in self.feature_cols if c in df.columns]
        X = _nan_safe(df[cols].values.astype(np.float32))
        y = _nan_safe(df[target_col].values.astype(np.float32))

        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values(0).values
            tickers = df.index.get_level_values(1).values
        else:
            dates = df.index.values
            tickers = np.array(["UNK"] * len(df))

        return StrategyData(
            X=X, y=y,
            feature_names=cols,
            dates=dates,
            tickers=tickers,
        )

    def run_walk_forward(
        self,
        strategy: StrategyModel,
        horizon_days: int,
        target_col: str,
    ) -> StrategyResult:
        """Run full walk-forward evaluation for one strategy at one horizon."""
        from training.model_selection import WalkForwardValidator, WalkForwardConfig

        if isinstance(self.panel.index, pd.MultiIndex):
            dates = self.panel.index.get_level_values(0).unique().sort_values()
        else:
            dates = self.panel.index.unique().sort_values()

        wf_config = WalkForwardConfig(
            train_start=str(dates[0].date()),
            test_end=str(dates[-1].date()),
            train_min_months=self.config.train_years * 12,
            val_months=self.config.val_months,
            test_months=self.config.test_months,
            step_months=self.config.step_months,
            embargo_days=self.config.embargo_days,
            expanding=True,
        )

        validator = WalkForwardValidator(wf_config)
        folds = validator.generate_folds(pd.DatetimeIndex(dates))

        fold_metrics = []  # type: List[FoldMetrics]
        t0 = time.time()

        for fold in folds:
            try:
                train_df, val_df, test_df = validator.split_data(self.panel, fold)

                if target_col not in train_df.columns:
                    continue

                train_data = self._make_strategy_data(train_df, target_col)
                val_data = (self._make_strategy_data(val_df, target_col)
                            if len(val_df) > 0 else None)
                test_data = self._make_strategy_data(test_df, target_col)

                if len(test_data.y) < 50:
                    continue

                # Train
                strategy.train(train_data, val_data, self.config)

                # Predict test
                test_preds = strategy.predict(test_data)
                # Predict train (for overfitting measurement)
                train_preds = strategy.predict(train_data)

                # Cross-sectional IC
                ic_mean, ic_std = compute_cross_sectional_ic(
                    test_preds, test_data.y, test_data.dates,
                    min_stocks=self.config.min_stocks_for_ic,
                )
                train_ic, _ = compute_cross_sectional_ic(
                    train_preds, train_data.y, train_data.dates,
                    min_stocks=self.config.min_stocks_for_ic,
                )

                # Investment metrics
                sharpe, mdd = compute_simple_investment_metrics(
                    test_preds, test_data.y,
                )

                pred_std = float(np.std(test_preds[~np.isnan(test_preds)]))
                n_dates = len(np.unique(test_data.dates))

                fold_metrics.append(FoldMetrics(
                    fold_idx=getattr(fold, "fold_idx", 0),
                    ic=ic_mean,
                    ic_std=ic_std,
                    sharpe=sharpe,
                    max_drawdown=mdd,
                    n_samples=len(test_data.y),
                    n_dates=n_dates,
                    pred_std=pred_std,
                    train_ic=train_ic,
                ))
                logger.info(
                    "  Fold %s: IC=%.4f  Train_IC=%.4f  Sharpe=%.2f",
                    getattr(fold, "fold_idx", "?"),
                    ic_mean, train_ic, sharpe,
                )

            except Exception as e:
                logger.warning("Fold failed for %s: %s", strategy.name, e)
                continue

        train_time = time.time() - t0
        horizon_label = "%dD" % horizon_days

        if not fold_metrics:
            return StrategyResult(
                name=strategy.name, horizon=horizon_label,
                horizon_days=horizon_days,
                ic_mean=0.0, ic_std=0.0, icir=0.0,
                sharpe=0.0, max_drawdown=0.0,
                overfit_score=1.0, composite=float("-inf"),
                fold_metrics=[], prod_ic=0.0,
                param_count=strategy.num_parameters(),
                train_time=train_time, status="FAIL",
            )

        # Aggregate
        ics = [f.ic for f in fold_metrics]
        ic_mean = float(np.mean(ics))
        ic_std_val = float(np.std(ics)) if len(ics) > 1 else 0.0
        icir = ic_mean / ic_std_val if ic_std_val > 1e-8 else 0.0

        mean_sharpe = float(np.mean([f.sharpe for f in fold_metrics]))
        mean_mdd = float(np.mean([f.max_drawdown for f in fold_metrics]))

        # Overfitting score
        mean_train_ic = float(np.mean([f.train_ic for f in fold_metrics]))
        if abs(mean_train_ic) > 1e-8:
            overfit = max(0.0, (mean_train_ic - ic_mean) / abs(mean_train_ic))
        else:
            overfit = 0.5
        overfit = min(overfit, 1.0)

        # Composite
        composite = (
            ic_mean
            + 0.5 * icir
            + 0.3 * mean_sharpe
            - 0.3 * ic_std_val
            - 0.5 * overfit
        )
        if math.isnan(composite):
            composite = float("-inf")

        # Production IC = last fold
        prod_ic = fold_metrics[-1].ic

        # Status
        status = self._determine_status(ic_mean, prod_ic, overfit)

        diag = strategy.get_diagnostics()

        return StrategyResult(
            name=strategy.name,
            horizon=horizon_label,
            horizon_days=horizon_days,
            ic_mean=ic_mean,
            ic_std=ic_std_val,
            icir=icir,
            sharpe=mean_sharpe,
            max_drawdown=mean_mdd,
            overfit_score=overfit,
            composite=composite,
            fold_metrics=fold_metrics,
            prod_ic=prod_ic,
            param_count=strategy.num_parameters(),
            train_time=train_time,
            status=status,
            gate_stats=diag.get("gate_stats"),
            ensemble_weights=diag.get("ensemble_weights"),
        )

    @staticmethod
    def _determine_status(ic_mean, prod_ic, overfit):
        if math.isnan(ic_mean) or ic_mean <= 0:
            return "FAIL"
        if overfit >= 0.6:
            return "WARN"
        if prod_ic >= 0.9 * ic_mean and ic_mean > 0:
            return "PASS"
        if ic_mean > 0:
            return "WARN"
        return "FAIL"

    def run_all(
        self,
        strategy_classes: List[type],
    ) -> List[StrategyResult]:
        """Run all strategies at all horizons.

        Args:
            strategy_classes: List of StrategyModel subclasses.

        Returns:
            List of StrategyResult sorted by composite score (desc).
        """
        results = []  # type: List[StrategyResult]

        for horizon_days in self.config.horizons:
            target_col = "fwd_return_%dd" % horizon_days

            if target_col not in self.panel.columns:
                logger.warning("Target column %s not in panel, skipping",
                               target_col)
                continue

            logger.info("=" * 60)
            logger.info("HORIZON: %dD (target=%s)", horizon_days, target_col)
            logger.info("=" * 60)

            for cls in strategy_classes:
                # Check supported horizons
                sh = getattr(cls, "supported_horizons", None)
                if sh is not None and horizon_days not in sh:
                    logger.info("Skipping %s at %dD (unsupported)",
                                cls.name, horizon_days)
                    continue

                strategy = cls()
                logger.info("--- %s / %dD ---", strategy.name, horizon_days)

                result = self.run_walk_forward(strategy, horizon_days, target_col)
                results.append(result)

                logger.info(
                    "RESULT %s/%dD: IC=%.4f ICIR=%.2f Sharpe=%.2f "
                    "Composite=%.4f [%s] (%d params, %.1fs)",
                    result.name, horizon_days,
                    result.ic_mean, result.icir, result.sharpe,
                    result.composite, result.status,
                    result.param_count, result.train_time,
                )

        results.sort(key=lambda r: r.composite, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Save / load benchmark results
# ---------------------------------------------------------------------------

def save_benchmark_results(
    results: List[StrategyResult],
    path: Optional[str] = None,
) -> str:
    """Save benchmark results to JSON.

    Args:
        results: List of StrategyResult.
        path: Output path. Defaults to artifacts/strategy_benchmark_results.json.

    Returns:
        Absolute path to saved file.
    """
    if path is None:
        artifacts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts",
        )
        os.makedirs(artifacts_dir, exist_ok=True)
        path = os.path.join(artifacts_dir, "strategy_benchmark_results.json")

    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_strategies": len(set(r.name for r in results)),
        "n_horizons": len(set(r.horizon for r in results)),
        "results": [r.to_dict() for r in results],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Benchmark results saved to %s", path)
    return os.path.abspath(path)


def run_integrity_checks(results: List[StrategyResult]) -> List[str]:
    """Run experimental integrity checks on benchmark results.

    Returns:
        List of warning messages (empty = all clear).
    """
    warnings = []  # type: List[str]

    for r in results:
        prefix = "%s/%s" % (r.name, r.horizon)

        # Check prediction std > 0
        for fm in r.fold_metrics:
            if fm.pred_std < 1e-10:
                warnings.append(
                    "%s fold %d: prediction std=0 (constant predictions)"
                    % (prefix, fm.fold_idx)
                )

        # Check cross-sectional size
        for fm in r.fold_metrics:
            if fm.n_dates < 5:
                warnings.append(
                    "%s fold %d: only %d test dates (too few for reliable IC)"
                    % (prefix, fm.fold_idx, fm.n_dates)
                )

        # Check for NaN
        if math.isnan(r.ic_mean):
            warnings.append("%s: IC is NaN" % prefix)
        if math.isnan(r.composite):
            warnings.append("%s: composite score is NaN" % prefix)

        # Check param count
        if r.param_count > 1_500_000:
            warnings.append(
                "%s: %d params exceeds 1.5M limit" % (prefix, r.param_count)
            )

    return warnings
