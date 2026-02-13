"""
Multi-Strategy Benchmark Framework
-----------------------------------
Compare alpha extraction strategies under identical walk-forward conditions.

Strategies (15 total):
  A  - LSTM Baseline (price-only temporal)
  B  - NLP Only (sentiment MLP)
  C  - Late Ensemble (A + B with ridge)
  D  - Residual Sentiment (market-residualized NLP)
  E  - Gated Hybrid (gated price-NLP fusion)
  E1 - Cross-Attention Hybrid (price/NLP cross-attention)
  E2 - Additive Residual (T + alpha*S fusion)
  F  - Cross-Sectional Attention LSTM (temporal + stock attention)
  F1 - Temporal Fusion (simplified TFT)
  G  - Transformer Forecast (encoder-only transformer)
  G1 - Efficient Transformer (linear attention)
  H  - Random Forest (sklearn, no torch)
  I  - Gradient Boosted Trees (LightGBM, no torch)
  J  - Short Horizon NLP (5D sentiment)
  K  - Short Horizon Hybrid (5D gated fusion)

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
    max_epochs: int = 10
    early_stop_patience: int = 3
    ranking_weight: float = 0.5
    max_params: int = 1_500_000
    batch_size: int = 128
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
            val_months=wf.get("val_months", 3),
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
class FoldPredictions:
    """Raw predictions from a single walk-forward fold."""
    fold_idx: int
    predictions: np.ndarray
    actuals: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray


@dataclass
class ExtendedMetrics:
    """Extended evaluation metrics beyond IC/Sharpe."""
    # Regression
    mse: float
    rmse: float
    mae: float
    r_squared: float
    # Per-stock R2
    mean_stock_r2: float
    median_stock_r2: float
    pct_r2_positive: float
    pct_r2_above_005: float
    stock_r2_values: Dict[str, float]
    # Directional
    hit_ratio: float
    precision: float
    recall: float
    f1: float
    # Calibration
    calib_slope: float
    calib_intercept: float
    # Market-level
    market_r2: float
    market_direction_accuracy: float
    # Backtest (long-only + long-short 20/20)
    lo_sharpe: float
    lo_cagr: float
    lo_max_dd: float
    ls_sharpe: float
    ls_cagr: float
    ls_max_dd: float
    lo_equity: Optional[np.ndarray] = None
    ls_equity: Optional[np.ndarray] = None
    equity_dates: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse": self.mse, "rmse": self.rmse, "mae": self.mae,
            "r_squared": self.r_squared,
            "mean_stock_r2": self.mean_stock_r2,
            "median_stock_r2": self.median_stock_r2,
            "pct_r2_positive": self.pct_r2_positive,
            "pct_r2_above_005": self.pct_r2_above_005,
            "hit_ratio": self.hit_ratio,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "calib_slope": self.calib_slope,
            "calib_intercept": self.calib_intercept,
            "market_r2": self.market_r2,
            "market_direction_accuracy": self.market_direction_accuracy,
            "lo_sharpe": self.lo_sharpe, "lo_cagr": self.lo_cagr,
            "lo_max_dd": self.lo_max_dd,
            "ls_sharpe": self.ls_sharpe, "ls_cagr": self.ls_cagr,
            "ls_max_dd": self.ls_max_dd,
        }


@dataclass
class DMTestResult:
    """Diebold-Mariano test result between two strategies."""
    strategy_a: str
    strategy_b: str
    horizon: str
    dm_stat: float
    p_value: float
    better: str


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
    fold_predictions: Optional[List[FoldPredictions]] = None
    extended: Optional[ExtendedMetrics] = None

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
        if self.extended:
            d["extended"] = self.extended.to_dict()
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

    class _CrossAttentionHybridNet(nn.Module):
        """Cross-attention fusion of price (temporal) and NLP (static). ~88K params.

        price_seq -> LightPriceEncoder -> T (64-dim)
        nlp_static -> Linear+GELU -> S (64-dim)
        CrossAttention(Q=T, K=V=S) -> F (64-dim)
        F -> Linear(64,32) -> ReLU -> Linear(32,1)
        """

        def __init__(self, n_price: int, n_nlp: int, hidden: int = 64):
            super().__init__()
            from models.hybrid_model import LightPriceEncoder
            self.price_enc = LightPriceEncoder(n_price, hidden)
            self.nlp_proj = nn.Sequential(
                nn.Linear(n_nlp, hidden),
                nn.GELU(),
            )
            self.cross_attn = nn.MultiheadAttention(
                hidden, num_heads=2, batch_first=True,
            )
            self.ln = nn.LayerNorm(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, price_seq, nlp_static):
            T = self.price_enc(price_seq).unsqueeze(1)  # (B, 1, 64)
            S = self.nlp_proj(nlp_static).unsqueeze(1)  # (B, 1, 64)
            F_attn, _ = self.cross_attn(T, S, S)        # (B, 1, 64)
            F_out = self.ln(T + F_attn).squeeze(1)      # (B, 64)
            return self.head(F_out).squeeze(-1)

    class _AdditiveResidualNet(nn.Module):
        """Additive residual fusion: F = T + sigmoid(alpha)*S. ~54K params.

        price_seq -> LightPriceEncoder -> T (64-dim)
        nlp_static -> Linear+GELU -> S (64-dim)
        F = T + sigmoid(alpha) * S    (alpha is learnable scalar)
        F -> Linear(64,32) -> ReLU -> Linear(32,1)
        """

        def __init__(self, n_price: int, n_nlp: int, hidden: int = 64):
            super().__init__()
            from models.hybrid_model import LightPriceEncoder
            self.price_enc = LightPriceEncoder(n_price, hidden)
            self.nlp_proj = nn.Sequential(
                nn.Linear(n_nlp, hidden),
                nn.GELU(),
            )
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.head = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, price_seq, nlp_static):
            T = self.price_enc(price_seq)        # (B, 64)
            S = self.nlp_proj(nlp_static)        # (B, 64)
            F = T + torch.sigmoid(self.alpha) * S
            return self.head(F).squeeze(-1)

    class _VariableSelectionGLU(nn.Module):
        """Variable selection with Gated Linear Unit for TFT."""

        def __init__(self, n_features: int, hidden: int = 64):
            super().__init__()
            self.fc = nn.Linear(n_features, hidden * 2)
            self.softmax_weights = nn.Linear(n_features, n_features)
            self.ln = nn.LayerNorm(hidden)

        def forward(self, x):
            # x: (B, n_features) or (B, seq, n_features)
            # Variable selection
            weights = torch.softmax(self.softmax_weights(x), dim=-1)
            selected = x * weights
            # GLU
            h = self.fc(selected)
            h1, h2 = h.chunk(2, dim=-1)
            return self.ln(h1 * torch.sigmoid(h2))

    class _GRNGate(nn.Module):
        """Gated Residual Network for TFT."""

        def __init__(self, hidden: int = 64, dropout: float = 0.1):
            super().__init__()
            self.fc1 = nn.Linear(hidden, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.gate = nn.Linear(hidden, hidden * 2)
            self.ln = nn.LayerNorm(hidden)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            h = self.dropout(self.fc2(h))
            g = self.gate(h)
            g1, g2 = g.chunk(2, dim=-1)
            gated = g1 * torch.sigmoid(g2)
            return self.ln(x + gated)

    class _SimplifiedTFTNet(nn.Module):
        """Simplified Temporal Fusion Transformer. ~80K params.

        features -> VariableSelectionGLU -> selected
        selected_seq -> LSTM(64, 2 layers) -> lstm_out
        lstm_out -> MultiheadAttention(64, 2 heads) -> attn
        attn[:,-1,:] -> GRN_gate(64) -> Linear(64,32) -> ReLU -> Linear(32,1)
        """

        def __init__(self, n_features: int, hidden: int = 64):
            super().__init__()
            self.vs = _VariableSelectionGLU(n_features, hidden)
            self.lstm = nn.LSTM(
                hidden, hidden, num_layers=2,
                batch_first=True, dropout=0.2,
            )
            self.attn = nn.MultiheadAttention(
                hidden, num_heads=2, batch_first=True,
            )
            self.attn_ln = nn.LayerNorm(hidden)
            self.grn = _GRNGate(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x: (B, seq, n_features)
            selected = self.vs(x)          # (B, seq, 64)
            lstm_out, _ = self.lstm(selected)  # (B, seq, 64)
            attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
            h = self.attn_ln(lstm_out + attn_out)
            h_last = h[:, -1, :]           # (B, 64)
            h_gated = self.grn(h_last)     # (B, 64)
            return self.head(h_gated).squeeze(-1)

    class _EfficientTransformerNet(nn.Module):
        """Efficient Transformer with linear attention. ~73K params.

        price_seq -> Linear(n_feat, 64) + PositionalEncoding
        -> 2x LinearAttentionLayer(64, 2 heads)   # O(n*d) kernel attention
        -> mean pool -> Linear(64,32) -> ReLU -> Linear(32,1)
        """

        def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden)
            self.pos_enc = nn.Parameter(torch.randn(1, 200, hidden) * 0.02)
            self.layers = nn.ModuleList([
                _LinearAttentionLayer(hidden, n_heads=2) for _ in range(n_layers)
            ])
            self.ln = nn.LayerNorm(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x: (B, seq, n_features)
            seq_len = x.size(1)
            h = self.input_proj(x) + self.pos_enc[:, :seq_len, :]
            for layer in self.layers:
                h = layer(h)
            h = self.ln(h.mean(dim=1))  # mean pool
            return self.head(h).squeeze(-1)

    class _LinearAttentionLayer(nn.Module):
        """Linear attention: O(n*d) complexity using kernel trick.

        Q' = elu(Q)+1, K' = elu(K)+1
        Attn = (Q' @ (K'^T @ V)) / (Q' @ K'.sum(0))
        """

        def __init__(self, hidden: int = 64, n_heads: int = 2):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = hidden // n_heads
            self.qkv = nn.Linear(hidden, hidden * 3)
            self.out_proj = nn.Linear(hidden, hidden)
            self.ln = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Linear(hidden * 2, hidden),
            )
            self.ln2 = nn.LayerNorm(hidden)

        def forward(self, x):
            B, S, D = x.shape
            qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, d)
            Q, K, V = qkv[0], qkv[1], qkv[2]

            # Kernel feature map: elu(x) + 1
            Q_prime = torch.nn.functional.elu(Q) + 1.0
            K_prime = torch.nn.functional.elu(K) + 1.0

            # Linear attention: O(n*d^2)
            KV = torch.einsum("bhsd,bhse->bhde", K_prime, V)  # (B,H,d,d)
            QKV = torch.einsum("bhsd,bhde->bhse", Q_prime, KV)  # (B,H,S,d)
            K_sum = K_prime.sum(dim=2, keepdim=True)  # (B,H,1,d)
            denom = torch.einsum("bhsd,bhrd->bhsr", Q_prime, K_sum)  # (B,H,S,1)
            attn_out = QKV / (denom + 1e-6)

            # Reshape back
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
            h = self.ln(x + self.out_proj(attn_out))
            h = self.ln2(h + self.ffn(h))
            return h


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
    """Convert 2D array to 3D sliding-window sequences (vectorized)."""
    n = X.shape[0]
    if n <= seq_len:
        pad_len = seq_len - n + 1
        X = np.vstack([np.zeros((pad_len, X.shape[1]), dtype=np.float32), X])
        n = X.shape[0]

    idx = np.arange(seq_len)[None, :] + np.arange(n - seq_len)[:, None]
    return X[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: train dual-input (seq + static) network
# ---------------------------------------------------------------------------

def _train_seq_static_loop(net, X_seq, X_static, y, X_seq_val, X_static_val,
                           y_val, config):
    """Train a dual-input PyTorch net with early stopping."""
    if not HAS_TORCH:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    X_seq_t = torch.from_numpy(X_seq).to(device)
    X_st_t = torch.from_numpy(X_static).to(device)
    y_t = torch.from_numpy(y).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    X_seq_v, X_st_v, y_v_t = None, None, None
    if X_seq_val is not None and y_val is not None and len(y_val) > 20:
        X_seq_v = torch.from_numpy(X_seq_val).to(device)
        X_st_v = torch.from_numpy(X_static_val).to(device)
        y_v_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf")
    patience_ctr = 0
    bs = config.batch_size

    net.train()
    for epoch in range(config.max_epochs):
        perm = torch.randperm(len(y_t))
        for i in range(0, len(y_t), bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            pred = net(X_seq_t[idx], X_st_t[idx])
            loss = criterion(pred, y_t[idx])
            loss.backward()
            optimizer.step()

        if X_seq_v is not None:
            net.eval()
            with torch.no_grad():
                vl = criterion(net(X_seq_v, X_st_v), y_v_t).item()
            net.train()
            if vl < best_val:
                best_val = vl
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.early_stop_patience:
                    break

    net.eval()


def _train_seq_loop(net, X_seq, y, X_seq_val, y_val, config):
    """Train a single-input sequential PyTorch net with early stopping."""
    if not HAS_TORCH:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    dataset = TensorDataset(
        torch.from_numpy(X_seq).to(device),
        torch.from_numpy(y).to(device),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    X_v_t, y_v_t = None, None
    if X_seq_val is not None and y_val is not None and len(y_val) > 20:
        X_v_t = torch.from_numpy(X_seq_val).to(device)
        y_v_t = torch.from_numpy(y_val).to(device)

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

        if X_v_t is not None:
            net.eval()
            with torch.no_grad():
                vl = criterion(net(X_v_t), y_v_t).item()
            net.train()
            if vl < best_val:
                best_val = vl
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.early_stop_patience:
                    break

    net.eval()


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
            stats_list = self._model.get_gate_stats()
            if stats_list and isinstance(stats_list, list):
                # Summarize per-epoch stats into a single dict
                means = [s.get("gate_mean", 0.5) for s in stats_list if isinstance(s, dict)]
                stds = [s.get("gate_std", 0.0) for s in stats_list if isinstance(s, dict)]
                summary = {
                    "gate_mean": float(np.mean(means)) if means else 0.5,
                    "gate_std": float(np.mean(stds)) if stds else 0.0,
                    "n_epochs": len(stats_list),
                }
                return {"gate_stats": summary}
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
# Strategy E1: Cross-Attention Hybrid
# ---------------------------------------------------------------------------

class AttentionHybridStrategy(StrategyModel):
    """Cross-attention fusion of price sequences and NLP features. ~88K params."""

    name = "E1_Attention_Hybrid"

    def __init__(self):
        self._net = None
        self._price_idx = []  # type: List[int]
        self._nlp_idx = []  # type: List[int]
        self._scaler = None
        self._seq_len = 20

    def train(self, train_data, val_data, config):
        from sklearn.preprocessing import StandardScaler

        self._price_idx = [i for i, n in enumerate(train_data.feature_names)
                           if not n.startswith("nlp_")]
        self._nlp_idx = [i for i, n in enumerate(train_data.feature_names)
                         if n.startswith("nlp_")]
        self._seq_len = config.sequence_length

        if not self._nlp_idx:
            logger.warning("%s: no nlp_* features", self.name)
            return

        X_price = _nan_safe(train_data.X[:, self._price_idx].astype(np.float32))
        X_nlp = _nan_safe(train_data.X[:, self._nlp_idx].astype(np.float32))
        y = _nan_safe(train_data.y.astype(np.float32))

        self._scaler = StandardScaler()
        X_price = self._scaler.fit_transform(X_price).astype(np.float32)

        X_seq = _build_sequences(X_price, self._seq_len)
        offset = len(y) - len(X_seq)
        X_nlp_aligned = X_nlp[offset:]
        y_aligned = y[offset:]

        X_seq_v, X_nlp_v, y_v = None, None, None
        if val_data is not None:
            X_pv = self._scaler.transform(
                _nan_safe(val_data.X[:, self._price_idx].astype(np.float32))
            ).astype(np.float32)
            X_nv = _nan_safe(val_data.X[:, self._nlp_idx].astype(np.float32))
            y_v_raw = _nan_safe(val_data.y.astype(np.float32))
            X_seq_v = _build_sequences(X_pv, self._seq_len)
            off_v = len(y_v_raw) - len(X_seq_v)
            X_nlp_v = X_nv[off_v:]
            y_v = y_v_raw[off_v:]

        self._net = _CrossAttentionHybridNet(
            len(self._price_idx), len(self._nlp_idx), config.hidden_dim,
        )
        _train_seq_static_loop(
            self._net, X_seq, X_nlp_aligned, y_aligned,
            X_seq_v, X_nlp_v, y_v, config,
        )

    def predict(self, data):
        if self._net is None:
            return np.zeros(len(data.X))

        device = next(self._net.parameters()).device
        X_price = self._scaler.transform(
            _nan_safe(data.X[:, self._price_idx].astype(np.float32))
        ).astype(np.float32)
        X_nlp = _nan_safe(data.X[:, self._nlp_idx].astype(np.float32))
        X_seq = _build_sequences(X_price, self._seq_len)
        offset = len(data.X) - len(X_seq)

        self._net.eval()
        with torch.no_grad():
            preds = self._net(
                torch.from_numpy(X_seq).to(device),
                torch.from_numpy(X_nlp[offset:]).to(device),
            ).cpu().numpy()

        if len(preds) < len(data.X):
            preds = np.concatenate([np.zeros(len(data.X) - len(preds)), preds])
        return preds

    def num_parameters(self):
        if self._net:
            return sum(p.numel() for p in self._net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy E2: Additive Residual
# ---------------------------------------------------------------------------

class AdditiveResidualStrategy(StrategyModel):
    """Additive residual fusion: T + sigmoid(alpha)*S. ~54K params."""

    name = "E2_Additive_Residual"

    def __init__(self):
        self._net = None
        self._price_idx = []  # type: List[int]
        self._nlp_idx = []  # type: List[int]
        self._scaler = None
        self._seq_len = 20

    def train(self, train_data, val_data, config):
        from sklearn.preprocessing import StandardScaler

        self._price_idx = [i for i, n in enumerate(train_data.feature_names)
                           if not n.startswith("nlp_")]
        self._nlp_idx = [i for i, n in enumerate(train_data.feature_names)
                         if n.startswith("nlp_")]
        self._seq_len = config.sequence_length

        if not self._nlp_idx:
            logger.warning("%s: no nlp_* features", self.name)
            return

        X_price = _nan_safe(train_data.X[:, self._price_idx].astype(np.float32))
        X_nlp = _nan_safe(train_data.X[:, self._nlp_idx].astype(np.float32))
        y = _nan_safe(train_data.y.astype(np.float32))

        self._scaler = StandardScaler()
        X_price = self._scaler.fit_transform(X_price).astype(np.float32)

        X_seq = _build_sequences(X_price, self._seq_len)
        offset = len(y) - len(X_seq)
        X_nlp_aligned = X_nlp[offset:]
        y_aligned = y[offset:]

        X_seq_v, X_nlp_v, y_v = None, None, None
        if val_data is not None:
            X_pv = self._scaler.transform(
                _nan_safe(val_data.X[:, self._price_idx].astype(np.float32))
            ).astype(np.float32)
            X_nv = _nan_safe(val_data.X[:, self._nlp_idx].astype(np.float32))
            y_v_raw = _nan_safe(val_data.y.astype(np.float32))
            X_seq_v = _build_sequences(X_pv, self._seq_len)
            off_v = len(y_v_raw) - len(X_seq_v)
            X_nlp_v = X_nv[off_v:]
            y_v = y_v_raw[off_v:]

        self._net = _AdditiveResidualNet(
            len(self._price_idx), len(self._nlp_idx), config.hidden_dim,
        )
        _train_seq_static_loop(
            self._net, X_seq, X_nlp_aligned, y_aligned,
            X_seq_v, X_nlp_v, y_v, config,
        )

    def predict(self, data):
        if self._net is None:
            return np.zeros(len(data.X))

        device = next(self._net.parameters()).device
        X_price = self._scaler.transform(
            _nan_safe(data.X[:, self._price_idx].astype(np.float32))
        ).astype(np.float32)
        X_nlp = _nan_safe(data.X[:, self._nlp_idx].astype(np.float32))
        X_seq = _build_sequences(X_price, self._seq_len)
        offset = len(data.X) - len(X_seq)

        self._net.eval()
        with torch.no_grad():
            preds = self._net(
                torch.from_numpy(X_seq).to(device),
                torch.from_numpy(X_nlp[offset:]).to(device),
            ).cpu().numpy()

        if len(preds) < len(data.X):
            preds = np.concatenate([np.zeros(len(data.X) - len(preds)), preds])
        return preds

    def num_parameters(self):
        if self._net:
            return sum(p.numel() for p in self._net.parameters())
        return 0

    def get_diagnostics(self):
        if self._net and hasattr(self._net, "alpha"):
            return {"alpha": float(torch.sigmoid(self._net.alpha).item())}
        return {}


# ---------------------------------------------------------------------------
# Strategy F1: Temporal Fusion (simplified TFT)
# ---------------------------------------------------------------------------

class TemporalFusionStrategy(StrategyModel):
    """Simplified Temporal Fusion Transformer. ~80K params."""

    name = "F1_Temporal_Fusion"

    def __init__(self):
        self._net = None
        self._scaler = None
        self._seq_len = 20

    def train(self, train_data, val_data, config):
        from sklearn.preprocessing import StandardScaler

        self._seq_len = config.sequence_length
        n_feat = len(train_data.feature_names)

        X = _nan_safe(train_data.X.astype(np.float32))
        y = _nan_safe(train_data.y.astype(np.float32))

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X).astype(np.float32)

        X_seq = _build_sequences(X, self._seq_len)
        y_seq = y[self._seq_len:] if len(y) > self._seq_len else y
        min_len = min(len(X_seq), len(y_seq))
        X_seq, y_seq = X_seq[:min_len], y_seq[:min_len]

        X_seq_v, y_v = None, None
        if val_data is not None:
            X_v = self._scaler.transform(
                _nan_safe(val_data.X.astype(np.float32))
            ).astype(np.float32)
            y_v_raw = _nan_safe(val_data.y.astype(np.float32))
            X_seq_v = _build_sequences(X_v, self._seq_len)
            y_v = y_v_raw[self._seq_len:] if len(y_v_raw) > self._seq_len else y_v_raw
            min_v = min(len(X_seq_v), len(y_v))
            X_seq_v, y_v = X_seq_v[:min_v], y_v[:min_v]

        self._net = _SimplifiedTFTNet(n_feat, config.hidden_dim)
        _train_seq_loop(self._net, X_seq, y_seq, X_seq_v, y_v, config)

    def predict(self, data):
        if self._net is None:
            return np.zeros(len(data.X))

        device = next(self._net.parameters()).device
        X = self._scaler.transform(
            _nan_safe(data.X.astype(np.float32))
        ).astype(np.float32)
        X_seq = _build_sequences(X, self._seq_len)

        self._net.eval()
        with torch.no_grad():
            preds = self._net(
                torch.from_numpy(X_seq).to(device)
            ).cpu().numpy()

        if len(preds) < len(data.X):
            preds = np.concatenate([np.zeros(len(data.X) - len(preds)), preds])
        return preds

    def num_parameters(self):
        if self._net:
            return sum(p.numel() for p in self._net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy G: Transformer Forecast
# ---------------------------------------------------------------------------

class TransformerForecastStrategy(StrategyModel):
    """Encoder-only Transformer from training.models."""

    name = "G_Transformer"

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

        self._model = create_model("transformer", {
            "epochs": config.max_epochs,
            "patience": config.early_stop_patience,
            "sequence_length": config.sequence_length,
            "d_model": config.hidden_dim,
            "n_heads": 2,
            "n_layers": 2,
            "d_ff": config.hidden_dim * 2,
            "dropout": 0.2,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        })
        self._model.fit(X_tr, y_tr, X_v, y_v, feature_names=self._price_names)

    def predict(self, data):
        if self._model is None:
            return np.zeros(len(data.X))
        X = _nan_safe(data.X[:, self._price_idx].astype(np.float32))
        preds = self._model.predict(X)
        return _nan_safe(preds)

    def num_parameters(self):
        if self._model and hasattr(self._model, "_inner") and self._model._inner:
            net = self._model._inner
            if hasattr(net, "parameters"):
                return sum(p.numel() for p in net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy G1: Efficient Transformer (linear attention)
# ---------------------------------------------------------------------------

class EfficientTransformerStrategy(StrategyModel):
    """Efficient Transformer with linear attention. ~73K params."""

    name = "G1_Efficient_Transformer"

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
        X_price = self._scaler.fit_transform(X_price).astype(np.float32)

        X_seq = _build_sequences(X_price, self._seq_len)
        y_seq = y[self._seq_len:] if len(y) > self._seq_len else y
        min_len = min(len(X_seq), len(y_seq))
        X_seq, y_seq = X_seq[:min_len], y_seq[:min_len]

        X_seq_v, y_v = None, None
        if val_data is not None:
            X_pv = self._scaler.transform(
                _nan_safe(val_data.X[:, self._price_idx].astype(np.float32))
            ).astype(np.float32)
            y_v_raw = _nan_safe(val_data.y.astype(np.float32))
            X_seq_v = _build_sequences(X_pv, self._seq_len)
            y_v = y_v_raw[self._seq_len:] if len(y_v_raw) > self._seq_len else y_v_raw
            min_v = min(len(X_seq_v), len(y_v))
            X_seq_v, y_v = X_seq_v[:min_v], y_v[:min_v]

        self._net = _EfficientTransformerNet(n_feat, config.hidden_dim)
        _train_seq_loop(self._net, X_seq, y_seq, X_seq_v, y_v, config)

    def predict(self, data):
        if self._net is None:
            return np.zeros(len(data.X))

        device = next(self._net.parameters()).device
        X_price = self._scaler.transform(
            _nan_safe(data.X[:, self._price_idx].astype(np.float32))
        ).astype(np.float32)
        X_seq = _build_sequences(X_price, self._seq_len)

        self._net.eval()
        with torch.no_grad():
            preds = self._net(
                torch.from_numpy(X_seq).to(device)
            ).cpu().numpy()

        if len(preds) < len(data.X):
            preds = np.concatenate([np.zeros(len(data.X) - len(preds)), preds])
        return preds

    def num_parameters(self):
        if self._net:
            return sum(p.numel() for p in self._net.parameters())
        return 0


# ---------------------------------------------------------------------------
# Strategy H: Random Forest (no torch required)
# ---------------------------------------------------------------------------

class RandomForestStrategy(StrategyModel):
    """Random Forest regressor using sklearn. Works without torch."""

    name = "H_Random_Forest"

    def __init__(self):
        self._model = None
        self._scaler = None

    def train(self, train_data, val_data, config):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        X_tr = _nan_safe(train_data.X.astype(np.float32))
        y_tr = _nan_safe(train_data.y.astype(np.float32))

        self._scaler = StandardScaler()
        X_tr = self._scaler.fit_transform(X_tr)

        self._model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        self._model.fit(X_tr, y_tr)

    def predict(self, data):
        if self._model is None:
            return np.zeros(len(data.X))
        X = self._scaler.transform(_nan_safe(data.X.astype(np.float32)))
        return self._model.predict(X).astype(np.float32)

    def num_parameters(self):
        if self._model is not None:
            # Approximate param count from tree structure
            return sum(t.tree_.node_count for t in self._model.estimators_)
        return 0


# ---------------------------------------------------------------------------
# Strategy I: Gradient Boosted Trees (LightGBM, no torch required)
# ---------------------------------------------------------------------------

class GradientBoostedTreeStrategy(StrategyModel):
    """LightGBM regressor using training.models. Works without torch."""

    name = "I_LightGBM"

    def __init__(self):
        self._model = None
        self._feature_names = []  # type: List[str]

    def train(self, train_data, val_data, config):
        from training.models import create_model

        self._feature_names = list(train_data.feature_names)

        X_tr = _nan_safe(train_data.X.astype(np.float32))
        y_tr = _nan_safe(train_data.y.astype(np.float32))

        X_v, y_v = None, None
        if val_data is not None:
            X_v = _nan_safe(val_data.X.astype(np.float32))
            y_v = _nan_safe(val_data.y.astype(np.float32))

        self._model = create_model("lightgbm", {
            "n_estimators": 300,
            "max_depth": 6,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 30,
        })
        self._model.fit(X_tr, y_tr, X_v, y_v,
                        feature_names=self._feature_names)

    def predict(self, data):
        if self._model is None:
            return np.zeros(len(data.X))
        X = _nan_safe(data.X.astype(np.float32))
        return _nan_safe(self._model.predict(X))

    def num_parameters(self):
        if self._model and hasattr(self._model, "model") and self._model.model:
            return self._model.model.num_trees() * 31  # approx
        return 0


# ---------------------------------------------------------------------------
# Strategy J: Short Horizon NLP (5D) -- formerly G
# ---------------------------------------------------------------------------

class ShortHorizonNLPStrategy(NLPOnlyStrategy):
    """NLP-only model specifically for short (5D) horizon.

    Identical architecture to B, but only evaluated at 5D.
    """

    name = "J_Short_NLP_5D"
    supported_horizons = [5]


# ---------------------------------------------------------------------------
# Strategy K: Short Horizon Hybrid (5D)
# ---------------------------------------------------------------------------

class ShortHorizonHybridStrategy(GatedHybridStrategy):
    """Gated hybrid model specifically for short (5D) horizon.

    Identical architecture to E, but only evaluated at 5D.
    """

    name = "K_Short_Hybrid_5D"
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
    "E1": AttentionHybridStrategy,
    "E2": AdditiveResidualStrategy,
    "F": CrossSectionalLSTMStrategy,
    "F1": TemporalFusionStrategy,
    "G": TransformerForecastStrategy,
    "G1": EfficientTransformerStrategy,
    "H": RandomForestStrategy,
    "I": GradientBoostedTreeStrategy,
    "J": ShortHorizonNLPStrategy,
    "K": ShortHorizonHybridStrategy,
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

        data_span_months = (dates[-1] - dates[0]).days / 30.44
        train_min_months = self.config.train_years * 12

        # Auto-adjust: reduce train_min_months if data span is too short
        min_needed = train_min_months + self.config.val_months + self.config.test_months + 2
        if data_span_months < min_needed:
            # Leave room for val + test + embargo
            overhead = self.config.val_months + self.config.test_months + 2
            train_min_months = max(6, int(data_span_months - overhead))
            logger.warning(
                "Data span %.0f months < %.0f needed. "
                "Auto-adjusted train_min_months: %d -> %d",
                data_span_months, min_needed,
                self.config.train_years * 12, train_min_months,
            )

        wf_config = WalkForwardConfig(
            train_start=str(dates[0].date()),
            test_end=str(dates[-1].date()),
            train_min_months=train_min_months,
            val_months=self.config.val_months,
            test_months=self.config.test_months,
            step_months=self.config.step_months,
            embargo_days=self.config.embargo_days,
            expanding=True,
        )

        validator = WalkForwardValidator(wf_config)
        folds = validator.generate_folds(pd.DatetimeIndex(dates))
        logger.info(
            "  %s/%dD: %d folds (train>=%dmo, val=%dmo, test=%dmo, data=%.0fmo)",
            strategy.name, horizon_days, len(folds),
            train_min_months, self.config.val_months,
            self.config.test_months, data_span_months,
        )

        fold_metrics = []  # type: List[FoldMetrics]
        fold_preds_list = []  # type: List[FoldPredictions]
        n_folds = len(folds)
        t0 = time.time()

        for fold_i, fold in enumerate(folds):
            print("      fold %d/%d ..." % (fold_i + 1, n_folds),
                  end="", flush=True)
            fold_t0 = time.time()
            try:
                train_df, val_df, test_df = validator.split_data(self.panel, fold)

                if target_col not in train_df.columns:
                    print(" skipped (no target)", flush=True)
                    continue

                train_data = self._make_strategy_data(train_df, target_col)
                val_data = (self._make_strategy_data(val_df, target_col)
                            if len(val_df) > 0 else None)
                test_data = self._make_strategy_data(test_df, target_col)

                if len(test_data.y) < 50:
                    print(" skipped (too few samples)", flush=True)
                    continue

                # Train
                strategy.train(train_data, val_data, self.config)

                # Predict test
                test_preds = strategy.predict(test_data)

                # Store fold predictions for extended metrics
                fold_idx_val = getattr(fold, "fold_idx", 0)
                fold_preds_list.append(FoldPredictions(
                    fold_idx=fold_idx_val,
                    predictions=test_preds.copy(),
                    actuals=test_data.y.copy(),
                    dates=test_data.dates.copy(),
                    tickers=test_data.tickers.copy(),
                ))

                # Cross-sectional IC
                ic_mean, ic_std = compute_cross_sectional_ic(
                    test_preds, test_data.y, test_data.dates,
                    min_stocks=self.config.min_stocks_for_ic,
                )
                train_ic = 0.0  # skip train inference for speed

                # Investment metrics
                sharpe, mdd = compute_simple_investment_metrics(
                    test_preds, test_data.y,
                )

                pred_std = float(np.std(test_preds[~np.isnan(test_preds)]))
                n_dates = len(np.unique(test_data.dates))

                fold_metrics.append(FoldMetrics(
                    fold_idx=fold_idx_val,
                    ic=ic_mean,
                    ic_std=ic_std,
                    sharpe=sharpe,
                    max_drawdown=mdd,
                    n_samples=len(test_data.y),
                    n_dates=n_dates,
                    pred_std=pred_std,
                    train_ic=train_ic,
                ))
                fold_elapsed = time.time() - fold_t0
                logger.info(
                    "  Fold %s: IC=%.4f  Train_IC=%.4f  Sharpe=%.2f",
                    fold_idx_val,
                    ic_mean, train_ic, sharpe,
                )
                print(" IC=%.4f (%.1fs)" % (ic_mean, fold_elapsed),
                      flush=True)

            except Exception as e:
                fold_elapsed = time.time() - fold_t0
                logger.warning("Fold failed for %s: %s", strategy.name, e)
                print(" FAILED (%.1fs)" % fold_elapsed, flush=True)
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
                fold_predictions=fold_preds_list if fold_preds_list else None,
            )

        # Aggregate
        ics = [f.ic for f in fold_metrics]
        ic_mean = float(np.mean(ics))
        ic_std_val = float(np.std(ics)) if len(ics) > 1 else 0.0
        icir = ic_mean / ic_std_val if ic_std_val > 1e-8 else 0.0

        mean_sharpe = float(np.mean([f.sharpe for f in fold_metrics]))
        mean_mdd = float(np.mean([f.max_drawdown for f in fold_metrics]))

        # Overfitting score -- IC stability penalty (no train inference needed)
        # High IC variance relative to mean IC signals overfitting
        overfit = min(1.0, max(0.0, ic_std_val / max(abs(ic_mean), 1e-8) - 1.0))

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
            fold_predictions=fold_preds_list if fold_preds_list else None,
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

    @staticmethod
    def compute_extended_metrics(result):
        # type: (StrategyResult) -> Optional[ExtendedMetrics]
        """Compute extended evaluation metrics from fold predictions."""
        if not result.fold_predictions:
            return None

        # Concatenate all fold predictions
        all_preds = np.concatenate([fp.predictions for fp in result.fold_predictions])
        all_actuals = np.concatenate([fp.actuals for fp in result.fold_predictions])
        all_dates = np.concatenate([fp.dates for fp in result.fold_predictions])
        all_tickers = np.concatenate([fp.tickers for fp in result.fold_predictions])

        valid = ~(np.isnan(all_preds) | np.isnan(all_actuals))
        preds = all_preds[valid]
        actuals = all_actuals[valid]
        dates = all_dates[valid]
        tickers = all_tickers[valid]

        if len(preds) < 30:
            return None

        # --- Regression metrics ---
        residuals = preds - actuals
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residuals)))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        # --- Per-stock R2 ---
        unique_tickers = np.unique(tickers)
        stock_r2 = {}  # type: Dict[str, float]
        for tk in unique_tickers:
            tk_str = str(tk)
            mask = tickers == tk
            if mask.sum() < 20:
                continue
            p_tk = preds[mask]
            a_tk = actuals[mask]
            ss_r = np.sum((p_tk - a_tk) ** 2)
            ss_t = np.sum((a_tk - np.mean(a_tk)) ** 2)
            stock_r2[tk_str] = float(1.0 - ss_r / ss_t) if ss_t > 1e-10 else 0.0

        r2_vals = list(stock_r2.values())
        if r2_vals:
            mean_stock_r2 = float(np.mean(r2_vals))
            median_stock_r2 = float(np.median(r2_vals))
            pct_pos = float(np.mean([v > 0 for v in r2_vals]) * 100)
            pct_above_005 = float(np.mean([v > 0.05 for v in r2_vals]) * 100)
        else:
            mean_stock_r2 = 0.0
            median_stock_r2 = 0.0
            pct_pos = 0.0
            pct_above_005 = 0.0

        # --- Directional metrics ---
        pred_dir = np.sign(preds)
        actual_dir = np.sign(actuals)
        hit_ratio = float(np.mean(pred_dir == actual_dir))

        # Precision/Recall for positive predictions
        tp = float(np.sum((pred_dir > 0) & (actual_dir > 0)))
        fp = float(np.sum((pred_dir > 0) & (actual_dir <= 0)))
        fn = float(np.sum((pred_dir <= 0) & (actual_dir > 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # --- Calibration ---
        try:
            calib_coef = np.polyfit(preds, actuals, 1)
            calib_slope = float(calib_coef[0])
            calib_intercept = float(calib_coef[1])
        except Exception:
            calib_slope = 0.0
            calib_intercept = 0.0

        # --- Market-level metrics ---
        unique_dates = np.unique(dates)
        mkt_preds = []
        mkt_actuals = []
        for d in unique_dates:
            d_mask = dates == d
            if d_mask.sum() >= 5:
                mkt_preds.append(float(np.mean(preds[d_mask])))
                mkt_actuals.append(float(np.mean(actuals[d_mask])))

        mkt_preds_arr = np.array(mkt_preds)
        mkt_actuals_arr = np.array(mkt_actuals)
        if len(mkt_preds_arr) > 5:
            ss_r_m = np.sum((mkt_preds_arr - mkt_actuals_arr) ** 2)
            ss_t_m = np.sum((mkt_actuals_arr - np.mean(mkt_actuals_arr)) ** 2)
            market_r2 = float(1.0 - ss_r_m / ss_t_m) if ss_t_m > 1e-10 else 0.0
            mkt_dir_acc = float(np.mean(
                np.sign(mkt_preds_arr) == np.sign(mkt_actuals_arr)
            ))
        else:
            market_r2 = 0.0
            mkt_dir_acc = 0.0

        # --- Backtest: Long-Only and Long-Short ---
        lo_equity_list = [1.0]
        ls_equity_list = [1.0]
        equity_dates_list = []

        sorted_dates = sorted(np.unique(dates))
        for d in sorted_dates:
            d_mask = dates == d
            if d_mask.sum() < 5:
                continue
            d_preds = preds[d_mask]
            d_actuals = actuals[d_mask]

            # Long-only: buy stocks with positive prediction
            long_mask = d_preds > 0
            if long_mask.sum() > 0:
                lo_ret = float(np.mean(d_actuals[long_mask]))
            else:
                lo_ret = 0.0

            # Long-short: top 20% long, bottom 20% short
            n_stocks = len(d_preds)
            k = max(1, n_stocks // 5)
            sorted_idx = np.argsort(d_preds)
            long_idx = sorted_idx[-k:]
            short_idx = sorted_idx[:k]
            ls_ret = float(
                np.mean(d_actuals[long_idx]) - np.mean(d_actuals[short_idx])
            ) / 2.0  # half weight each side

            lo_equity_list.append(lo_equity_list[-1] * (1.0 + lo_ret))
            ls_equity_list.append(ls_equity_list[-1] * (1.0 + ls_ret))
            equity_dates_list.append(d)

        lo_equity = np.array(lo_equity_list[1:])
        ls_equity = np.array(ls_equity_list[1:])
        equity_dates = np.array(equity_dates_list) if equity_dates_list else None

        def _compute_backtest_stats(equity):
            if len(equity) < 2:
                return 0.0, 0.0, 0.0
            returns = np.diff(equity) / equity[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) < 2 or np.std(returns) < 1e-10:
                return 0.0, 0.0, 0.0
            sharpe_val = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            # CAGR
            n_years = len(returns) / 252.0
            if n_years > 0 and equity[-1] > 0 and equity[0] > 0:
                cagr = float((equity[-1] / equity[0]) ** (1.0 / max(n_years, 0.01)) - 1.0)
            else:
                cagr = 0.0
            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
            return sharpe_val, cagr, max_dd

        lo_sharpe, lo_cagr, lo_max_dd = _compute_backtest_stats(lo_equity)
        ls_sharpe, ls_cagr, ls_max_dd = _compute_backtest_stats(ls_equity)

        return ExtendedMetrics(
            mse=mse, rmse=rmse, mae=mae, r_squared=r_squared,
            mean_stock_r2=mean_stock_r2, median_stock_r2=median_stock_r2,
            pct_r2_positive=pct_pos, pct_r2_above_005=pct_above_005,
            stock_r2_values=stock_r2,
            hit_ratio=hit_ratio, precision=precision, recall=recall, f1=f1,
            calib_slope=calib_slope, calib_intercept=calib_intercept,
            market_r2=market_r2, market_direction_accuracy=mkt_dir_acc,
            lo_sharpe=lo_sharpe, lo_cagr=lo_cagr, lo_max_dd=lo_max_dd,
            ls_sharpe=ls_sharpe, ls_cagr=ls_cagr, ls_max_dd=ls_max_dd,
            lo_equity=lo_equity if len(lo_equity) > 0 else None,
            ls_equity=ls_equity if len(ls_equity) > 0 else None,
            equity_dates=equity_dates,
        )

    @staticmethod
    def compute_dm_tests(results):
        # type: (List[StrategyResult]) -> List[DMTestResult]
        """Compute Diebold-Mariano tests between all strategy pairs per horizon.

        Uses Newey-West HAC standard errors for robust inference.
        """
        from scipy.stats import norm

        dm_results = []  # type: List[DMTestResult]

        # Group results by horizon
        horizons = sorted(set(r.horizon for r in results))
        for h in horizons:
            h_results = [r for r in results if r.horizon == h and r.fold_predictions]
            if len(h_results) < 2:
                continue

            # Build aligned prediction arrays per strategy
            # Use (date_str, ticker_str) as key for alignment
            strategy_preds = {}  # type: Dict[str, Dict[tuple, Tuple[float, float]]]
            for r in h_results:
                pred_map = {}  # type: Dict[tuple, Tuple[float, float]]
                for fp in r.fold_predictions:
                    for i in range(len(fp.predictions)):
                        if np.isnan(fp.predictions[i]) or np.isnan(fp.actuals[i]):
                            continue
                        key = (str(fp.dates[i]), str(fp.tickers[i]))
                        pred_map[key] = (fp.predictions[i], fp.actuals[i])
                strategy_preds[r.name] = pred_map

            names = list(strategy_preds.keys())
            for i_a in range(len(names)):
                for i_b in range(i_a + 1, len(names)):
                    name_a = names[i_a]
                    name_b = names[i_b]
                    map_a = strategy_preds[name_a]
                    map_b = strategy_preds[name_b]

                    # Find common keys
                    common = set(map_a.keys()) & set(map_b.keys())
                    if len(common) < 30:
                        continue

                    # Compute loss differential
                    d_vals = []
                    for key in common:
                        pa, actual = map_a[key]
                        pb, _ = map_b[key]
                        loss_a = (pa - actual) ** 2
                        loss_b = (pb - actual) ** 2
                        d_vals.append(loss_a - loss_b)

                    d = np.array(d_vals)
                    n = len(d)
                    d_mean = float(np.mean(d))

                    # Newey-West HAC SE
                    h_lag = max(1, int(n ** (1.0 / 3.0)))
                    gamma_0 = float(np.mean((d - d_mean) ** 2))
                    nw_var = gamma_0
                    for lag in range(1, h_lag + 1):
                        gamma_j = float(np.mean(
                            (d[lag:] - d_mean) * (d[:-lag] - d_mean)
                        ))
                        w = 1.0 - lag / (h_lag + 1.0)
                        nw_var += 2.0 * w * gamma_j

                    se = np.sqrt(max(nw_var / n, 1e-20))
                    dm_stat = d_mean / se if se > 1e-10 else 0.0
                    p_value = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))
                    p_value = max(0.0, min(1.0, p_value))

                    better = name_a if d_mean < 0 else name_b

                    dm_results.append(DMTestResult(
                        strategy_a=name_a,
                        strategy_b=name_b,
                        horizon=h,
                        dm_stat=float(dm_stat),
                        p_value=p_value,
                        better=better,
                    ))

        return dm_results

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

        # Pre-compute total evaluations for progress tracking
        total_evals = 0
        for horizon_days in self.config.horizons:
            target_col = "fwd_return_%dd" % horizon_days
            if target_col not in self.panel.columns:
                continue
            for cls in strategy_classes:
                sh = getattr(cls, "supported_horizons", None)
                if sh is not None and horizon_days not in sh:
                    continue
                total_evals += 1

        completed = 0
        run_start = time.time()

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

                completed += 1
                strategy = cls()

                # Progress display
                elapsed = time.time() - run_start
                if completed > 1 and elapsed > 0:
                    avg_per_eval = elapsed / (completed - 1)
                    remaining = avg_per_eval * (total_evals - completed + 1)
                    eta_min = remaining / 60.0
                    eta_str = "ETA %.0fmin" % eta_min
                else:
                    eta_str = "ETA --"

                progress_msg = (
                    "[%d/%d] %s / %dD  (%.1fmin elapsed, %s)"
                    % (completed, total_evals, strategy.name,
                       horizon_days, elapsed / 60.0, eta_str)
                )
                logger.info("=" * 60)
                logger.info("PROGRESS: %s", progress_msg)
                logger.info("=" * 60)
                print("\n>>> [%d/%d] Running: %s / %dD  |  elapsed %.1fmin  |  %s"
                      % (completed, total_evals, strategy.name,
                         horizon_days, elapsed / 60.0, eta_str),
                      flush=True)

                result = self.run_walk_forward(strategy, horizon_days, target_col)

                # Compute extended metrics from fold predictions
                try:
                    result.extended = self.compute_extended_metrics(result)
                except Exception as ext_e:
                    logger.warning("Extended metrics failed for %s: %s",
                                   strategy.name, ext_e)

                results.append(result)

                logger.info(
                    "RESULT %s/%dD: IC=%.4f ICIR=%.2f Sharpe=%.2f "
                    "Composite=%.4f [%s] (%d params, %.1fs)",
                    result.name, horizon_days,
                    result.ic_mean, result.icir, result.sharpe,
                    result.composite, result.status,
                    result.param_count, result.train_time,
                )
                print("    Done: %s/%dD -> IC=%.4f Sharpe=%.2f [%s] (%.1fs)"
                      % (result.name, horizon_days, result.ic_mean,
                         result.sharpe, result.status, result.train_time),
                      flush=True)

        total_elapsed = time.time() - run_start
        print("\n>>> All %d evaluations complete in %.1f min"
              % (total_evals, total_elapsed / 60.0), flush=True)

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
