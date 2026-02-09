"""
Time-Series Transformer Model
------------------------------
Encoder-only Transformer for financial time-series forecasting.

Architecture:
  - Learnable positional encoding (not sinusoidal -- financial data has no fixed frequency)
  - Input projection + LayerNorm
  - N Transformer encoder layers with multi-head self-attention
  - Gaussian output heads (same interface as LSTM-Gaussian)

When to use Transformer over LSTM:
  - Longer sequences (>120 timesteps): attention scales better than recurrence
  - When temporal patterns span multiple scales simultaneously
  - When you have sufficient training data (>10K sequences)

When LSTM is better:
  - Short sequences (<60 timesteps)
  - Small datasets (Transformer overfits faster)
  - When ordering matters more than global context

Both share the same interface for easy switching:
    model_type = "lstm"      -> LSTMGaussianModel
    model_type = "transformer" -> TransformerGaussianModel
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for financial time-series.

    Unlike sinusoidal encoding, learned positions can capture
    irregular temporal patterns (e.g., Monday effects, month-end).
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("positions", torch.arange(max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = self.positions[:seq_len].unsqueeze(0)  # (1, seq_len)
        return self.dropout(x + self.pos_embed(pos))


class TransformerGaussianNet(nn.Module):
    """Encoder-only Transformer with Gaussian output heads."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 512,
        n_horizons: int = 3,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Aggregation: learned weighted pool over sequence
        self.pool_weights = nn.Linear(d_model, 1)

        # Per-horizon Gaussian output heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model // 2, 2),  # [mean, log_variance]
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, n_horizons, 2)
        """
        # Project and add position
        h = self.input_proj(x)         # (batch, seq_len, d_model)
        h = self.pos_encoding(h)

        # Transformer encoder (causal mask not needed -- we only predict from past)
        h = self.encoder(h)            # (batch, seq_len, d_model)

        # Attention-weighted pooling
        weights = self.pool_weights(h)       # (batch, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = (h * weights).sum(dim=1)     # (batch, d_model)

        # Per-horizon output
        outputs = [head(pooled) for head in self.horizon_heads]
        return torch.stack(outputs, dim=1)    # (batch, n_horizons, 2)


def gaussian_nll_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Same loss as LSTM-Gaussian for consistency."""
    mu = predictions[:, :, 0]
    log_var = torch.clamp(predictions[:, :, 1], min=-10, max=10)
    loss = 0.5 * (log_var + (targets - mu) ** 2 / torch.exp(log_var))
    return loss.mean()


class TransformerGaussianModel:
    """Complete Transformer-Gaussian model matching LSTMGaussianModel interface."""

    def __init__(self, config: Optional[Dict] = None):
        defaults = {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 256,
            "dropout": 0.2,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "epochs": 150,
            "patience": 20,
            "sequence_length": 60,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "warmup_steps": 500,
            "horizons": [21, 63, 126],
            "use_mixed_precision": False,
            "freeze_layers": 0,
        }
        self.config = {**defaults, **(config or {})}
        self.net = None
        self.scaler = StandardScaler()
        self.device = torch.device("cpu")
        self.is_fitted = False
        self._train_losses = []
        self._val_losses = []

    def _create_sequences(self, X, y=None):
        seq_len = self.config["sequence_length"]
        n = X.shape[0]
        if n <= seq_len:
            return np.empty((0, seq_len, X.shape[1])), None

        X_seq, y_seq = [], [] if y is not None else None
        for i in range(seq_len, n):
            X_seq.append(X[i - seq_len:i])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32) if y is not None else None
        return X_seq, y_seq

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        n_horizons = len(self.config["horizons"])
        if y_train.ndim == 1:
            y_train = np.repeat(y_train[:, None], n_horizons, axis=1)
        if y_val is not None and y_val.ndim == 1:
            y_val = np.repeat(y_val[:, None], n_horizons, axis=1)

        X_train_s = self.scaler.fit_transform(X_train)
        X_train_seq, y_train_seq = self._create_sequences(X_train_s, y_train)

        if X_train_seq.shape[0] == 0:
            return {"train_losses": [], "val_losses": []}

        X_val_seq, y_val_seq = None, None
        if X_val is not None and y_val is not None:
            X_val_s = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_s, y_val)

        input_size = X_train.shape[1]
        self.net = TransformerGaussianNet(
            input_size=input_size,
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            n_layers=self.config["n_layers"],
            d_ff=self.config["d_ff"],
            dropout=self.config["dropout"],
            max_seq_len=self.config["sequence_length"] + 10,
            n_horizons=n_horizons,
        ).to(self.device)

        # Freeze first N layers if configured
        freeze_layers = self.config.get("freeze_layers", 0)
        if freeze_layers > 0:
            frozen = 0
            for name, param in self.net.named_parameters():
                if frozen < freeze_layers and ("input_proj" in name or "encoder" in name):
                    param.requires_grad = False
                    frozen += 1

        trainable_params = (
            filter(lambda p: p.requires_grad, self.net.parameters())
            if freeze_layers > 0
            else self.net.parameters()
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Warmup + cosine schedule
        def lr_lambda(step):
            warmup = self.config["warmup_steps"]
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(
                self.config["epochs"] * (len(X_train_seq) // self.config["batch_size"] + 1) - warmup, 1
            )
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        train_ds = TensorDataset(
            torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True,
        )

        # Mixed precision setup
        use_amp = self.config.get("use_mixed_precision", False) and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
        grad_clip = self.config["grad_clip"]

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self._train_losses = []
        self._val_losses = []

        for epoch in range(self.config["epochs"]):
            self.net.train()
            epoch_loss = 0.0
            n_b = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        preds = self.net(X_b)
                        loss = gaussian_nll_loss(preds, y_b)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = self.net(X_b)
                    loss = gaussian_nll_loss(preds, y_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                n_b += 1

            self._train_losses.append(epoch_loss / max(n_b, 1))

            if X_val_seq is not None and len(X_val_seq) > 0:
                self.net.eval()
                with torch.no_grad():
                    Xv = torch.from_numpy(X_val_seq).to(self.device)
                    yv = torch.from_numpy(y_val_seq).to(self.device)
                    val_loss = gaussian_nll_loss(self.net(Xv), yv).item()
                self._val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["patience"]:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_fitted = True
        return {
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
            "best_val_loss": best_val_loss,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (means, variances) of shape (n_valid, n_horizons)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X_s = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_s)
        if X_seq.shape[0] == 0:
            n_h = len(self.config["horizons"])
            return np.zeros((0, n_h)), np.ones((0, n_h))

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(X_seq).to(self.device)).cpu().numpy()

        means = out[:, :, 0]
        variances = np.exp(np.clip(out[:, :, 1], -10, 10))
        return means, variances

    def predict_distribution(self, X, n_samples=1000):
        means, variances = self.predict(X)
        stds = np.sqrt(variances)
        rng = np.random.RandomState(42)
        samples = rng.normal(
            loc=means[:, :, None], scale=stds[:, :, None],
            size=(*means.shape, n_samples),
        )
        return {"means": means, "variances": variances, "samples": samples}

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "config": self.config,
            "net_state": self.net.state_dict() if self.net else None,
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.config = state["config"]
        if state["scaler_mean"] is not None:
            self.scaler.mean_ = np.array(state["scaler_mean"])
            self.scaler.scale_ = np.array(state["scaler_scale"])
            self.scaler.n_features_in_ = len(self.scaler.mean_)
        if state["net_state"] is not None:
            n_features = state["net_state"]["input_proj.0.weight"].shape[1]
            self.net = TransformerGaussianNet(
                input_size=n_features,
                d_model=self.config["d_model"],
                n_heads=self.config["n_heads"],
                n_layers=self.config["n_layers"],
                d_ff=self.config["d_ff"],
                dropout=self.config["dropout"],
                n_horizons=len(self.config["horizons"]),
            ).to(self.device)
            self.net.load_state_dict(state["net_state"])
            self.is_fitted = True


# ---------------------------------------------------------------------------
# Factory for switching between LSTM and Transformer
# ---------------------------------------------------------------------------

def create_forecasting_model(model_type: str = "lstm", config: Optional[Dict] = None):
    """Factory function. Use model_type='lstm' or model_type='transformer'."""
    from models.lstm_gaussian import LSTMGaussianModel
    if model_type == "lstm":
        return LSTMGaussianModel(config)
    elif model_type == "transformer":
        return TransformerGaussianModel(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lstm' or 'transformer'.")
