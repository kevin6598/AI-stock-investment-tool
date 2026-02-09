"""
LSTM-Gaussian Model
-------------------
LSTM with Gaussian output layer (mean + variance) for probabilistic forecasting.

Key features:
  - Outputs distribution parameters (mu, sigma) not point estimates
  - Gaussian negative log-likelihood loss
  - Multi-horizon support via separate output heads
  - Early stopping with patience
  - Model save/load for versioning

Input tensor shape: (batch, sequence_length, n_features)
Output: (batch, n_horizons, 2) where last dim = [mean, log_variance]
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import json
import logging

logger = logging.getLogger(__name__)


class GaussianLSTMNet(nn.Module):
    """LSTM encoder with Gaussian output heads for each horizon."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_horizons: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection with batch norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
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
            bidirectional=False,
        )

        # Temporal attention over LSTM outputs
        self.attn_query = nn.Linear(hidden_size, hidden_size)
        self.attn_key = nn.Linear(hidden_size, hidden_size)
        self.attn_scale = hidden_size ** 0.5

        # Per-horizon Gaussian output heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_size // 2, 2),  # [mean, log_variance]
            )
            for _ in range(n_horizons)
        ])

        self.n_horizons = n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (batch, n_horizons, 2) where [..., 0] = mean, [..., 1] = log_var
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Temporal attention: attend over all timesteps
        query = self.attn_query(lstm_out[:, -1:, :])   # (batch, 1, hidden)
        keys = self.attn_key(lstm_out)                   # (batch, seq_len, hidden)
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / self.attn_scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, 1, seq_len)
        context = torch.bmm(attn_weights, lstm_out)         # (batch, 1, hidden)
        context = context.squeeze(1)                         # (batch, hidden)

        # Per-horizon outputs
        outputs = []
        for head in self.horizon_heads:
            out = head(context)  # (batch, 2)
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # (batch, n_horizons, 2)


def gaussian_nll_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood loss.

    Args:
        predictions: (batch, n_horizons, 2) - [mean, log_variance]
        targets: (batch, n_horizons) - actual returns

    Returns:
        Scalar loss.

    Loss = 0.5 * [log_var + (y - mu)^2 / exp(log_var)]
    """
    mu = predictions[:, :, 0]
    log_var = predictions[:, :, 1]

    # Clamp log_var to prevent numerical issues
    log_var = torch.clamp(log_var, min=-10, max=10)

    loss = 0.5 * (log_var + (targets - mu) ** 2 / torch.exp(log_var))
    return loss.mean()


class LSTMGaussianModel:
    """Complete LSTM-Gaussian model with training, prediction, and save/load."""

    def __init__(self, config: Optional[Dict] = None):
        defaults = {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "epochs": 150,
            "patience": 20,
            "sequence_length": 60,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "horizons": [21, 63, 126],  # 1M, 3M, 6M in trading days
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

    def _create_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding window sequences from 2D panel data."""
        seq_len = self.config["sequence_length"]
        n = X.shape[0]

        if n <= seq_len:
            return np.empty((0, seq_len, X.shape[1])), None

        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(seq_len, n):
            X_seq.append(X[i - seq_len:i])
            if y is not None:
                y_seq.append(y[i])

        X_seq = np.array(X_seq, dtype=np.float32)
        if y_seq is not None:
            y_seq = np.array(y_seq, dtype=np.float32)
        return X_seq, y_seq

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """Train the LSTM-Gaussian model.

        Args:
            X_train: (n_samples, n_features) training features.
            y_train: (n_samples, n_horizons) or (n_samples,) targets.
            X_val: validation features.
            y_val: validation targets.

        Returns:
            Dict with training history.
        """
        # Ensure y is 2D
        n_horizons = len(self.config["horizons"])
        if y_train.ndim == 1:
            y_train = np.repeat(y_train[:, None], n_horizons, axis=1)
        if y_val is not None and y_val.ndim == 1:
            y_val = np.repeat(y_val[:, None], n_horizons, axis=1)

        # Scale features
        X_train_s = self.scaler.fit_transform(X_train)
        X_train_seq, y_train_seq = self._create_sequences(X_train_s, y_train)

        if X_train_seq.shape[0] == 0:
            logger.warning("Insufficient training data for sequence creation")
            return {"train_losses": [], "val_losses": []}

        if X_val is not None and y_val is not None:
            X_val_s = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_s, y_val)
        else:
            X_val_seq, y_val_seq = None, None

        # Build network
        input_size = X_train.shape[1]
        self.net = GaussianLSTMNet(
            input_size=input_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            n_horizons=n_horizons,
        ).to(self.device)

        # Freeze first N layers if configured
        freeze_layers = self.config.get("freeze_layers", 0)
        if freeze_layers > 0:
            frozen = 0
            for name, param in self.net.named_parameters():
                if frozen < freeze_layers and ("input_proj" in name or "lstm" in name):
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6,
        )

        # DataLoader
        train_ds = TensorDataset(
            torch.from_numpy(X_train_seq),
            torch.from_numpy(y_train_seq),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True,
        )

        # Mixed precision setup
        use_amp = self.config.get("use_mixed_precision", False) and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
        grad_clip = self.config["grad_clip"]

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self._train_losses = []
        self._val_losses = []

        for epoch in range(self.config["epochs"]):
            # Train
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0
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

                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            avg_train = epoch_loss / max(n_batches, 1)
            self._train_losses.append(avg_train)

            # Validate
            if X_val_seq is not None and len(X_val_seq) > 0:
                self.net.eval()
                with torch.no_grad():
                    Xv = torch.from_numpy(X_val_seq).to(self.device)
                    yv = torch.from_numpy(y_val_seq).to(self.device)
                    val_preds = self.net(Xv)
                    val_loss = gaussian_nll_loss(val_preds, yv).item()
                self._val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["patience"]:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_fitted = True
        return {
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self._train_losses),
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty.

        Args:
            X: (n_samples, n_features) features.

        Returns:
            (means, variances) each of shape (n_valid, n_horizons).
            n_valid = n_samples - sequence_length.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        X_s = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_s)

        if X_seq.shape[0] == 0:
            n_h = len(self.config["horizons"])
            return np.zeros((0, n_h)), np.ones((0, n_h))

        self.net.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X_seq).to(self.device)
            out = self.net(Xt).cpu().numpy()  # (n, n_horizons, 2)

        means = out[:, :, 0]
        log_vars = np.clip(out[:, :, 1], -10, 10)
        variances = np.exp(log_vars)

        return means, variances

    def predict_distribution(self, X: np.ndarray, n_samples: int = 1000
                             ) -> Dict[str, np.ndarray]:
        """Sample from predicted distribution for Monte Carlo methods.

        Returns:
            Dict with 'means', 'variances', 'samples' (n_valid, n_horizons, n_samples)
        """
        means, variances = self.predict(X)
        stds = np.sqrt(variances)

        rng = np.random.RandomState(42)
        samples = rng.normal(
            loc=means[:, :, None],
            scale=stds[:, :, None],
            size=(*means.shape, n_samples),
        )

        return {"means": means, "variances": variances, "samples": samples}

    def save(self, path: str) -> None:
        """Save model state and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "config": self.config,
            "net_state": self.net.state_dict() if self.net else None,
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load model state and config."""
        state = torch.load(path, map_location=self.device)
        self.config = state["config"]

        if state["scaler_mean"] is not None:
            self.scaler.mean_ = np.array(state["scaler_mean"])
            self.scaler.scale_ = np.array(state["scaler_scale"])
            self.scaler.n_features_in_ = len(self.scaler.mean_)

        if state["net_state"] is not None:
            n_features = state["net_state"]["input_proj.0.weight"].shape[1]
            n_horizons = len(self.config["horizons"])
            self.net = GaussianLSTMNet(
                input_size=n_features,
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                dropout=self.config["dropout"],
                n_horizons=n_horizons,
            ).to(self.device)
            self.net.load_state_dict(state["net_state"])
            self.is_fitted = True

        self._train_losses = state.get("train_losses", [])
        self._val_losses = state.get("val_losses", [])
