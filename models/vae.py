"""
Financial Variational Autoencoder (VAE)
----------------------------------------
Learns compressed latent factors from financial time-series features.

Architecture:
  - Encoder: MLP -> (mu, log_var) latent space
  - Decoder: MLP -> reconstructed features
  - Reparameterization trick for differentiable sampling
  - Beta-annealing: gradually increase KL weight during training
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FinancialVAE(nn.Module):
    """VAE for extracting latent factors from financial features.

    The latent space captures compressed representations of market state
    that can be fed as additional features to the fusion engine.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden layer size.
        latent_dim: Latent factor dimensionality.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10, 10)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed features."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        compute_reconstruction: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input features tensor.
            compute_reconstruction: If False, skip decoder (faster inference).

        Returns:
            Dict with keys: z (latent), mu, log_var, and optionally reconstruction.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        result = {
            "z": z,
            "mu": mu,
            "log_var": log_var,
        }
        if compute_reconstruction:
            result["reconstruction"] = self.decode(z)
        return result

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent factors (deterministic -- uses mu)."""
        mu, _ = self.encode(x)
        return mu


class VAELoss(nn.Module):
    """VAE loss with beta-annealing for KL divergence.

    Total loss = Reconstruction + beta * KL divergence

    Beta annealing schedule: linear warmup from 0 to beta_max over
    a specified number of steps, preventing posterior collapse.
    """

    def __init__(self, beta_max: float = 1.0, warmup_steps: int = 1000):
        super().__init__()
        self.beta_max = beta_max
        self.warmup_steps = warmup_steps
        self._step = 0

    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            reconstruction: decoded output
            original: input features
            mu: latent mean
            log_var: latent log-variance

        Returns:
            Dict with total, recon_loss, kl_loss
        """
        recon_loss = F.mse_loss(reconstruction, original)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Beta annealing
        beta = min(self.beta_max, self.beta_max * self._step / max(self.warmup_steps, 1))
        self._step += 1

        total = recon_loss + beta * kl_loss
        return {
            "total": total,
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "beta": beta,
        }


def pretrain_vae_cross_sectional(
    vae: FinancialVAE,
    panel: "np.ndarray",
    feature_cols: Optional[list] = None,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
) -> Dict[str, list]:
    """Pretrain VAE on cross-sectional data (all stocks at each date).

    Groups data by date, treats cross-sections as batches.
    After pretraining, VAE weights should be frozen in hybrid training.

    Args:
        vae: FinancialVAE instance.
        panel: 2D numpy array of features, shape (n_samples, n_features).
        feature_cols: Feature column names (unused, for documentation).
        epochs: Number of pretraining epochs.
        lr: Learning rate.
        batch_size: Batch size.
        device: Device string.

    Returns:
        Dict with loss history: {"total": [...], "recon": [...], "kl": [...]}.
    """
    import numpy as np

    vae = vae.to(device)
    vae.train()

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = VAELoss(beta_max=0.5, warmup_steps=200)

    # Convert to tensor
    X = torch.from_numpy(panel.astype(np.float32)).to(device)
    n = X.size(0)

    history = {"total": [], "recon": [], "kl": []}  # type: Dict[str, list]

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = X[batch_idx]

            optimizer.zero_grad()
            out = vae(batch, compute_reconstruction=True)
            loss_out = loss_fn(
                out["reconstruction"], batch, out["mu"], out["log_var"],
            )
            loss_out["total"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss_out["total"].item()
            epoch_recon += loss_out["recon_loss"].item()
            epoch_kl += loss_out["kl_loss"].item()
            n_batches += 1

        if n_batches > 0:
            history["total"].append(epoch_loss / n_batches)
            history["recon"].append(epoch_recon / n_batches)
            history["kl"].append(epoch_kl / n_batches)

    vae.eval()
    return history
