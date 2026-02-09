"""
Multi-Task Loss Functions
--------------------------
Composite loss for the hybrid multi-modal model:
  - Quantile regression loss (pinball)
  - Binary cross-entropy for directional classification
  - Ranking loss (pairwise margin)
  - Calibration penalty (ensures quantile ordering)
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles: Optional[List[float]] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, n_quantiles)
            targets: (batch,) or (batch, 1)
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        errors = targets - predictions  # (batch, n_quantiles)
        loss = torch.max(
            self.quantiles * errors,
            (self.quantiles - 1) * errors,
        )
        return loss.mean()


class DirectionalLoss(nn.Module):
    """BCE loss for P(up) prediction."""

    def forward(self, p_up: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_up: (batch,) predicted probability of positive return
            targets: (batch,) actual returns (converted to binary)
        """
        labels = (targets > 0).float()
        return F.binary_cross_entropy(p_up.clamp(1e-6, 1 - 1e-6), labels)


class RankingLoss(nn.Module):
    """Pairwise margin ranking loss for cross-sectional ordering."""

    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch,) predicted returns
            targets: (batch,) actual returns
        """
        n = predictions.size(0)
        if n < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Sample pairs
        n_pairs = min(n * (n - 1) // 2, 256)
        idx_i = torch.randint(0, n, (n_pairs,), device=predictions.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=predictions.device)

        pred_diff = predictions[idx_i] - predictions[idx_j]
        target_sign = torch.sign(targets[idx_i] - targets[idx_j])

        loss = F.relu(self.margin - target_sign * pred_diff)
        return loss.mean()


class CalibrationPenalty(nn.Module):
    """Penalty for quantile crossing (ensures monotonic quantile ordering)."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, quantile_preds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantile_preds: (batch, n_quantiles) -- should be monotonically increasing
        """
        if quantile_preds.size(1) < 2:
            return torch.tensor(0.0, device=quantile_preds.device)

        diffs = quantile_preds[:, 1:] - quantile_preds[:, :-1]
        crossings = F.relu(-diffs)
        return self.weight * crossings.mean()


class MultiTaskLoss(nn.Module):
    """Combined loss for the hybrid model.

    Components:
      - quantile_loss: distributional forecasting
      - directional_loss: P(up) classification
      - ranking_loss: cross-sectional ordering
      - calibration_penalty: quantile monotonicity

    Weights are learnable (uncertainty-weighted multi-task learning).
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        ranking_margin: float = 0.01,
        calibration_weight: float = 0.5,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantiles)
        self.directional_loss = DirectionalLoss()
        self.ranking_loss = RankingLoss(ranking_margin)
        self.calibration_penalty = CalibrationPenalty(calibration_weight)

        if learnable_weights:
            # Log-variance parameters for uncertainty weighting
            self.log_vars = nn.Parameter(torch.zeros(4))
        else:
            self.log_vars = None

    def forward(
        self,
        quantile_preds: torch.Tensor,
        p_up: torch.Tensor,
        point_preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Args:
            quantile_preds: (batch, n_quantiles)
            p_up: (batch,) direction probability
            point_preds: (batch,) point estimate
            targets: (batch,) realized returns

        Returns:
            dict with 'total' loss and individual components.
        """
        l_quant = self.quantile_loss(quantile_preds, targets)
        l_dir = self.directional_loss(p_up, targets)
        l_rank = self.ranking_loss(point_preds, targets)
        l_calib = self.calibration_penalty(quantile_preds)

        if self.log_vars is not None:
            # Uncertainty-weighted sum: L_i / (2 * sigma_i^2) + log(sigma_i)
            precisions = torch.exp(-self.log_vars)
            total = (
                precisions[0] * l_quant + self.log_vars[0]
                + precisions[1] * l_dir + self.log_vars[1]
                + precisions[2] * l_rank + self.log_vars[2]
                + precisions[3] * l_calib + self.log_vars[3]
            )
        else:
            total = l_quant + l_dir + l_rank + l_calib

        return {
            "total": total,
            "quantile": l_quant.detach(),
            "directional": l_dir.detach(),
            "ranking": l_rank.detach(),
            "calibration": l_calib.detach(),
        }
