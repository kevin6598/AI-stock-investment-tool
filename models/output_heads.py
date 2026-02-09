"""
Output Heads
--------------
Dual output heads for the hybrid multi-modal model:

1. RetailDirectionalHead - Primary output for retail investors:
   - P(up): probability of positive return
   - Confidence: model's self-assessed confidence
   - Quantiles: distributional forecast (7 quantiles)
   - Hold signal: should the investor hold?
   - Risk score: normalized 0-1 risk assessment

2. AuxiliaryFactorHead - Secondary outputs for model training:
   - Volatility forecast
   - Regime logits (bull/bear/crisis)
   - Factor returns (market, size, value, momentum)
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetailDirectionalHead(nn.Module):
    """Primary prediction head for retail investment signals.

    Outputs:
        p_up: P(positive return) in [0, 1]
        confidence: Model confidence in [0, 1]
        quantiles: (batch, 7) quantile forecasts
        hold_signal: P(hold) in [0, 1]
        risk_score: Risk level in [0, 1]
    """

    def __init__(
        self,
        input_dim: int,
        n_quantiles: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden = input_dim // 2

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Direction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

        # Quantile head
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_quantiles),
        )

        # Hold signal head
        self.hold_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )

        # Risk score head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "p_up": self.direction_head(h).squeeze(-1),
            "confidence": self.confidence_head(h).squeeze(-1),
            "quantiles": self.quantile_head(h),
            "hold_signal": self.hold_head(h).squeeze(-1),
            "risk_score": self.risk_head(h).squeeze(-1),
        }


class AuxiliaryFactorHead(nn.Module):
    """Auxiliary output head for multi-task regularization.

    Predicting related quantities helps the shared representation
    learn more robust features.

    Outputs:
        vol_forecast: predicted future volatility
        regime_logits: (batch, 3) logits for [bull, bear, crisis]
        factor_returns: (batch, 4) predicted factor returns [mkt, size, value, mom]
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 3,
        n_factors: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden = input_dim // 2

        # Volatility forecast
        self.vol_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # volatility is always positive
        )

        # Regime classification
        self.regime_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_regimes),
        )

        # Factor return prediction
        self.factor_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_factors),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "vol_forecast": self.vol_head(x).squeeze(-1),
            "regime_logits": self.regime_head(x),
            "factor_returns": self.factor_head(x),
        }
