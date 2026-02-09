"""
Technical Indicator Embedding
------------------------------
Group-wise MLPs that embed raw technical indicators into dense vectors,
then fuse them with a learned fusion layer.

Groups:
  - Momentum (RSI, MACD, momentum returns, stochastic)
  - Oscillators (Bollinger position, CCI, stochastic, MFI)
  - Volume (volume ratio, OBV, volume spikes, VWAP deviation)
  - Volatility (ATR, volatility windows, vol clustering)
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class GroupMLP(nn.Module):
    """Small MLP for a single feature group."""

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        hidden = max(input_dim * 2, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TechnicalIndicatorEmbedding(nn.Module):
    """Group-wise embedding of technical indicators.

    Each feature group is processed by its own MLP, then all group
    embeddings are fused via a learned fusion layer.

    Args:
        group_dims: Dict mapping group name to number of input features.
        embed_dim: Output embedding dimension per group.
        fusion_dim: Output dimension after fusion.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        group_dims: Dict[str, int],
        embed_dim: int = 32,
        fusion_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.group_names = sorted(group_dims.keys())
        self.embed_dim = embed_dim

        # Per-group MLPs
        self.group_mlps = nn.ModuleDict({
            name: GroupMLP(dim, embed_dim, dropout)
            for name, dim in group_dims.items()
        })

        # Fusion layer: concatenated group embeddings -> fusion_dim
        total_embed = embed_dim * len(group_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_embed, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Group attention weights (learned importance of each group)
        self.group_attention = nn.Linear(total_embed, len(group_dims))

    def forward(
        self, group_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            group_inputs: Dict of group_name -> (batch, group_dim) tensors.

        Returns:
            (batch, fusion_dim) fused embedding.
        """
        embeddings = []
        for name in self.group_names:
            if name in group_inputs:
                emb = self.group_mlps[name](group_inputs[name])
            else:
                batch_size = next(iter(group_inputs.values())).size(0)
                emb = torch.zeros(
                    batch_size, self.embed_dim,
                    device=next(iter(group_inputs.values())).device,
                )
            embeddings.append(emb)

        # Concatenate and fuse
        concat = torch.cat(embeddings, dim=-1)  # (batch, total_embed)

        # Attention-weighted fusion
        attn_weights = torch.softmax(self.group_attention(concat), dim=-1)  # (batch, n_groups)
        weighted = torch.stack(embeddings, dim=1)  # (batch, n_groups, embed_dim)
        attn_out = (weighted * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, embed_dim)

        # Combine direct fusion and attention
        fused = self.fusion(concat)
        return fused

    def get_group_weights(self, group_inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get current attention weights per group (for interpretability)."""
        embeddings = []
        for name in self.group_names:
            if name in group_inputs:
                emb = self.group_mlps[name](group_inputs[name])
            else:
                batch_size = next(iter(group_inputs.values())).size(0)
                emb = torch.zeros(batch_size, self.embed_dim,
                                  device=next(iter(group_inputs.values())).device)
            embeddings.append(emb)
        concat = torch.cat(embeddings, dim=-1)
        attn = torch.softmax(self.group_attention(concat), dim=-1).mean(dim=0)
        return {name: float(attn[i]) for i, name in enumerate(self.group_names)}
