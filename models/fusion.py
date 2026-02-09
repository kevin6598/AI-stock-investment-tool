"""
Multi-Modal Fusion Engine
---------------------------
Combines embeddings from different modalities (text, time-series, VAE latent,
technical indicators) using multiple fusion strategies:

1. CrossAttentionFusion - Queries one modality using another as keys/values
2. GatedFusion - Learned gating to control modality contributions
3. FiLMConditioning - Feature-wise Linear Modulation for conditioning
4. Dynamic weight layer - Soft routing based on input characteristics
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between two modalities.

    Modality A attends to modality B (A queries, B keys/values).
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, dim) or (batch, seq, dim)
            key_value: (batch, dim) or (batch, seq, dim)
        """
        # Ensure 3D for attention
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)

        attn_out, _ = self.attn(query, key_value, key_value)
        x = self.norm(query + attn_out)
        x = self.norm2(x + self.ffn(x))

        # Squeeze back if original was 2D
        if x.size(1) == 1:
            x = x.squeeze(1)
        return x


class GatedFusion(nn.Module):
    """Gated fusion of multiple modality embeddings.

    Learns a gate per modality that controls how much each modality
    contributes to the fused representation.
    """

    def __init__(self, dim: int, n_modalities: int, dropout: float = 0.1):
        super().__init__()
        self.n_modalities = n_modalities

        # Gate network: takes concatenated inputs, outputs per-modality gates
        self.gate = nn.Sequential(
            nn.Linear(dim * n_modalities, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_modalities),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modality_embeddings: list of (batch, dim) tensors

        Returns:
            (batch, dim) fused embedding
        """
        concat = torch.cat(modality_embeddings, dim=-1)  # (batch, dim * n)
        gates = torch.softmax(self.gate(concat), dim=-1)  # (batch, n)

        stacked = torch.stack(modality_embeddings, dim=1)  # (batch, n, dim)
        gated = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # (batch, dim)

        return self.output_proj(gated)


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation.

    Conditions one representation using another via:
        output = gamma * input + beta
    where gamma, beta are generated from the conditioning signal.
    """

    def __init__(self, input_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, input_dim)
        self.beta_net = nn.Linear(cond_dim, input_dim)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) features to modulate
            condition: (batch, cond_dim) conditioning signal
        """
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return gamma * x + beta


class MultiModalFusionEngine(nn.Module):
    """Master fusion engine combining multiple strategies.

    Architecture:
      1. Cross-attention between temporal and indicator embeddings
      2. FiLM conditioning of temporal features on sentiment
      3. Gated fusion of all modalities
      4. Dynamic weight layer for final combination

    Args:
        temporal_dim: Dimension of temporal encoder output (LSTM/Transformer).
        indicator_dim: Dimension of TI embedding output.
        sentiment_dim: Dimension of sentiment embedding.
        vae_dim: Dimension of VAE latent vector.
        fusion_dim: Output dimension of the fusion engine.
        n_heads: Attention heads for cross-attention.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        temporal_dim: int = 128,
        indicator_dim: int = 64,
        sentiment_dim: int = 64,
        vae_dim: int = 16,
        fusion_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Projection layers to align dimensions
        self.proj_temporal = nn.Linear(temporal_dim, fusion_dim) if temporal_dim != fusion_dim else nn.Identity()
        self.proj_indicator = nn.Linear(indicator_dim, fusion_dim) if indicator_dim != fusion_dim else nn.Identity()
        self.proj_sentiment = nn.Linear(sentiment_dim, fusion_dim) if sentiment_dim != fusion_dim else nn.Identity()
        self.proj_vae = nn.Linear(vae_dim, fusion_dim) if vae_dim != fusion_dim else nn.Identity()

        # Cross-attention: temporal <-> indicator
        self.cross_attn = CrossAttentionFusion(fusion_dim, n_heads, dropout)

        # FiLM: condition temporal on sentiment
        self.film = FiLMConditioning(fusion_dim, fusion_dim)

        # Gated fusion of all 4 modalities
        self.gated_fusion = GatedFusion(fusion_dim, n_modalities=4, dropout=dropout)

        # Dynamic weight layer
        self.dynamic_weights = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Final output
        self.output = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        temporal_emb: torch.Tensor,
        indicator_emb: torch.Tensor,
        sentiment_emb: torch.Tensor,
        vae_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temporal_emb: (batch, temporal_dim)
            indicator_emb: (batch, indicator_dim)
            sentiment_emb: (batch, sentiment_dim)
            vae_emb: (batch, vae_dim)

        Returns:
            (batch, fusion_dim) fused representation
        """
        # Project to common dimension
        t = self.proj_temporal(temporal_emb)
        i = self.proj_indicator(indicator_emb)
        s = self.proj_sentiment(sentiment_emb)
        v = self.proj_vae(vae_emb)

        # Cross-attention: temporal queries indicator context
        t_cross = self.cross_attn(t, i)

        # FiLM: condition temporal on sentiment
        t_film = self.film(t_cross, s)

        # Gated fusion
        gated = self.gated_fusion([t_film, i, s, v])

        # Dynamic combination
        concat_all = torch.cat([t_film, i, s, v], dim=-1)
        dynamic = self.dynamic_weights(concat_all)

        # Merge gated and dynamic paths
        merged = torch.cat([gated, dynamic], dim=-1)
        return self.output(merged)
