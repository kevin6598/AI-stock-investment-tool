"""
Hybrid Multi-Modal Network
-----------------------------
Master orchestrator that wires all sub-modules together:

  Input Features
       |
  +----+----+----+----+
  |    |    |    |    |
  TI  LSTM  VAE  LLM  Ticker
  Emb  Enc  Enc  Emb  Emb
  |    |    |    |    |
  +----+----+----+----+
       |
  Fusion Engine
       |
  +----+----+
  |         |
  Retail    Aux
  Head      Head

Sub-modules:
  - Temporal encoder (LSTM or Transformer)
  - TechnicalIndicatorEmbedding (group-wise MLPs)
  - FinancialVAE (latent factor extraction)
  - Sentiment embedding (from NLP features)
  - Ticker embedding (for cross-stock learning)
  - MultiModalFusionEngine
  - RetailDirectionalHead + AuxiliaryFactorHead
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalEncoder(nn.Module):
    """Shared temporal encoder (LSTM with attention).

    Processes sequential features and returns a fixed-size representation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, hidden_size)
        """
        h = self.input_proj(x)
        lstm_out, _ = self.lstm(h)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        combined = self.norm(lstm_out + attn_out)
        return combined[:, -1, :]  # last timestep


class SentimentEncoder(nn.Module):
    """Encode NLP sentiment features into a dense vector."""

    def __init__(self, input_dim: int, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridMultiModalNet(nn.Module):
    """Master network orchestrating all sub-modules.

    Args:
        n_features: Total number of input features.
        n_tickers: Number of tickers for embedding lookup.
        seq_len: Sequence length for temporal encoder.
        hidden_dim: Hidden dimension for temporal encoder.
        ti_group_dims: Dict of group_name -> n_features for TI embedding.
        ti_embed_dim: Per-group embedding dimension.
        vae_latent_dim: VAE latent space size.
        sentiment_dim: Number of NLP sentiment features.
        fusion_dim: Fusion output dimension.
        n_quantiles: Number of output quantiles.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_features: int,
        n_tickers: int = 50,
        seq_len: int = 60,
        hidden_dim: int = 128,
        ti_group_dims: Optional[Dict[str, int]] = None,
        ti_embed_dim: int = 32,
        vae_latent_dim: int = 16,
        sentiment_dim: int = 27,
        fusion_dim: int = 128,
        n_quantiles: int = 7,
        dropout: float = 0.2,
        ablation_config: Optional[Dict[str, bool]] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.ablation_config = ablation_config or {}

        # Ticker embedding
        self.ticker_embedding = nn.Embedding(
            num_embeddings=n_tickers + 1,  # +1 for zero-shot (unseen tickers)
            embedding_dim=32,
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Technical indicator embedding
        if ti_group_dims is not None and len(ti_group_dims) > 0:
            from models.indicator_embedding import TechnicalIndicatorEmbedding
            self.ti_embedding = TechnicalIndicatorEmbedding(
                group_dims=ti_group_dims,
                embed_dim=ti_embed_dim,
                fusion_dim=64,
                dropout=dropout,
            )
            ti_out_dim = 64
        else:
            self.ti_embedding = None
            ti_out_dim = 64
            self._ti_fallback = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.GELU(),
            )

        # VAE for latent factors
        from models.vae import FinancialVAE
        self.vae = FinancialVAE(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            latent_dim=vae_latent_dim,
            dropout=dropout,
        )

        # Sentiment encoder
        self.sentiment_encoder = SentimentEncoder(
            input_dim=sentiment_dim,
            embed_dim=64,
            dropout=dropout,
        )

        # Fusion engine
        from models.fusion import MultiModalFusionEngine
        self.fusion = MultiModalFusionEngine(
            temporal_dim=hidden_dim,
            indicator_dim=ti_out_dim,
            sentiment_dim=64,
            vae_dim=vae_latent_dim,
            fusion_dim=fusion_dim,
            n_heads=4,
            dropout=dropout,
        )

        # Combine fusion output with ticker embedding
        self.pre_head = nn.Sequential(
            nn.Linear(fusion_dim + 32, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        from models.output_heads import RetailDirectionalHead, AuxiliaryFactorHead
        self.retail_head = RetailDirectionalHead(
            input_dim=fusion_dim,
            n_quantiles=n_quantiles,
            dropout=dropout,
        )
        self.aux_head = AuxiliaryFactorHead(
            input_dim=fusion_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
        ticker_ids: torch.Tensor,
        sentiment_features: Optional[torch.Tensor] = None,
        ti_group_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_seq: (batch, seq_len, n_features) sequential input
            x_static: (batch, n_features) latest timestep features
            ticker_ids: (batch,) integer ticker IDs
            sentiment_features: (batch, sentiment_dim) NLP features
            ti_group_inputs: Dict of group_name -> (batch, group_dim)

        Returns:
            Dict with retail and auxiliary outputs, plus VAE reconstruction.
        """
        batch_size = x_seq.size(0)

        # 1. Temporal encoding
        temporal_emb = self.temporal_encoder(x_seq)  # (batch, hidden_dim)
        if self.ablation_config.get("disable_temporal"):
            temporal_emb = torch.zeros_like(temporal_emb)

        # 2. Technical indicator embedding
        if self.ti_embedding is not None and ti_group_inputs is not None:
            ti_emb = self.ti_embedding(ti_group_inputs)  # (batch, 64)
        elif hasattr(self, '_ti_fallback'):
            ti_emb = self._ti_fallback(x_static)
        else:
            ti_emb = torch.zeros(batch_size, 64, device=x_seq.device)
        if self.ablation_config.get("disable_ti_embedding"):
            ti_emb = torch.zeros_like(ti_emb)

        # 3. VAE latent factors (skip reconstruction at inference)
        vae_out = self.vae(x_static, compute_reconstruction=self.training)
        vae_emb = vae_out["z"]  # (batch, latent_dim)
        if self.ablation_config.get("disable_vae"):
            vae_emb = torch.zeros_like(vae_emb)

        # 4. Sentiment embedding
        if sentiment_features is not None:
            sent_emb = self.sentiment_encoder(sentiment_features)  # (batch, 64)
        else:
            sent_emb = torch.zeros(batch_size, 64, device=x_seq.device)
        if self.ablation_config.get("disable_sentiment"):
            sent_emb = torch.zeros_like(sent_emb)

        # 5. Fusion
        fused = self.fusion(temporal_emb, ti_emb, sent_emb, vae_emb)  # (batch, fusion_dim)

        # 6. Ticker embedding
        ticker_emb = self.ticker_embedding(ticker_ids)  # (batch, 32)
        if self.ablation_config.get("disable_ticker_embedding"):
            ticker_emb = torch.zeros_like(ticker_emb)

        # 7. Combine and predict
        combined = torch.cat([fused, ticker_emb], dim=-1)
        h = self.pre_head(combined)

        retail_out = self.retail_head(h)
        aux_out = self.aux_head(h)

        result = {
            **retail_out,
            **{"aux_{}".format(k): v for k, v in aux_out.items()},
            "vae_mu": vae_out["mu"],
            "vae_log_var": vae_out["log_var"],
        }
        if "reconstruction" in vae_out:
            result["vae_reconstruction"] = vae_out["reconstruction"]
        return result


# ============================================================================
# GatedHybridNet -- IC-Optimized Hybrid Architecture
# ============================================================================


class LightPriceEncoder(nn.Module):
    """Lightweight LSTM encoder for price/technical features (~50K params).

    2-layer unidirectional LSTM -> LayerNorm -> Dropout -> 64-dim output.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, hidden_size) temporal embedding
        """
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # last timestep
        return self.dropout(self.norm(last))


class LightNLPEncoder(nn.Module):
    """Lightweight MLP encoder for NLP features (~10K params).

    Linear(27->32) -> GELU -> Dropout -> Linear(32->32)
    """

    def __init__(self, input_dim: int = 27, embed_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) NLP features
        Returns:
            (batch, embed_dim) NLP embedding
        """
        return self.net(x)


class RegimeConditionedGate(nn.Module):
    """Regime-aware gating mechanism for fusing NLP into price signal (~300 params).

    Input: temporal_emb(64) + nlp_emb(32) + vol(1) + regime(1) = 98
    Output: scalar gate g in [0, 1]
    Fusion: F = g * project(S) + (1-g) * T
    """

    def __init__(self, temporal_dim: int = 64, nlp_dim: int = 32):
        super().__init__()
        gate_input_dim = temporal_dim + nlp_dim + 2  # +vol +regime
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 1),
            nn.Sigmoid(),
        )
        self.nlp_project = nn.Linear(nlp_dim, temporal_dim)

    def forward(
        self,
        temporal_emb: torch.Tensor,
        nlp_emb: torch.Tensor,
        vol: torch.Tensor,
        regime: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            temporal_emb: (batch, 64)
            nlp_emb: (batch, 32)
            vol: (batch, 1) volatility indicator
            regime: (batch, 1) regime indicator

        Returns:
            (fused, gate_values): fused (batch, 64), gate_values (batch, 1)
        """
        gate_input = torch.cat([temporal_emb, nlp_emb, vol, regime], dim=-1)
        g = self.gate(gate_input)  # (batch, 1)

        nlp_projected = self.nlp_project(nlp_emb)  # (batch, 64)
        fused = g * nlp_projected + (1 - g) * temporal_emb
        return fused, g


class CrossSectionalAttentionLayer(nn.Module):
    """Single-head attention across stocks at the same date (~16K params).

    Q = K = V = Linear(64->64), scaled dot-product + residual + LayerNorm.
    """

    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, embed_dim) -- treated as (1, batch, embed_dim) sequence

        Returns:
            (batch, embed_dim) with cross-sectional attention applied
        """
        # Treat batch dim as sequence for cross-sectional attention
        x_seq = x.unsqueeze(0)  # (1, batch, embed_dim)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        out = self.norm(x_seq + attn_out)
        return out.squeeze(0)  # (batch, embed_dim)


class GatedHybridNet(nn.Module):
    """IC-optimized hybrid network with gated NLP fusion (~200-400K params).

    Architecture:
        price_features -> LightPriceEncoder -> T(64)
        nlp_features   -> LightNLPEncoder   -> S(32)
        (T, S, vol, regime) -> RegimeConditionedGate -> g
        F = g * project(S) + (1-g) * T         [64-dim]
        F -> CrossSectionalAttentionLayer -> F2 [64-dim]
        F2 -> Linear(64->32) -> ReLU -> Linear(32->1) -> point estimate
    """

    def __init__(
        self,
        n_price_features: int,
        n_nlp_features: int = 27,
        temporal_hidden: int = 64,
        nlp_embed: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_price_features = n_price_features
        self.n_nlp_features = n_nlp_features

        self.price_encoder = LightPriceEncoder(
            input_size=n_price_features,
            hidden_size=temporal_hidden,
            dropout=dropout,
        )
        self.nlp_encoder = LightNLPEncoder(
            input_dim=n_nlp_features,
            embed_dim=nlp_embed,
            dropout=dropout,
        )
        self.gate = RegimeConditionedGate(
            temporal_dim=temporal_hidden,
            nlp_dim=nlp_embed,
        )
        self.cross_attention = CrossSectionalAttentionLayer(
            embed_dim=temporal_hidden,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(temporal_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        nlp_features: torch.Tensor,
        vol: torch.Tensor,
        regime: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            price_seq: (batch, seq_len, n_price_features) sequential price data
            nlp_features: (batch, n_nlp_features) NLP features
            vol: (batch, 1) volatility indicator
            regime: (batch, 1) regime indicator

        Returns:
            Dict with 'prediction' (batch, 1) and 'gate' (batch, 1)
        """
        T = self.price_encoder(price_seq)       # (batch, 64)
        S = self.nlp_encoder(nlp_features)      # (batch, 32)
        F, g = self.gate(T, S, vol, regime)     # (batch, 64), (batch, 1)
        F2 = self.cross_attention(F)            # (batch, 64)
        pred = self.head(F2)                    # (batch, 1)
        return {"prediction": pred, "gate": g}
