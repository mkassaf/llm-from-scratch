"""
transformer.py — Stage 2: Pre-Training (Model Architecture)
-------------------------------------------------------------
A single Transformer block as used in GPT:
  LayerNorm → Multi-Head Attention → residual
  LayerNorm → Feed-Forward Network → residual
Book reference: Chapter 4 — Implementing a GPT Model from Scratch
"""

import torch
import torch.nn as nn
from torch import Tensor
from .attention import MultiHeadAttention


# ── GELU Activation (used in GPT-2) ───────────────────────────────────────────

class GELU(nn.Module):
    """Gaussian Error Linear Unit — smoother alternative to ReLU."""

    def forward(self, x: Tensor) -> Tensor:
        import math
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        )


# ── Position-wise Feed-Forward Network ────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU, applied to each position independently.
    Hidden dim is typically 4× the embedding dim.
    """

    def __init__(self, embed_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * embed_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── Transformer Block ──────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One Transformer decoder block (GPT-style, pre-LayerNorm):

        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Args:
        embed_dim      : embedding / model dimension
        num_heads      : number of attention heads
        context_length : max sequence length
        dropout        : dropout probability
        expansion      : FFN hidden-dim multiplier (default 4)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
        expansion: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, context_length, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, expansion, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Attention sub-layer with residual
        x = x + self.drop(self.attn(self.norm1(x)))
        # FFN sub-layer with residual
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    block = TransformerBlock(embed_dim=64, num_heads=4, context_length=16)
    x = torch.randn(2, 16, 64)
    out = block(x)
    print(f"TransformerBlock output: {out.shape}")    # (2, 16, 64)

    # Count parameters
    params = sum(p.numel() for p in block.parameters())
    print(f"Block parameters: {params:,}")
