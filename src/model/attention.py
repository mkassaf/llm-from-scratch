"""
attention.py — Stage 2: Pre-Training (Model Architecture)
-----------------------------------------------------------
Implements:
  - Scaled Dot-Product Self-Attention (single head)
  - Causal (masked) Self-Attention
  - Multi-Head Attention
Book reference: Chapter 3 — Coding Attention Mechanisms
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── 1. Scaled Dot-Product Attention (building block) ──────────────────────────

def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor = None, dropout: float = 0.0
) -> Tensor:
    """
    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

    Args:
        Q, K, V : (batch, heads, seq, d_k)
        mask    : (seq, seq) boolean — True positions are masked out
    Returns:
        context : (batch, heads, seq, d_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    if dropout > 0.0:
        weights = F.dropout(weights, p=dropout, training=True)

    return torch.matmul(weights, V)


# ── 2. Single-Head Causal Self-Attention ──────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Single-head self-attention with a causal (autoregressive) mask.
    Each token can only attend to itself and previous tokens.
    """

    def __init__(self, embed_dim: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        # Causal mask: upper triangle = True (masked)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq, embed_dim)
        B, T, C = x.shape
        Q = self.W_q(x).unsqueeze(1)   # (B, 1, T, C)
        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)

        context = scaled_dot_product_attention(
            Q, K, V,
            mask=self.mask[:T, :T],
            dropout=self.dropout if self.training else 0.0,
        )
        context = context.squeeze(1)   # (B, T, C)
        return self.out_proj(context)


# ── 3. Multi-Head Attention ────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Splits the embedding dimension across `num_heads` heads, runs attention
    in parallel, then concatenates and projects.

    Args:
        embed_dim      : total embedding dimension (must be divisible by num_heads)
        num_heads      : number of attention heads
        context_length : maximum sequence length (for causal mask)
        dropout        : attention dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Project and reshape to (B, H, T, D)
        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        context = scaled_dot_product_attention(
            Q, K, V,
            mask=self.mask[:T, :T],
            dropout=self.dropout if self.training else 0.0,
        )

        # Merge heads: (B, H, T, D) → (B, T, C)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context)


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, C = 2, 10, 64   # batch=2, seq=10, embed=64

    x = torch.randn(B, T, C)

    single = CausalSelfAttention(embed_dim=C, context_length=T)
    out = single(x)
    print(f"CausalSelfAttention output: {out.shape}")    # (2, 10, 64)

    mha = MultiHeadAttention(embed_dim=C, num_heads=4, context_length=T)
    out = mha(x)
    print(f"MultiHeadAttention output : {out.shape}")    # (2, 10, 64)
