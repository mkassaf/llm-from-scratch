"""
gpt.py — Stage 2: Pre-Training (Model Architecture)
-----------------------------------------------------
Full GPT model: Embedding → N × TransformerBlock → LayerNorm → LM Head
Book reference: Chapter 4 — Implementing a GPT Model from Scratch
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from .transformer import TransformerBlock
from ..data.dataset import EmbeddingLayer


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    """Hyperparameters for the GPT model."""
    vocab_size:     int   = 50257   # GPT-2 vocabulary
    context_length: int   = 1024    # max sequence length
    embed_dim:      int   = 768     # model width
    num_heads:      int   = 12      # attention heads
    num_layers:     int   = 12      # transformer blocks
    dropout:        float = 0.1
    expansion:      int   = 4       # FFN hidden-dim multiplier

    # Predefined configs (matching GPT-2 sizes)
    @classmethod
    def small(cls):  return cls(embed_dim=768,  num_heads=12, num_layers=12)
    @classmethod
    def medium(cls): return cls(embed_dim=1024, num_heads=16, num_layers=24)
    @classmethod
    def large(cls):  return cls(embed_dim=1280, num_heads=20, num_layers=36)


# ── GPT Model ─────────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Decoder-only Transformer (GPT architecture).

    Forward pass returns logits of shape (batch, seq_len, vocab_size).
    Use cross-entropy loss against the target (shifted-by-one) token IDs
    for next-token prediction pre-training.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = EmbeddingLayer(cfg.vocab_size, cfg.embed_dim, cfg.context_length)
        self.drop       = nn.Dropout(cfg.dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.context_length,
                             cfg.dropout, cfg.expansion)
            for _ in range(cfg.num_layers)
        ])
        self.norm       = nn.LayerNorm(cfg.embed_dim)
        self.lm_head    = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Weight tying: share embedding weights with output projection
        self.lm_head.weight = self.embedding.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch, seq_len)
        x = self.drop(self.embedding(token_ids))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)   # (batch, seq_len, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tensor:
        """
        Greedy / top-k sampled autoregressive generation.

        Args:
            prompt_ids     : (1, seq_len) starting token IDs
            max_new_tokens : number of tokens to generate
            temperature    : >1 = more random, <1 = more confident
            top_k          : restrict sampling to top-k logits (0 = greedy)
        Returns:
            Tensor of shape (1, seq_len + max_new_tokens)
        """
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            # Crop to context window
            ids_cond = ids[:, -self.cfg.context_length:]
            logits = self(ids_cond)[:, -1, :] / temperature  # (1, vocab_size)

            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

        return ids

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embedding.pos_emb.weight.numel()
        return n


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = GPTConfig(
        vocab_size=50257, context_length=64,
        embed_dim=128, num_heads=4, num_layers=2,
    )
    model = GPT(cfg)
    print(f"GPT parameters: {model.num_parameters():,}")

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    print(f"Logits shape: {logits.shape}")   # (2, 16, 50257)

    # Generation
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(prompt, max_new_tokens=10, top_k=10)
    print(f"Generated sequence length: {out.shape[1]}")
