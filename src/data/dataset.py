"""
dataset.py — Stage 1: Data Preprocessing
------------------------------------------
Sliding-window dataset that produces (input, target) token pairs for
next-token prediction, plus token + positional embedding layers.
Book reference: Chapter 2 — Working with Text Data
"""

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List


# ── Sliding-window Dataset ─────────────────────────────────────────────────────

class GPTDataset(Dataset):
    """
    Converts a token ID list into overlapping (input, target) windows.

    Example (context_length=4, stride=1):
        tokens = [1, 2, 3, 4, 5, 6]
        x[0] = [1, 2, 3, 4]   y[0] = [2, 3, 4, 5]
        x[1] = [2, 3, 4, 5]   y[1] = [3, 4, 5, 6]
    """

    def __init__(self, token_ids: List[int], context_length: int, stride: int):
        self.input_ids: List[Tensor] = []
        self.target_ids: List[Tensor] = []

        ids = torch.tensor(token_ids, dtype=torch.long)
        for i in range(0, len(ids) - context_length, stride):
            self.input_ids.append(ids[i : i + context_length])
            self.target_ids.append(ids[i + 1 : i + context_length + 1])

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    tokenizer,
    batch_size: int = 4,
    context_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """End-to-end helper: raw text → DataLoader."""
    token_ids = tokenizer.encode(text)
    dataset = GPTDataset(token_ids, context_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


# ── Embedding Layer (tokens + positions) ──────────────────────────────────────

class EmbeddingLayer(nn.Module):
    """
    Combines token embeddings and learnable positional embeddings.

    Args:
        vocab_size     : number of unique tokens
        embed_dim      : embedding dimensionality (e.g. 768 for GPT-2 small)
        context_length : maximum sequence length
    """

    def __init__(self, vocab_size: int, embed_dim: int, context_length: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(context_length, embed_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch, seq_len)
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device)
        return self.token_emb(token_ids) + self.pos_emb(positions)


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Fake token IDs for demonstration
    token_ids = list(range(50))
    ds = GPTDataset(token_ids, context_length=8, stride=4)
    print(f"Dataset size: {len(ds)} samples")
    x, y = ds[0]
    print(f"Input : {x.tolist()}")
    print(f"Target: {y.tolist()}")

    # Embedding demo
    emb = EmbeddingLayer(vocab_size=256, embed_dim=64, context_length=16)
    batch = torch.randint(0, 256, (2, 8))
    out = emb(batch)
    print(f"\nEmbedding output shape: {out.shape}")  # (2, 8, 64)
