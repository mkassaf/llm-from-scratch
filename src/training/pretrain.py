"""
pretrain.py — Stage 2: Pre-Training
--------------------------------------
Training loop for next-token prediction on raw text.
Book reference: Chapter 5 — Pre-Training on Unlabeled Data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss over the vocabulary.

    Args:
        logits  : (batch, seq_len, vocab_size)
        targets : (batch, seq_len)
    Returns:
        scalar loss
    """
    B, T, V = logits.shape
    return nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))


# ── Evaluation helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> Tuple[float, float]:
    """Returns (avg_loss, perplexity) over `num_batches` batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += compute_loss(logits, y).item()
        count += 1
    avg = total_loss / max(count, 1)
    return avg, torch.exp(torch.tensor(avg)).item()


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    eval_every: int = 200,
    save_path: str = "gpt_pretrained.pt",
) -> dict:
    """
    Main pre-training loop.

    Returns a history dict with train/val loss per evaluation step.
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "step": []}
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = compute_loss(logits, y)
            loss.backward()

            # Gradient clipping (important for stable training)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            if global_step % eval_every == 0:
                train_loss, train_ppl = evaluate(model, train_loader, device)
                val_loss,   val_ppl   = evaluate(model, val_loader,   device)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["step"].append(global_step)

                print(
                    f"Epoch {epoch:3d} | Step {global_step:6d} | "
                    f"Train loss {train_loss:.4f} (ppl {train_ppl:.1f}) | "
                    f"Val loss {val_loss:.4f} (ppl {val_ppl:.1f})"
                )
                model.train()

        print(f"--- Epoch {epoch} complete ---")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    return history


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")

    from src.model.gpt import GPT, GPTConfig
    from src.data.tokenizer import BPETokenizer
    from src.data.dataset import create_dataloader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Tiny smoke test with random token IDs
    cfg = GPTConfig(vocab_size=50257, context_length=32,
                    embed_dim=64, num_heads=2, num_layers=2)
    model = GPT(cfg)

    dummy_ids = list(range(200))  # replace with real tokenized text
    from src.data.dataset import GPTDataset
    from torch.utils.data import DataLoader as DL

    ds = GPTDataset(dummy_ids, context_length=32, stride=8)
    loader = DL(ds, batch_size=4, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    history = train(model, loader, loader, opt, DEVICE,
                    num_epochs=2, eval_every=5, save_path="smoke_test.pt")
    print("Smoke test passed!")
