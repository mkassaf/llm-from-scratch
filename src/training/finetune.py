"""
finetune.py — Stage 3: Fine-Tuning
-------------------------------------
Fine-tuning a pre-trained GPT for:
  (A) Text classification  (e.g. spam detection)
  (B) Instruction following (e.g. alpaca-style)
Book reference: Chapter 6 & 7
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


# ── A. Classification Fine-Tuning ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Replaces the LM head with a linear classifier.
    Uses the last token's hidden state as the sequence representation.
    """

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(embed_dim, num_classes)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: (batch, seq_len, embed_dim)
        # Use the last non-padding token
        pooled = hidden_states[:, -1, :]   # (batch, embed_dim)
        return self.fc(self.drop(pooled))  # (batch, num_classes)


def attach_classifier(gpt_model: nn.Module, num_classes: int, freeze_base: bool = True):
    """
    Attach a classification head to a pre-trained GPT model.

    Args:
        gpt_model   : pre-trained GPT instance
        num_classes : number of output classes
        freeze_base : if True, freeze all transformer weights (only train head)
    Returns:
        modified model
    """
    embed_dim = gpt_model.cfg.embed_dim

    if freeze_base:
        for param in gpt_model.parameters():
            param.requires_grad = False

    # Replace LM head with classification head
    gpt_model.lm_head = ClassificationHead(embed_dim, num_classes)
    return gpt_model


# ── B. Instruction Fine-Tuning Dataset ────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning (alpaca-style format).

    Each entry is a dict with keys: "instruction", "input" (optional), "output".
    Formats them as:

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
    """

    PROMPT_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )
    PROMPT_NO_INPUT = (
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        context_length: int = 512,
    ):
        self.samples: List[Dict[str, Tensor]] = []
        for entry in data:
            if entry.get("input", "").strip():
                text = self.PROMPT_TEMPLATE.format(**entry)
            else:
                text = self.PROMPT_NO_INPUT.format(**entry)

            ids = tokenizer.encode(text)[:context_length]
            input_ids  = torch.tensor(ids[:-1], dtype=torch.long)
            target_ids = torch.tensor(ids[1:],  dtype=torch.long)
            self.samples.append({"input_ids": input_ids, "target_ids": target_ids})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["input_ids"], self.samples[idx]["target_ids"]


def collate_fn(batch, pad_id: int = 0):
    """Pad variable-length sequences in a batch to the same length."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    inputs  = torch.stack([nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_id) for x in inputs])
    targets = torch.stack([nn.functional.pad(y, (0, max_len - y.size(0)), value=-100)   for y in targets])
    return inputs, targets


# ── Fine-Tuning Loop (shared for both variants) ───────────────────────────────

def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 3,
    task: str = "lm",          # "lm" for instruction / "clf" for classification
    num_classes: int = 2,
) -> None:
    model.to(device)

    loss_fn = (
        nn.CrossEntropyLoss(ignore_index=-100) if task == "lm"
        else nn.CrossEntropyLoss()
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)

            if task == "lm":
                B, T, V = out.shape
                loss = loss_fn(out.view(B * T, V), y.view(B * T))
            else:
                loss = loss_fn(out, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} — train loss: {avg:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fine-tuning module loaded. Run via notebook or training script.")
    print("Supported tasks: 'lm' (instruction) | 'clf' (classification)")
