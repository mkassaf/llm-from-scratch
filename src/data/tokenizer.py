"""
tokenizer.py — Stage 1: Data Preprocessing
-------------------------------------------
A simple character-level tokenizer and a wrapper around tiktoken (BPE).
Book reference: Chapter 2 — Working with Text Data
"""

import re
from typing import List, Dict


# ── Simple character-level tokenizer ──────────────────────────────────────────

class SimpleTokenizer:
    """Splits text on whitespace/punctuation and maps tokens to integer IDs."""

    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    @classmethod
    def from_corpus(cls, text: str) -> "SimpleTokenizer":
        """Build vocabulary from a raw text corpus."""
        # Split on spaces and punctuation, keep punctuation as tokens
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        vocab = {token: idx for idx, token in enumerate(sorted(set(tokens)))}
        # Add special tokens
        vocab.setdefault("<|unk|>", len(vocab))
        vocab.setdefault("<|endoftext|>", len(vocab))
        return cls(vocab)

    def encode(self, text: str) -> List[int]:
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        unk_id = self.str_to_int.get("<|unk|>", 0)
        return [self.str_to_int.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.int_to_str.get(i, "<|unk|>") for i in ids]
        text = " ".join(tokens)
        # Remove space before punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# ── BPE tokenizer via tiktoken (GPT-2 / GPT-4 compatible) ─────────────────────

class BPETokenizer:
    """
    Wraps tiktoken's GPT-2 encoding.
    Install: pip install tiktoken
    """

    def __init__(self, encoding_name: str = "gpt2"):
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError("Run: pip install tiktoken")

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str, allowed_special: set = {"<|endoftext|>"}) -> List[int]:
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Hello, world! This is a tokenizer demo."

    print("=== Simple Tokenizer ===")
    tok = SimpleTokenizer.from_corpus(sample)
    ids = tok.encode(sample)
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))

    print("\n=== BPE Tokenizer (tiktoken / GPT-2) ===")
    bpe = BPETokenizer()
    ids = bpe.encode(sample)
    print("Encoded:", ids)
    print("Decoded:", bpe.decode(ids))
    print(f"Vocab size: {bpe.vocab_size:,}")
