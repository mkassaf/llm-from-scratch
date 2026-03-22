# Building LLMs from Scratch

A hands-on learning repository following the book **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka.

---

## Learning Roadmap

This repo is structured around the book's three core stages:

```
Raw Text
   │
   ▼
Stage 1: Data Preprocessing
   │  - Tokenization (BPE / character-level)
   │  - Token Embeddings
   │  - Positional Encodings
   ▼
Stage 2: Pre-Training
   │  - Self-Attention Mechanism
   │  - Multi-Head Attention
   │  - Transformer Block
   │  - GPT Architecture
   │  - Training Loop (next-token prediction)
   ▼
Stage 3: Fine-Tuning
      - Instruction Fine-Tuning
      - Classification (e.g. spam detection)
      - RLHF basics
      ▼
  Your LLM!
```

---

## Repository Structure

```
llm-from-scratch/
│
├── notebooks/              # Jupyter notebooks (one per topic)
│   ├── 01_tokenization.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_attention.ipynb
│   ├── 04_transformer_block.ipynb
│   ├── 05_gpt_architecture.ipynb
│   ├── 06_pretraining.ipynb
│   └── 07_finetuning.ipynb
│
├── src/
│   ├── data/               # Data preprocessing utilities
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── model/              # Model architecture
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   └── gpt.py
│   └── training/           # Training & fine-tuning
│       ├── pretrain.py
│       └── finetune.py
│
├── data/
│   └── samples/            # Sample text files for experimentation
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/mkassaf/llm-from-scratch.git
cd llm-from-scratch

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

---

## Topics Covered

| # | Topic | Notebook | Source Module |
|---|-------|----------|--------------|
| 1 | Tokenization & BPE | `01_tokenization.ipynb` | `src/data/tokenizer.py` |
| 2 | Token & Positional Embeddings | `02_embeddings.ipynb` | `src/data/dataset.py` |
| 3 | Self-Attention & Multi-Head Attention | `03_attention.ipynb` | `src/model/attention.py` |
| 4 | Transformer Block (LayerNorm, FFN, Dropout) | `04_transformer_block.ipynb` | `src/model/transformer.py` |
| 5 | Full GPT Architecture | `05_gpt_architecture.ipynb` | `src/model/gpt.py` |
| 6 | Pre-Training (next-token prediction) | `06_pretraining.ipynb` | `src/training/pretrain.py` |
| 7 | Fine-Tuning (classification + instruction) | `07_finetuning.ipynb` | `src/training/finetune.py` |

---

## Key Concepts

**Attention Mechanism** — The core of the Transformer. It lets the model weigh the importance of different tokens in the sequence to understand context and meaning.

**Positional Encoding** — Since Transformers have no inherent sense of order, positional encodings inject sequence position information into the embeddings.

**Pre-Training** — The model learns on large unlabeled text by predicting the next token. This creates a general-purpose foundation model.

**Fine-Tuning** — The pre-trained model is adapted to a specific task (e.g., classification, chat) using smaller labeled datasets.

---

## Progress Tracker

- [ ] Chapter 1 — Understanding Large Language Models
- [ ] Chapter 2 — Working with Text Data (Tokenization)
- [ ] Chapter 3 — Coding Attention Mechanisms
- [ ] Chapter 4 — Implementing a GPT Model from Scratch
- [ ] Chapter 5 — Pre-Training on Unlabeled Data
- [ ] Chapter 6 — Fine-Tuning for Classification
- [ ] Chapter 7 — Fine-Tuning with Human Feedback

---

## References

- 📘 [Build a Large Language Model (From Scratch) — Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- 🤗 [Hugging Face Transformers](https://github.com/huggingface/transformers)
- 🔥 [PyTorch Documentation](https://pytorch.org/docs/)
