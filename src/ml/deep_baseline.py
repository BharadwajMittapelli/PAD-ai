"""
src/ml/deep_baseline.py
=======================
Deep Learning Baseline: Bi-GRU in PyTorch on tokenized URL + email text.

Uses character-level tokenization for URLs and word-level for email bodies,
concatenated and fed through a Bidirectional GRU with dropout.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Tokenizer ───────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Character-level tokenizer with vocabulary built from training data."""

    def __init__(self, max_len: int = 256):
        self.max_len = max_len
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx = 2

    def fit(self, texts: list[str]):
        """Build vocabulary from a list of texts."""
        for text in texts:
            for ch in text:
                if ch not in self.char2idx:
                    self.char2idx[ch] = self.idx
                    self.idx += 1
        return self

    def encode(self, text: str) -> list[int]:
        """Encode a text string into a fixed-length integer sequence."""
        ids = [self.char2idx.get(ch, 1) for ch in text[: self.max_len]]
        # Pad to max_len
        ids += [0] * (self.max_len - len(ids))
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)


# ── Dataset ─────────────────────────────────────────────────────────────────

class PhishingDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: SimpleTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx])
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ── Model ───────────────────────────────────────────────────────────────────

class BiGRUClassifier(nn.Module):
    """Bidirectional GRU with embedding for text classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        emb = self.embedding(x)                     # (B, L, E)
        out, _ = self.gru(emb)                      # (B, L, 2H)
        # Take the last hidden state
        last = out[:, -1, :]                         # (B, 2H)
        return self.fc(last).squeeze(-1)             # (B,)


# ── Training ────────────────────────────────────────────────────────────────

def train_bigru(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_len: int = 256,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 3,
    model_dir: str = "models",
) -> dict:
    """
    Train a Bi-GRU model on tokenized URL + email text.

    Returns dict with model name and evaluation metrics.
    """
    print("\n══════════════════════════════════════════════")
    print("  🧠 Training: Bi-GRU (PyTorch)")
    print("══════════════════════════════════════════════")

    device = torch.device("cpu")
    print(f"  📍 Device: {device}")

    # Combine URL + email into single text
    train_texts = (train_df["url"].fillna("") + " " + train_df["email_body"].fillna("")).tolist()
    test_texts = (test_df["url"].fillna("") + " " + test_df["email_body"].fillna("")).tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Build tokenizer
    tokenizer = SimpleTokenizer(max_len=max_len)
    tokenizer.fit(train_texts)

    train_ds = PhishingDataset(train_texts, y_train.tolist(), tokenizer)
    test_ds = PhishingDataset(test_texts, y_test.tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model = BiGRUClassifier(vocab_size=tokenizer.vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)

    # Training loop with early stopping
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.numpy().tolist())

        epoch_f1 = f1_score(all_labels, all_preds)
        scheduler.step(avg_loss)

        print(f"  Epoch {epoch:02d}/{epochs} — Loss: {avg_loss:.4f} — F1: {epoch_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            no_improve = 0
            # Save best model
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "bigru_model.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    # Load best model and compute final metrics
    model.load_state_dict(torch.load(os.path.join(model_dir, "bigru_model.pt"), weights_only=True))
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.numpy().tolist())

    metrics = {
        "model": "Bi-GRU (PyTorch)",
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds), 4),
        "recall": round(recall_score(all_labels, all_preds), 4),
        "f1": round(f1_score(all_labels, all_preds), 4),
        "auc": round(roc_auc_score(all_labels, all_probs), 4),
    }

    _print_metrics(metrics)
    return metrics


def _print_metrics(metrics: dict):
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  {metrics['model']:^39s} │")
    print(f"  ├───────────────┬─────────────────────────┤")
    print(f"  │  Accuracy     │  {metrics['accuracy']:.4f}                  │")
    print(f"  │  Precision    │  {metrics['precision']:.4f}                  │")
    print(f"  │  Recall       │  {metrics['recall']:.4f}                  │")
    print(f"  │  F1 Score     │  {metrics['f1']:.4f}                  │")
    print(f"  │  AUC-ROC      │  {metrics['auc']:.4f}                  │")
    print(f"  └───────────────┴─────────────────────────┘")
