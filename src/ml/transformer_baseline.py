"""
src/ml/transformer_baseline.py
==============================
Advanced Baseline: Fine-tune BERT-base for phishing classification.

Uses Hugging Face Transformers with the Trainer API.
Runs on CPU (no GPU required).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def train_bert(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "bert-base-uncased",
    max_len: int = 128,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    model_dir: str = "models/bert",
) -> dict:
    """
    Fine-tune BERT-base for phishing classification.

    Input format: "[URL] {url} [SEP] {email_body}"
    Runs on CPU. Returns dict with evaluation metrics.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset as HFDataset

    print("\n══════════════════════════════════════════════")
    print("  🤖 Training: BERT-base (Hugging Face)")
    print("══════════════════════════════════════════════")
    print(f"  📍 Device: cpu")
    print(f"  📋 Model: {model_name}")
    print(f"  ⚠️  CPU training — this may take 15–30 min...")

    # ── Prepare data ────────────────────────────────────────────────────────
    def _format_input(row):
        url = row.get("url", "") or ""
        body = row.get("email_body", "") or ""
        return f"[URL] {url} [SEP] {body}"

    train_texts = [_format_input(row) for _, row in train_df.iterrows()]
    test_texts = [_format_input(row) for _, row in test_df.iterrows()]
    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    # ── Tokenize ────────────────────────────────────────────────────────────
    print("  ⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("  ⏳ Tokenizing data...")
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
    )

    # Convert to HF Datasets
    train_dataset = HFDataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels,
    })
    test_dataset = HFDataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels,
    })

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # ── Model ───────────────────────────────────────────────────────────────
    print("  ⏳ Loading BERT-base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # ── Training args ───────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=50,
        report_to="none",  # Don't send to wandb/tensorboard
        no_cuda=True,       # Force CPU
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print("  🚀 Starting BERT training...")
    trainer.train()

    # ── Evaluate ────────────────────────────────────────────────────────────
    print("  📊 Evaluating...")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    # Softmax for AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_prob = probs[:, 1]
    y_test = np.array(test_labels)

    metrics = {
        "model": "BERT-base (Fine-tuned)",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    # Save
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

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
