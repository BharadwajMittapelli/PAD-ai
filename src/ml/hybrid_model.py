"""
src/ml/hybrid_model.py
======================
PAD-ai Phase 2 — Hybrid Multi-Model Architecture

Three sub-models fused via a stacking meta-learner:
    A. DistilBERT fine-tuned text classifier
    B. XGBoost on 40 hand-crafted URL features
    C. Stacking ensemble (Logistic Regression meta-learner)

Usage:
    from src.ml.hybrid_model import (
        DistilBERTPhishingClassifier,
        URLXGBoostClassifier,
        HybridStackingEnsemble,
    )
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ═════════════════════════════════════════════════════════════════════════════
# Sub-Model A: DistilBERT Fine-tuned Text Classifier
# ═════════════════════════════════════════════════════════════════════════════

class DistilBERTPhishingClassifier:
    """
    Fine-tuned DistilBERT for binary phishing classification.

    Input format: "[URL] {url} [SEP] {email_body}"
    Optimised for high recall on the phishing class.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_len: int = 128,
        model_dir: str = "models/hybrid/distilbert",
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _format_input(self, url: str, email_body: str) -> str:
        """Format URL + email body into a single input string."""
        url = url or ""
        body = email_body or ""
        return f"[URL] {url} [SEP] {body}"

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 16,
    ) -> dict:
        """
        Fine-tune DistilBERT for phishing classification.

        Parameters
        ----------
        train_df, val_df : pd.DataFrame
            Must have columns: url, email_body, label.
        epochs : int
            Training epochs (3-5 recommended).
        lr : float
            Learning rate (2e-5 default for transformers).
        batch_size : int
            Per-device batch size.

        Returns
        -------
        dict
            Training metrics from evaluation.
        """
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset as HFDataset
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
        )

        print("\n══════════════════════════════════════════════")
        print("  🤖 Training: DistilBERT Text Classifier")
        print("══════════════════════════════════════════════")

        # Prepare texts
        train_texts = [
            self._format_input(row.get("url", ""), row.get("email_body", ""))
            for _, row in train_df.iterrows()
        ]
        val_texts = [
            self._format_input(row.get("url", ""), row.get("email_body", ""))
            for _, row in val_df.iterrows()
        ]

        train_labels = train_df["label"].tolist()
        val_labels = val_df["label"].tolist()

        # Tokenize
        print(f"  ⏳ Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print("  ⏳ Tokenizing data...")
        train_enc = self.tokenizer(
            train_texts, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt",
        )
        val_enc = self.tokenizer(
            val_texts, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt",
        )

        # Create HF Datasets
        train_ds = HFDataset.from_dict({
            "input_ids": train_enc["input_ids"],
            "attention_mask": train_enc["attention_mask"],
            "labels": train_labels,
        })
        val_ds = HFDataset.from_dict({
            "input_ids": val_enc["input_ids"],
            "attention_mask": val_enc["attention_mask"],
            "labels": val_labels,
        })
        train_ds.set_format("torch")
        val_ds.set_format("torch")

        # Model
        print(f"  ⏳ Loading {self.model_name} model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2,
        )

        # Training args — optimised for recall
        os.makedirs(self.model_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_dir, "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=os.path.join(self.model_dir, "logs"),
            logging_steps=50,
            report_to="none",
            no_cuda=True,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "f1": f1_score(labels, preds, zero_division=0),
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        print("  🚀 Starting DistilBERT training...")
        trainer.train()

        # Evaluate
        results = trainer.evaluate()
        print(f"  ✅ DistilBERT F1: {results.get('eval_f1', 0):.4f}")

        # Save
        trainer.save_model(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        self._is_trained = True

        return results

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict phishing probabilities for a list of texts.

        Parameters
        ----------
        texts : list of str
            Pre-formatted input texts.

        Returns
        -------
        np.ndarray
            Shape (n_samples,) — probability of phishing class.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        self.model.eval()
        encoded = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).numpy()

        return probs[:, 1]  # probability of phishing class

    def predict_single(self, url: str, email_body: str) -> Tuple[float, int]:
        """Predict a single sample. Returns (phishing_prob, predicted_label)."""
        text = self._format_input(url, email_body)
        prob = self.predict_proba([text])[0]
        label = int(prob > 0.5)
        return float(prob), label

    def save(self, path: str = None):
        """Save model and tokenizer."""
        path = path or self.model_dir
        if self.model is not None and self.tokenizer is not None:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def load(self, path: str = None):
        """Load model and tokenizer from disk."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        path = path or self.model_dir
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()
        self._is_trained = True
        print(f"  ✅ Loaded DistilBERT from {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Sub-Model B: XGBoost URL Feature Classifier
# ═════════════════════════════════════════════════════════════════════════════

class URLXGBoostClassifier:
    """
    XGBoost classifier trained on 40 hand-crafted URL features.

    Uses Optuna hyperparameter tuning for optimal performance.
    """

    def __init__(self, model_path: str = "models/hybrid/xgb_url.joblib"):
        self.model_path = model_path
        self.model = None
        self._is_trained = False
        self.best_params = {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_optuna_trials: int = 50,
    ) -> dict:
        """
        Train XGBoost with Optuna hyperparameter tuning.

        Parameters
        ----------
        X_train, X_val : np.ndarray
            Feature matrices (40 URL features).
        y_train, y_val : np.ndarray
            Binary labels.
        n_optuna_trials : int
            Number of Optuna optimisation trials.

        Returns
        -------
        dict
            Evaluation metrics on validation set.
        """
        import optuna
        from xgboost import XGBClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print("\n══════════════════════════════════════════════")
        print("  🌳 Training: XGBoost URL Classifier (Optuna)")
        print("══════════════════════════════════════════════")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
            }
            clf = XGBClassifier(
                **params, random_state=42,
                eval_metric="logloss", verbosity=0,
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            return f1_score(y_val, preds)

        print(f"  ⏳ Running Optuna ({n_optuna_trials} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

        self.best_params = study.best_params
        print(f"  ✅ Best F1: {study.best_value:.4f}")

        # Retrain with best params
        self.model = XGBClassifier(
            **self.best_params, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        self.model.fit(X_train, y_train)
        self._is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]

        metrics = {
            "model": "XGBoost-URL (Optuna)",
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_val, y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_val, y_prob), 4),
        }

        # Save
        self.save()
        print(f"  💾 Saved XGBoost model to {self.model_path}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict phishing probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with 40 URL features.

        Returns
        -------
        np.ndarray
            Shape (n_samples,) — probability of phishing class.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, url_features: np.ndarray) -> Tuple[float, int]:
        """Predict a single sample. Returns (phishing_prob, label)."""
        X = url_features.reshape(1, -1) if url_features.ndim == 1 else url_features
        prob = self.predict_proba(X)[0]
        return float(prob), int(prob > 0.5)

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = joblib.load(path)
        self._is_trained = True
        print(f"  ✅ Loaded XGBoost from {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Stacking Ensemble (Meta-Learner)
# ═════════════════════════════════════════════════════════════════════════════

class HybridStackingEnsemble:
    """
    Stacking meta-learner that combines predictions from:
    - DistilBERT text classifier
    - XGBoost URL classifier
    - AI-Generated detector score
    + top URL features

    Uses Logistic Regression as the meta-learner for interpretability.
    """

    def __init__(self, model_path: str = "models/hybrid/meta_learner.joblib"):
        self.model_path = model_path
        self.model = None
        self._is_trained = False
        self.feature_names = [
            "bert_prob", "xgb_prob", "ai_score",
            "url_length", "domain_length", "num_dots", "num_hyphens",
            "subdomain_count", "suspicious_keyword_count",
            "has_ip_address", "is_https", "url_entropy", "digit_ratio",
        ]

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _build_meta_features(
        self,
        bert_probs: np.ndarray,
        xgb_probs: np.ndarray,
        ai_scores: np.ndarray,
        url_features_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Build the meta-feature matrix for the stacking ensemble.

        Parameters
        ----------
        bert_probs : np.ndarray  — shape (n,)
        xgb_probs : np.ndarray   — shape (n,)
        ai_scores : np.ndarray   — shape (n,)
        url_features_df : pd.DataFrame
            Must contain the 10 selected URL feature columns.

        Returns
        -------
        np.ndarray
            Shape (n, 13) meta-feature matrix.
        """
        # Select the top URL features for meta-learning
        url_feature_cols = [
            "url_length", "domain_length", "num_dots", "num_hyphens",
            "subdomain_count", "suspicious_keyword_count",
            "has_ip_address", "is_https", "url_entropy", "digit_ratio",
        ]

        url_feats = np.zeros((len(bert_probs), len(url_feature_cols)))
        for i, col in enumerate(url_feature_cols):
            if col in url_features_df.columns:
                url_feats[:, i] = url_features_df[col].values

        meta_X = np.column_stack([
            bert_probs.reshape(-1, 1),
            xgb_probs.reshape(-1, 1),
            ai_scores.reshape(-1, 1),
            url_feats,
        ])
        return meta_X

    def train(
        self,
        bert_probs: np.ndarray,
        xgb_probs: np.ndarray,
        ai_scores: np.ndarray,
        url_features_df: pd.DataFrame,
        y_true: np.ndarray,
    ) -> dict:
        """
        Train the meta-learner on out-of-fold predictions.

        Parameters
        ----------
        bert_probs, xgb_probs, ai_scores : np.ndarray
            Predicted probabilities from each sub-model.
        url_features_df : pd.DataFrame
            URL features for meta-feature construction.
        y_true : np.ndarray
            Ground truth labels.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        )

        print("\n══════════════════════════════════════════════")
        print("  🔗 Training: Stacking Meta-Learner")
        print("══════════════════════════════════════════════")

        meta_X = self._build_meta_features(
            bert_probs, xgb_probs, ai_scores, url_features_df,
        )

        # Logistic Regression meta-learner
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",  # bias towards recall
        )
        self.model.fit(meta_X, y_true)
        self._is_trained = True

        # Evaluate on training data (real eval done via cross-validation)
        y_pred = self.model.predict(meta_X)
        y_prob = self.model.predict_proba(meta_X)[:, 1]

        metrics = {
            "model": "Hybrid Stacking Ensemble",
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_true, y_prob), 4),
        }

        # Save
        self.save()
        print(f"  ✅ Meta-learner trained — F1: {metrics['f1']:.4f}")
        return metrics

    def predict(
        self,
        bert_probs: np.ndarray,
        xgb_probs: np.ndarray,
        ai_scores: np.ndarray,
        url_features_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using the stacking ensemble.

        Returns
        -------
        (labels, probabilities)
            Predicted labels and phishing probabilities.
        """
        if not self._is_trained:
            raise RuntimeError("Meta-learner not trained. Call train() or load() first.")

        meta_X = self._build_meta_features(
            bert_probs, xgb_probs, ai_scores, url_features_df,
        )
        labels = self.model.predict(meta_X)
        probs = self.model.predict_proba(meta_X)[:, 1]
        return labels, probs

    def predict_single(
        self,
        bert_prob: float,
        xgb_prob: float,
        ai_score: float,
        url_features: dict,
    ) -> Tuple[bool, float]:
        """
        Predict a single sample.

        Returns
        -------
        (is_phishing, confidence)
        """
        url_df = pd.DataFrame([url_features])
        labels, probs = self.predict(
            np.array([bert_prob]),
            np.array([xgb_prob]),
            np.array([ai_score]),
            url_df,
        )
        return bool(labels[0]), float(probs[0])

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = joblib.load(path)
        self._is_trained = True
        print(f"  ✅ Loaded meta-learner from {path}")
