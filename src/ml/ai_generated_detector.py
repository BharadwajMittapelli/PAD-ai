"""
src/ml/ai_generated_detector.py
===============================
PAD-ai Phase 2 — AI-Generated Phishing Email Detector

Detects LLM-generated phishing emails using three signals:
    1. Perplexity scoring (DistilGPT-2)
    2. Stylometric feature analysis (12 features)
    3. Fine-tuned Random Forest classifier

This is the KEY NOVELTY of the PAD-ai project.

Usage:
    from src.ml.ai_generated_detector import AIGeneratedDetector

    detector = AIGeneratedDetector()
    result = detector.predict("Dear valued customer...")
"""

import os
import sys
import math
import re
import string
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ═════════════════════════════════════════════════════════════════════════════
# Signal 1: Perplexity Scoring
# ═════════════════════════════════════════════════════════════════════════════

class PerplexityScorer:
    """
    Compute per-token perplexity using a causal language model.

    LLM-generated text tends to have LOWER perplexity (more predictable),
    while human-written text has HIGHER perplexity (more variable).
    """

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load the language model."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  ⏳ Loading {self.model_name} for perplexity scoring...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._loaded = True
        print(f"  ✅ {self.model_name} loaded for perplexity scoring")

    def compute_perplexity(self, text: str, max_length: int = 512) -> float:
        """
        Compute the perplexity of a text string.

        Parameters
        ----------
        text : str
            Input text to score.
        max_length : int
            Maximum token length for the model.

        Returns
        -------
        float
            Perplexity score. Lower = more predictable (likely AI).
            Returns -1.0 on error.
        """
        import torch

        if not text or not text.strip():
            return -1.0

        self._load_model()

        try:
            encodings = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_length, padding=False,
            )
            input_ids = encodings["input_ids"].to(self.device)

            if input_ids.size(1) < 2:
                return -1.0

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

            perplexity = math.exp(loss)
            return round(perplexity, 4)

        except Exception as e:
            print(f"  ⚠️ Perplexity computation failed: {e}")
            return -1.0

    def compute_batch(self, texts: List[str]) -> np.ndarray:
        """Compute perplexity for a batch of texts."""
        scores = []
        for i, text in enumerate(texts):
            score = self.compute_perplexity(text)
            scores.append(score)
            if (i + 1) % 50 == 0:
                print(f"    Perplexity: {i+1}/{len(texts)} computed")
        return np.array(scores, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# Signal 2: Stylometric Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════

# Common English stopwords (subset to avoid NLTK dependency at import time)
_STOPWORDS = frozenset([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been",
    "has", "had", "did", "does", "doing", "am",
])


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _split_words(text: str) -> List[str]:
    """Split text into words, removing punctuation."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def _compute_shannon_entropy(sequence: List[str]) -> float:
    """Compute Shannon entropy of a sequence of tokens."""
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    n = len(sequence)
    return -sum(
        (count / n) * math.log2(count / n)
        for count in counts.values()
    )


def extract_stylometric_features(text: str) -> Dict[str, float]:
    """
    Extract 12 stylometric features from an email text.

    These features capture writing style differences between
    human-written and AI-generated text.

    Parameters
    ----------
    text : str
        Email body text.

    Returns
    -------
    dict
        12 stylometric features.
    """
    if not text or not text.strip():
        return {k: 0.0 for k in STYLOMETRIC_FEATURE_NAMES}

    sentences = _split_sentences(text)
    words = _split_words(text)
    n_words = len(words)
    n_chars = len(text)

    features = {}

    # ── 1. Average sentence length (in words) ────────────────────────────
    sent_lengths = [len(_split_words(s)) for s in sentences]
    features["avg_sentence_length"] = (
        round(np.mean(sent_lengths), 4) if sent_lengths else 0.0
    )

    # ── 2. Sentence length variance ──────────────────────────────────────
    # AI text tends to have LOWER variance (more uniform)
    features["sentence_length_variance"] = (
        round(np.var(sent_lengths), 4) if len(sent_lengths) > 1 else 0.0
    )

    # ── 3. Type-token ratio (unique word ratio) ─────────────────────────
    # AI text often has a higher unique word ratio
    unique_words = set(words)
    features["unique_word_ratio"] = (
        round(len(unique_words) / max(n_words, 1), 4)
    )

    # ── 4. Average word length ───────────────────────────────────────────
    features["avg_word_length"] = (
        round(np.mean([len(w) for w in words]), 4) if words else 0.0
    )

    # ── 5. Punctuation ratio ─────────────────────────────────────────────
    punct_count = sum(1 for c in text if c in string.punctuation)
    features["punctuation_ratio"] = round(punct_count / max(n_chars, 1), 4)

    # ── 6. Uppercase ratio ───────────────────────────────────────────────
    upper_count = sum(1 for c in text if c.isupper())
    features["uppercase_ratio"] = round(upper_count / max(n_chars, 1), 4)

    # ── 7. Stopword ratio ────────────────────────────────────────────────
    stopword_count = sum(1 for w in words if w in _STOPWORDS)
    features["stopword_ratio"] = round(stopword_count / max(n_words, 1), 4)

    # ── 8. Hapax legomena ratio (words appearing only once) ──────────────
    word_counts = Counter(words)
    hapax = sum(1 for count in word_counts.values() if count == 1)
    features["hapax_legomena_ratio"] = round(hapax / max(n_words, 1), 4)

    # ── 9. Yule's K measure (vocabulary richness) ────────────────────────
    # Higher K = less diverse vocabulary
    if n_words > 0:
        freq_spectrum = Counter(word_counts.values())
        m1 = n_words
        m2 = sum(i * i * freq for i, freq in freq_spectrum.items())
        yule_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0.0
        features["yule_k_measure"] = round(yule_k, 4)
    else:
        features["yule_k_measure"] = 0.0

    # ── 10. Flesch Reading Ease ──────────────────────────────────────────
    # AI text tends to score higher (easier to read)
    n_syllables = sum(_count_syllables(w) for w in words)
    n_sentences = max(len(sentences), 1)
    if n_words > 0:
        fre = (
            206.835
            - 1.015 * (n_words / n_sentences)
            - 84.6 * (n_syllables / n_words)
        )
        features["flesch_reading_ease"] = round(fre, 4)
    else:
        features["flesch_reading_ease"] = 0.0

    # ── 11. Automated Readability Index ──────────────────────────────────
    if n_words > 0 and n_sentences > 0:
        ari = (
            4.71 * (n_chars / n_words)
            + 0.5 * (n_words / n_sentences)
            - 21.43
        )
        features["automated_readability_index"] = round(ari, 4)
    else:
        features["automated_readability_index"] = 0.0

    # ── 12. Bigram entropy ───────────────────────────────────────────────
    # Measures predictability of word sequences
    if len(words) > 1:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        features["bigram_entropy"] = round(_compute_shannon_entropy(bigrams), 4)
    else:
        features["bigram_entropy"] = 0.0

    return features


# Feature names for consistent ordering
STYLOMETRIC_FEATURE_NAMES: List[str] = [
    "avg_sentence_length", "sentence_length_variance", "unique_word_ratio",
    "avg_word_length", "punctuation_ratio", "uppercase_ratio",
    "stopword_ratio", "hapax_legomena_ratio", "yule_k_measure",
    "flesch_reading_ease", "automated_readability_index", "bigram_entropy",
]


def _count_syllables(word: str) -> int:
    """Approximate syllable count for English words."""
    word = word.lower().strip()
    if not word:
        return 0

    # Simple heuristic: count vowel groups
    count = len(re.findall(r'[aeiouy]+', word))
    # Adjust for silent e
    if word.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)


def stylometric_features_to_array(features: Dict[str, float]) -> List[float]:
    """Convert stylometric feature dict to ordered array."""
    return [float(features.get(f, 0.0)) for f in STYLOMETRIC_FEATURE_NAMES]


# ═════════════════════════════════════════════════════════════════════════════
# Signal 3: AI-Generated Classifier
# ═════════════════════════════════════════════════════════════════════════════

class AIGeneratedDetector:
    """
    Detects AI-generated phishing emails by combining:
    - Perplexity scoring (optional, requires model download)
    - 12 stylometric features
    - Random Forest classifier

    Provides a single ``predict()`` method that returns whether
    the email was likely AI-generated along with a confidence score.
    """

    def __init__(
        self,
        model_path: str = "models/ai_detector/ai_detector.joblib",
        use_perplexity: bool = True,
    ):
        self.model_path = model_path
        self.use_perplexity = use_perplexity
        self.model = None
        self._is_trained = False
        self.perplexity_scorer = None

        if use_perplexity:
            self.perplexity_scorer = PerplexityScorer()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _extract_features(self, text: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract all features for a single text.

        Returns (feature_array, feature_dict)
        """
        # Stylometric features (12)
        stylo_features = extract_stylometric_features(text)
        feature_array = stylometric_features_to_array(stylo_features)

        # Perplexity score (1)
        if self.use_perplexity and self.perplexity_scorer is not None:
            ppl = self.perplexity_scorer.compute_perplexity(text)
            feature_array.append(ppl)
            stylo_features["perplexity"] = ppl
        else:
            feature_array.append(-1.0)
            stylo_features["perplexity"] = -1.0

        return np.array(feature_array, dtype=np.float64), stylo_features

    def _extract_features_batch(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        """Extract features for a batch of texts."""
        all_features = []
        all_dicts = []

        # Batch stylometric features
        for text in texts:
            stylo = extract_stylometric_features(text)
            all_dicts.append(stylo)
            all_features.append(stylometric_features_to_array(stylo))

        X = np.array(all_features, dtype=np.float64)

        # Batch perplexity
        if self.use_perplexity and self.perplexity_scorer is not None:
            print("  ⏳ Computing perplexity scores...")
            ppl_scores = self.perplexity_scorer.compute_batch(texts)
            X = np.column_stack([X, ppl_scores])
            for i, d in enumerate(all_dicts):
                d["perplexity"] = float(ppl_scores[i])
        else:
            X = np.column_stack([X, np.full(len(texts), -1.0)])
            for d in all_dicts:
                d["perplexity"] = -1.0

        return X, all_dicts

    def train(
        self,
        texts: List[str],
        labels: List[int],
        n_estimators: int = 200,
    ) -> dict:
        """
        Train the AI-generated detector.

        Parameters
        ----------
        texts : list of str
            Email body texts.
        labels : list of int
            0 = human-written, 1 = AI-generated.
        n_estimators : int
            Number of trees in the Random Forest.

        Returns
        -------
        dict
            Training metrics.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
        )

        print("\n══════════════════════════════════════════════")
        print("  🤖 Training: AI-Generated Detector")
        print("══════════════════════════════════════════════")

        X, _ = self._extract_features_batch(texts)

        # Handle invalid perplexity values
        X = np.nan_to_num(X, nan=-1.0, posinf=-1.0, neginf=-1.0)

        y = np.array(labels)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self._is_trained = True

        y_pred = self.model.predict(X)
        metrics = {
            "model": "AI-Generated Detector",
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        }

        self.save()
        print(f"  ✅ AI-Generated Detector trained — F1: {metrics['f1']:.4f}")
        return metrics

    def predict(self, text: str) -> Dict:
        """
        Predict whether an email is AI-generated.

        Parameters
        ----------
        text : str
            Email body text.

        Returns
        -------
        dict
            {
                "is_ai_generated": bool,
                "confidence": float,
                "perplexity": float,
                "stylometric_features": dict,
            }
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        feature_array, feature_dict = self._extract_features(text)
        feature_array = np.nan_to_num(
            feature_array.reshape(1, -1), nan=-1.0, posinf=-1.0, neginf=-1.0,
        )

        prob = self.model.predict_proba(feature_array)[0]
        is_ai = bool(prob[1] > 0.5)
        confidence = float(prob[1]) if is_ai else float(prob[0])

        return {
            "is_ai_generated": is_ai,
            "ai_probability": float(prob[1]),
            "confidence": round(confidence, 4),
            "perplexity": feature_dict.get("perplexity", -1.0),
            "stylometric_features": feature_dict,
        }

    def predict_score(self, text: str) -> float:
        """
        Return just the AI-generated probability score (0–1).

        Useful for feeding into the stacking ensemble.
        """
        if not self._is_trained:
            return 0.5  # neutral fallback

        try:
            result = self.predict(text)
            return result["ai_probability"]
        except Exception:
            return 0.5

    def predict_batch_scores(self, texts: List[str]) -> np.ndarray:
        """Return AI-generated probability scores for a batch."""
        if not self._is_trained:
            return np.full(len(texts), 0.5)

        X, _ = self._extract_features_batch(texts)
        X = np.nan_to_num(X, nan=-1.0, posinf=-1.0, neginf=-1.0)
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str = None):
        path = path or self.model_path
        self.model = joblib.load(path)
        self._is_trained = True
        print(f"  ✅ Loaded AI-Generated Detector from {path}")
