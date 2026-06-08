"""
tests/test_phase2.py
====================
PAD-ai Phase 2 — Unit Tests

Covers:
- Preprocessing pipeline (text cleaning, extended URL features, TF-IDF)
- Stylometric feature extraction
- AI-generated detector
- Pipeline integration
- Evaluator
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ═════════════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_clean_text_basic(self):
        from src.preprocessing import clean_text

        assert clean_text("Hello   World") == "hello world"
        assert clean_text("<p>HTML</p>") == "html"
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_text_removes_urls(self):
        from src.preprocessing import clean_text

        result = clean_text(
            "Visit https://evil.com for details", remove_urls=True
        )
        assert "evil.com" not in result
        assert "visit" in result

    def test_extract_url_features_extended_count(self):
        """Extended features should return 40 features (25 + 15)."""
        from src.preprocessing import extract_url_features_extended

        features = extract_url_features_extended("https://www.google.com/search?q=test")
        assert len(features) == 40, f"Expected 40 features, got {len(features)}"

    def test_extract_url_features_extended_includes_base(self):
        """Extended features should include all original 25 base features."""
        from src.preprocessing import extract_url_features_extended
        from src.utils import get_feature_keys

        features = extract_url_features_extended("https://example.com")
        base_keys = get_feature_keys()
        for key in base_keys:
            assert key in features, f"Missing base feature: {key}"

    def test_extract_url_features_extended_new_features(self):
        """Verify the 15 new features are present."""
        from src.preprocessing import extract_url_features_extended, EXTENDED_FEATURE_ORDER

        features = extract_url_features_extended("http://evil.com/path/to/login.php")
        for key in EXTENDED_FEATURE_ORDER:
            assert key in features, f"Missing extended feature: {key}"

    def test_extended_features_path_depth(self):
        from src.preprocessing import extract_url_features_extended

        features = extract_url_features_extended("http://evil.com/a/b/c/login")
        assert features["path_depth"] == 4

    def test_extended_features_shortened_url(self):
        from src.preprocessing import extract_url_features_extended

        features = extract_url_features_extended("https://bit.ly/abc123")
        assert features["is_shortened_url"] == 1

    def test_extended_features_port_number(self):
        from src.preprocessing import extract_url_features_extended

        features = extract_url_features_extended("http://evil.com:8080/login")
        assert features["has_port_number"] == 1

    def test_extended_features_punycode(self):
        from src.preprocessing import extract_url_features_extended

        features = extract_url_features_extended("http://xn--pypal-4ve.com/login")
        assert features["has_punycode"] == 1

    def test_extended_features_to_array_length(self):
        from src.preprocessing import extract_url_features_extended, extended_features_to_array

        features = extract_url_features_extended("https://google.com")
        array = extended_features_to_array(features)
        assert len(array) == 40

    def test_create_tfidf_features(self):
        from src.preprocessing import create_tfidf_features

        texts = ["hello world", "phishing attempt", "safe email content"]
        X, vectorizer = create_tfidf_features(texts, max_features=100)
        assert X.shape[0] == 3
        assert X.shape[1] <= 100

        # Transform with existing vectorizer
        X2, _ = create_tfidf_features(["new text"], vectorizer=vectorizer)
        assert X2.shape[0] == 1
        assert X2.shape[1] == X.shape[1]


# ═════════════════════════════════════════════════════════════════════════════
# Stylometric Feature Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestStylometricFeatures:

    def test_stylometric_features_count(self):
        """Should return exactly 12 stylometric features."""
        from src.ml.ai_generated_detector import (
            extract_stylometric_features,
            STYLOMETRIC_FEATURE_NAMES,
        )

        features = extract_stylometric_features(
            "This is a test sentence. Here is another one."
        )
        assert len(features) == 12
        for name in STYLOMETRIC_FEATURE_NAMES:
            assert name in features, f"Missing: {name}"

    def test_stylometric_features_empty_text(self):
        from src.ml.ai_generated_detector import extract_stylometric_features

        features = extract_stylometric_features("")
        assert all(v == 0.0 for v in features.values())

    def test_stylometric_features_values_reasonable(self):
        from src.ml.ai_generated_detector import extract_stylometric_features

        text = (
            "Dear customer, your account has been suspended. "
            "Please verify your identity immediately. "
            "Click the link below to restore access."
        )
        features = extract_stylometric_features(text)

        assert features["avg_sentence_length"] > 0
        assert 0 <= features["unique_word_ratio"] <= 1
        assert 0 <= features["punctuation_ratio"] <= 1
        assert 0 <= features["stopword_ratio"] <= 1

    def test_stylometric_to_array(self):
        from src.ml.ai_generated_detector import (
            extract_stylometric_features,
            stylometric_features_to_array,
        )

        features = extract_stylometric_features("Test sentence here.")
        array = stylometric_features_to_array(features)
        assert len(array) == 12
        assert all(isinstance(v, float) for v in array)


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPipeline:

    def test_pipeline_init(self):
        """Pipeline should initialise without error even without models."""
        from src.pipeline import PhishGuardPredictor

        predictor = PhishGuardPredictor(model_dir="models/nonexistent")
        assert predictor is not None

    def test_pipeline_predict_url_fallback(self):
        """URL prediction should work via Phase 1 fallback."""
        from src.pipeline import PhishGuardPredictor

        predictor = PhishGuardPredictor()
        result = predictor.predict_url("https://www.google.com")

        assert "label" in result
        assert "is_phishing" in result
        assert "confidence" in result
        assert "url_score" in result
        assert isinstance(result["is_phishing"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_pipeline_predict_url_phishing(self):
        """Known phishing URL should be flagged."""
        from src.pipeline import PhishGuardPredictor

        predictor = PhishGuardPredictor()
        result = predictor.predict_url("http://192.168.1.1/login/verify-account")

        assert result["is_phishing"] is True

    def test_pipeline_available_models(self):
        """available_models should return a dict of booleans."""
        from src.pipeline import PhishGuardPredictor

        predictor = PhishGuardPredictor()
        status = predictor.available_models
        assert isinstance(status, dict)
        assert "phase1_rf" in status

    def test_pipeline_predict_email(self):
        """Email prediction should return valid result structure."""
        from src.pipeline import PhishGuardPredictor

        predictor = PhishGuardPredictor()
        result = predictor.predict_email(
            "URGENT: Your account has been suspended. Verify immediately!"
        )
        assert "label" in result
        assert "text_score" in result
        assert "ai_generated_score" in result


# ═════════════════════════════════════════════════════════════════════════════
# Evaluator Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluator:

    def test_compute_metrics(self):
        from src.ml.evaluator import ComprehensiveEvaluator

        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.3, 0.7, 0.6])

        metrics = ComprehensiveEvaluator.compute_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert "fpr" in metrics
        assert "mcc" in metrics
        assert "confusion_matrix" in metrics

    def test_false_positive_rate(self):
        from src.ml.evaluator import ComprehensiveEvaluator

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])  # 1 FP

        fpr = ComprehensiveEvaluator.compute_false_positive_rate(y_true, y_pred)
        assert fpr == 0.25  # 1 FP out of 4 negatives

    def test_ablation_study(self):
        from src.ml.evaluator import ComprehensiveEvaluator

        evaluator = ComprehensiveEvaluator(output_dir="models/test_eval")
        results = [
            {"model": "Model A", "accuracy": 0.9, "precision": 0.85,
             "recall": 0.88, "f1": 0.86, "auc": 0.92, "fpr": 0.03, "mcc": 0.8},
            {"model": "Model B", "accuracy": 0.85, "precision": 0.80,
             "recall": 0.90, "f1": 0.85, "auc": 0.88, "fpr": 0.05, "mcc": 0.7},
        ]
        df = evaluator.ablation_study(results)

        assert len(df) == 2
        assert df.iloc[0]["f1"] >= df.iloc[1]["f1"]  # sorted by F1

        # Cleanup
        import shutil
        if os.path.exists("models/test_eval"):
            shutil.rmtree("models/test_eval")


# ═════════════════════════════════════════════════════════════════════════════
# Explainability Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestExplainability:

    def test_shap_without_model(self):
        """SHAP should return empty result without a model."""
        pytest.importorskip("shap")
        from src.ml.explainability import PredictionExplainer

        explainer = PredictionExplainer(tree_model=None)
        result = explainer.explain_shap(np.zeros(10))
        assert result["shap_values"] == []
        assert result["top_features"] == []

    def test_lime_highlights_html(self):
        from src.ml.explainability import PredictionExplainer

        explainer = PredictionExplainer()
        lime_result = {
            "top_tokens": [
                {"token": "verify", "weight": 0.3, "direction": "increases risk"},
                {"token": "google", "weight": -0.2, "direction": "decreases risk"},
            ]
        }
        html = explainer.plot_lime_highlights(lime_result)
        assert "verify" in html
        assert "google" in html
        assert "<span" in html
