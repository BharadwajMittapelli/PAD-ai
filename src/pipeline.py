"""
src/pipeline.py
===============
PAD-ai Phase 2 — Unified Inference Pipeline

PhishGuardPredictor wraps all sub-models into a single API:
    - predict_email(text) → text-only analysis
    - predict_url(url)    → URL-only analysis
    - predict_combined(email, url) → full hybrid pipeline
    - explain_prediction() → SHAP/LIME explanations

Provides graceful fallback: if hybrid models aren't available,
falls back to the Phase 1 RandomForest URL detector.

Usage:
    from src.pipeline import PhishGuardPredictor

    predictor = PhishGuardPredictor()
    result = predictor.predict_combined("Verify your account...", "http://evil.com/login")
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import extract_url_features, features_to_array
from src.preprocessing import (
    extract_url_features_extended,
    extended_features_to_array,
    clean_text,
)


class PhishGuardPredictor:
    """
    End-to-end inference pipeline for PAD-ai Phase 2.

    Combines:
    - DistilBERT text classifier
    - XGBoost URL classifier
    - AI-Generated phishing detector
    - Stacking ensemble meta-learner
    - SHAP/LIME explainability

    Falls back to Phase 1 RandomForest if hybrid models unavailable.
    """

    def __init__(
        self,
        model_dir: str = "models/hybrid",
        ai_detector_path: str = "models/ai_detector/ai_detector.joblib",
        use_perplexity: bool = False,  # disabled by default for speed
    ):
        self.model_dir = model_dir
        self.ai_detector_path = ai_detector_path

        # Sub-models (lazily loaded)
        self._bert_model = None
        self._xgb_model = None
        self._ai_detector = None
        self._ensemble = None
        self._phase1_detector = None
        self._explainer = None

        self.use_perplexity = use_perplexity
        self._hybrid_available = False

        # Try loading hybrid models
        self._try_load_hybrid()

    def _try_load_hybrid(self):
        """Attempt to load all hybrid model components."""
        try:
            # DistilBERT
            bert_dir = os.path.join(self.model_dir, "distilbert")
            if os.path.exists(bert_dir):
                from src.ml.hybrid_model import DistilBERTPhishingClassifier
                self._bert_model = DistilBERTPhishingClassifier(model_dir=bert_dir)
                self._bert_model.load(bert_dir)

            # XGBoost
            xgb_path = os.path.join(self.model_dir, "xgb_url.joblib")
            if os.path.exists(xgb_path):
                from src.ml.hybrid_model import URLXGBoostClassifier
                self._xgb_model = URLXGBoostClassifier(model_path=xgb_path)
                self._xgb_model.load(xgb_path)

            # AI-Generated detector
            if os.path.exists(self.ai_detector_path):
                from src.ml.ai_generated_detector import AIGeneratedDetector
                self._ai_detector = AIGeneratedDetector(
                    model_path=self.ai_detector_path,
                    use_perplexity=self.use_perplexity,
                )
                self._ai_detector.load(self.ai_detector_path)

            # Stacking ensemble
            meta_path = os.path.join(self.model_dir, "meta_learner.joblib")
            if os.path.exists(meta_path):
                from src.ml.hybrid_model import HybridStackingEnsemble
                self._ensemble = HybridStackingEnsemble(model_path=meta_path)
                self._ensemble.load(meta_path)

            # Check if we have at least the core models
            self._hybrid_available = (
                self._bert_model is not None
                and self._xgb_model is not None
            )

            if self._hybrid_available:
                print("  [OK] Hybrid pipeline loaded successfully")
            else:
                print("  [INFO] Hybrid models not found -- using Phase 1 fallback")

        except Exception as e:
            print(f"  [WARN] Error loading hybrid models: {type(e).__name__}")
            self._hybrid_available = False

    def _get_phase1_detector(self):
        """Load Phase 1 RandomForest detector as fallback."""
        if self._phase1_detector is None:
            try:
                from src.ml.model import PhishingDetector
                self._phase1_detector = PhishingDetector()
                if not self._phase1_detector.is_trained:
                    self._phase1_detector._train_seed()
            except Exception as e:
                print(f"  [WARN] Phase 1 detector unavailable: {type(e).__name__}")
        return self._phase1_detector

    # ── Public API ───────────────────────────────────────────────────────

    def predict_email(self, text: str) -> Dict:
        """
        Predict phishing using email text only.

        Parameters
        ----------
        text : str
            Email body text.

        Returns
        -------
        dict
            {
                label: str,
                is_phishing: bool,
                confidence: float,
                text_score: float,
                ai_generated_score: float,
                ai_generated: bool,
                model_used: str,
            }
        """
        cleaned = clean_text(text)
        result = {
            "label": "safe",
            "is_phishing": False,
            "confidence": 0.0,
            "text_score": 0.5,
            "ai_generated_score": 0.0,
            "ai_generated": False,
            "model_used": "none",
        }

        # DistilBERT prediction
        if self._bert_model is not None and self._bert_model.is_trained:
            prob, label = self._bert_model.predict_single("", text)
            result["text_score"] = prob
            result["is_phishing"] = label == 1
            result["confidence"] = prob if label == 1 else 1.0 - prob
            result["label"] = "phishing" if label == 1 else "safe"
            result["model_used"] = "distilbert"
        else:
            # Simple heuristic fallback
            from src.ml.ai_generated_detector import extract_stylometric_features
            features = extract_stylometric_features(text)
            # Check for urgency patterns
            urgency_words = ["urgent", "immediately", "suspend", "verify", "confirm", "alert"]
            text_lower = text.lower()
            urgency_count = sum(1 for w in urgency_words if w in text_lower)
            score = min(urgency_count * 0.15, 0.9)
            result["text_score"] = score
            result["is_phishing"] = score > 0.5
            result["confidence"] = max(score, 1.0 - score)
            result["label"] = "phishing" if score > 0.5 else "safe"
            result["model_used"] = "heuristic"

        # AI-generated detection
        if self._ai_detector is not None and self._ai_detector.is_trained:
            ai_result = self._ai_detector.predict(text)
            result["ai_generated_score"] = ai_result["ai_probability"]
            result["ai_generated"] = ai_result["is_ai_generated"]

        return result

    def predict_url(self, url: str) -> Dict:
        """
        Predict phishing using URL only.

        Parameters
        ----------
        url : str
            URL string.

        Returns
        -------
        dict
            {
                label, is_phishing, confidence,
                url_score, features, model_used,
            }
        """
        features = extract_url_features_extended(url)
        feature_array = np.array(extended_features_to_array(features), dtype=np.float32)

        result = {
            "label": "safe",
            "is_phishing": False,
            "confidence": 0.0,
            "url_score": 0.5,
            "features": features,
            "model_used": "none",
        }

        # XGBoost prediction
        if self._xgb_model is not None and self._xgb_model.is_trained:
            prob, label = self._xgb_model.predict_single(feature_array)
            result["url_score"] = prob
            result["is_phishing"] = label == 1
            result["confidence"] = prob if label == 1 else 1.0 - prob
            result["label"] = "phishing" if label == 1 else "safe"
            result["model_used"] = "xgboost"
        else:
            # Phase 1 fallback
            detector = self._get_phase1_detector()
            if detector is not None:
                is_phish, conf, _ = detector.predict(url)
                result["url_score"] = conf if is_phish else 1.0 - conf
                result["is_phishing"] = is_phish
                result["confidence"] = conf
                result["label"] = "phishing" if is_phish else "safe"
                result["model_used"] = "random_forest_v1"

        return result

    def predict_combined(self, email: str, url: str) -> Dict:
        """
        Full hybrid pipeline prediction combining all models.

        Parameters
        ----------
        email : str
            Email body text.
        url : str
            URL string.

        Returns
        -------
        dict
            {
                label, is_phishing, confidence,
                text_score, url_score, ai_score,
                features, model_used,
            }
        """
        # Extract URL features
        features = extract_url_features_extended(url)
        feature_array = np.array(extended_features_to_array(features), dtype=np.float32)
        cleaned_email = clean_text(email)

        result = {
            "label": "safe",
            "is_phishing": False,
            "confidence": 0.0,
            "text_score": 0.5,
            "url_score": 0.5,
            "ai_score": 0.0,
            "ai_generated": False,
            "features": features,
            "model_used": "none",
        }

        # ── Sub-model predictions ─────────────────────────────────────────

        # Text model
        bert_prob = 0.5
        if self._bert_model is not None and self._bert_model.is_trained:
            bert_prob, _ = self._bert_model.predict_single(url, email)
            result["text_score"] = bert_prob

        # URL model
        xgb_prob = 0.5
        if self._xgb_model is not None and self._xgb_model.is_trained:
            xgb_prob, _ = self._xgb_model.predict_single(feature_array)
            result["url_score"] = xgb_prob

        # AI-Generated detector
        ai_score = 0.0
        if self._ai_detector is not None and self._ai_detector.is_trained:
            ai_result = self._ai_detector.predict(email)
            ai_score = ai_result["ai_probability"]
            result["ai_score"] = ai_score
            result["ai_generated"] = ai_result["is_ai_generated"]

        # ── Ensemble prediction ───────────────────────────────────────────

        if self._ensemble is not None and self._ensemble.is_trained:
            url_df = pd.DataFrame([features])
            is_phish, confidence = self._ensemble.predict_single(
                bert_prob, xgb_prob, ai_score, features,
            )
            result["is_phishing"] = is_phish
            result["confidence"] = confidence
            result["label"] = "phishing" if is_phish else "safe"
            result["model_used"] = "hybrid_ensemble"
        elif self._hybrid_available:
            # Simple weighted average if no meta-learner
            combined = 0.5 * bert_prob + 0.4 * xgb_prob + 0.1 * ai_score
            result["is_phishing"] = combined > 0.5
            result["confidence"] = combined if combined > 0.5 else 1.0 - combined
            result["label"] = "phishing" if combined > 0.5 else "safe"
            result["model_used"] = "weighted_average"
        else:
            # Phase 1 fallback
            url_result = self.predict_url(url)
            result.update({
                "is_phishing": url_result["is_phishing"],
                "confidence": url_result["confidence"],
                "label": url_result["label"],
                "url_score": url_result["url_score"],
                "model_used": url_result["model_used"] + " (fallback)",
            })

        return result

    # ── Explainability ───────────────────────────────────────────────────

    def explain_prediction(
        self,
        email: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Dict:
        """
        Get an explanation for a prediction.

        Parameters
        ----------
        email : str, optional
            Email body text.
        url : str, optional
            URL string.

        Returns
        -------
        dict
            Explanation including top contributing features and tokens.
        """
        from src.ml.explainability import PredictionExplainer

        # Get prediction first
        if email and url:
            prediction = self.predict_combined(email, url)
        elif url:
            prediction = self.predict_url(url)
        elif email:
            prediction = self.predict_email(email)
        else:
            return {"error": "Provide at least email or url"}

        # Build explainer
        tree_model = self._xgb_model.model if (
            self._xgb_model and self._xgb_model.is_trained
        ) else None

        from src.preprocessing import get_extended_feature_names
        feature_names = get_extended_feature_names()

        explainer = PredictionExplainer(
            tree_model=tree_model,
            feature_names=feature_names,
        )

        # URL features for SHAP
        url_features = None
        if url:
            features = extract_url_features_extended(url)
            url_features = np.array(extended_features_to_array(features), dtype=np.float32)

        # Text prediction function for LIME
        text_predict_fn = None
        if email and self._bert_model and self._bert_model.is_trained:
            def text_predict_fn(texts):
                probs = self._bert_model.predict_proba(texts)
                return np.column_stack([1 - probs, probs])

        return explainer.explain_prediction(
            prediction_result=prediction,
            url_features=url_features,
            email_text=email,
            text_predict_fn=text_predict_fn,
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_hybrid_available(self) -> bool:
        """Whether the full hybrid pipeline is loaded."""
        return self._hybrid_available

    @property
    def available_models(self) -> Dict[str, bool]:
        """Status of each sub-model."""
        return {
            "distilbert": self._bert_model is not None and self._bert_model.is_trained,
            "xgboost_url": self._xgb_model is not None and self._xgb_model.is_trained,
            "ai_detector": self._ai_detector is not None and self._ai_detector.is_trained,
            "ensemble": self._ensemble is not None and self._ensemble.is_trained,
            "phase1_rf": self._get_phase1_detector() is not None,
        }

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "PhishGuardPredictor":
        """Load a complete pipeline from a directory."""
        return cls(model_dir=model_dir)
