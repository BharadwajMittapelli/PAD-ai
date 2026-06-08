"""
src/ml/explainability.py
========================
PAD-ai Phase 2 — Explainability & Optimisation

Provides:
- SHAP explanations for tree-based models (XGBoost/RF)
- LIME explanations for text (DistilBERT)
- Unified prediction explanation
- Optuna ensemble weight tuning
- Plotly visualisation helpers

Usage:
    from src.ml.explainability import PredictionExplainer

    explainer = PredictionExplainer(model, feature_names)
    result = explainer.explain_prediction(email_text, url, pipeline)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class PredictionExplainer:
    """
    Unified explainability layer combining SHAP and LIME
    for both URL features and text analysis.
    """

    def __init__(
        self,
        tree_model=None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        tree_model : sklearn/xgboost model, optional
            Tree-based model for SHAP TreeExplainer.
        feature_names : list of str, optional
            Feature names for the tree model.
        """
        self.tree_model = tree_model
        self.feature_names = feature_names or []
        self._shap_explainer = None

    # ── SHAP Explanations (Tree-based models) ────────────────────────────

    def _init_shap_explainer(self):
        """Lazily initialise SHAP TreeExplainer."""
        if self._shap_explainer is None and self.tree_model is not None:
            try:
                import shap
                self._shap_explainer = shap.TreeExplainer(self.tree_model)
            except ImportError:
                print("  ⚠️ shap not installed. Install with: pip install shap")

    def explain_shap(
        self,
        X_instance: np.ndarray,
        top_k: int = 10,
    ) -> Dict:
        """
        Compute SHAP values for a single instance.

        Parameters
        ----------
        X_instance : np.ndarray
            Single feature vector (1D or 2D with 1 row).
        top_k : int
            Number of top contributing features to return.

        Returns
        -------
        dict
            {
                "shap_values": list,
                "top_features": [{name, value, shap_value, direction}],
                "base_value": float,
            }
        """
        try:
            import shap
        except ImportError:
            return {"shap_values": [], "top_features": [], "base_value": 0.0}

        self._init_shap_explainer()
        if self._shap_explainer is None:
            return {"shap_values": [], "top_features": [], "base_value": 0.0}

        X = X_instance.reshape(1, -1) if X_instance.ndim == 1 else X_instance

        shap_values = self._shap_explainer.shap_values(X)

        # Handle multi-output (binary classification returns list of 2)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # positive class (phishing)
        else:
            sv = shap_values[0]

        base_value = self._shap_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

        # Build top features
        n_features = len(sv)
        names = self.feature_names[:n_features] if self.feature_names else [
            f"feature_{i}" for i in range(n_features)
        ]
        values = X[0][:n_features]

        feature_importance = []
        for i in range(n_features):
            feature_importance.append({
                "name": names[i],
                "value": round(float(values[i]), 4),
                "shap_value": round(float(sv[i]), 4),
                "direction": "increases risk" if sv[i] > 0 else "decreases risk",
            })

        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_features = feature_importance[:top_k]

        return {
            "shap_values": [round(float(v), 4) for v in sv],
            "top_features": top_features,
            "base_value": round(float(base_value), 4),
        }

    # ── LIME Explanations (Text models) ──────────────────────────────────

    def explain_lime(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> Dict:
        """
        Generate LIME text explanation for a prediction.

        Parameters
        ----------
        text : str
            Input text to explain.
        predict_fn : callable
            Function that takes a list of texts and returns
            shape (n, 2) probability array.
        num_features : int
            Number of top features to return.
        num_samples : int
            Number of perturbed samples for LIME.

        Returns
        -------
        dict
            {
                "top_tokens": [{token, weight, direction}],
                "prediction_local": float,
            }
        """
        try:
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=["Safe", "Phishing"])
            explanation = explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

            top_tokens = []
            for token, weight in explanation.as_list():
                top_tokens.append({
                    "token": token,
                    "weight": round(float(weight), 4),
                    "direction": "increases risk" if weight > 0 else "decreases risk",
                })

            return {
                "top_tokens": top_tokens,
                "prediction_local": round(
                    float(explanation.local_pred[0]) if hasattr(explanation, 'local_pred') else 0.0,
                    4,
                ),
            }

        except Exception as e:
            print(f"  ⚠️ LIME explanation failed: {e}")
            return {"top_tokens": [], "prediction_local": 0.0}

    # ── Unified Explanation ──────────────────────────────────────────────

    def explain_prediction(
        self,
        prediction_result: Dict,
        url_features: Optional[np.ndarray] = None,
        email_text: Optional[str] = None,
        text_predict_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Generate a unified explanation combining SHAP and LIME.

        Parameters
        ----------
        prediction_result : dict
            Output from PhishGuardPredictor.predict_combined().
        url_features : np.ndarray, optional
            URL feature vector for SHAP explanation.
        email_text : str, optional
            Email text for LIME explanation.
        text_predict_fn : callable, optional
            Text model predict function for LIME.

        Returns
        -------
        dict
            Combined explanation with top URL features,
            top text tokens, and a human-readable summary.
        """
        explanation = {
            "prediction": prediction_result.get("label", "unknown"),
            "confidence": prediction_result.get("confidence", 0.0),
            "url_explanation": {},
            "text_explanation": {},
            "ai_generated_score": prediction_result.get("ai_score", 0.0),
            "explanation_text": "",
        }

        # SHAP for URL features
        if url_features is not None and self.tree_model is not None:
            explanation["url_explanation"] = self.explain_shap(url_features)

        # LIME for text
        if email_text and text_predict_fn:
            explanation["text_explanation"] = self.explain_lime(
                email_text, text_predict_fn,
            )

        # Generate human-readable explanation
        explanation["explanation_text"] = self._generate_explanation_text(explanation)

        return explanation

    def _generate_explanation_text(self, explanation: Dict) -> str:
        """Generate a human-readable explanation string."""
        parts = []
        pred = explanation.get("prediction", "unknown")
        conf = explanation.get("confidence", 0)

        if pred == "phishing":
            parts.append(
                f"⚠️ This input was classified as PHISHING with {conf*100:.1f}% confidence."
            )
        else:
            parts.append(
                f"✅ This input appears SAFE with {conf*100:.1f}% confidence."
            )

        # URL feature highlights
        url_exp = explanation.get("url_explanation", {})
        top_url = url_exp.get("top_features", [])
        if top_url:
            risk_features = [f for f in top_url[:3] if f["direction"] == "increases risk"]
            if risk_features:
                names = ", ".join(f["name"] for f in risk_features)
                parts.append(f"Key URL risk factors: {names}.")

        # Text token highlights
        text_exp = explanation.get("text_explanation", {})
        top_tokens = text_exp.get("top_tokens", [])
        if top_tokens:
            risk_tokens = [t for t in top_tokens[:3] if t["direction"] == "increases risk"]
            if risk_tokens:
                tokens = ", ".join(f'"{t["token"]}"' for t in risk_tokens)
                parts.append(f"Suspicious text patterns: {tokens}.")

        # AI-generated flag
        ai_score = explanation.get("ai_generated_score", 0)
        if ai_score > 0.7:
            parts.append("🤖 This email appears to be AI-generated.")
        elif ai_score > 0.4:
            parts.append("⚠️ Some indicators of AI-generated content detected.")

        return " ".join(parts)

    # ── Plotly Visualisations ────────────────────────────────────────────

    def plot_shap_waterfall(
        self,
        shap_result: Dict,
        title: str = "Feature Contributions (SHAP)",
    ):
        """
        Create a SHAP waterfall-style bar chart using Plotly.

        Parameters
        ----------
        shap_result : dict
            Output from explain_shap().
        title : str
            Chart title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        top_features = shap_result.get("top_features", [])
        if not top_features:
            return go.Figure()

        # Reverse for bottom-to-top display
        features = list(reversed(top_features[:10]))
        names = [f["name"] for f in features]
        values = [f["shap_value"] for f in features]
        colours = ["#ef4444" if v > 0 else "#10b981" for v in values]

        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.4f}" for v in values],
            textposition="auto",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="SHAP Value (impact on phishing prediction)",
            template="plotly_dark",
            height=max(300, len(features) * 40),
            margin=dict(l=150, r=20, t=50, b=30),
        )

        return fig

    def plot_lime_highlights(self, lime_result: Dict) -> str:
        """
        Create an HTML visualisation of LIME text explanations.

        Parameters
        ----------
        lime_result : dict
            Output from explain_lime().

        Returns
        -------
        str
            HTML string with highlighted tokens.
        """
        top_tokens = lime_result.get("top_tokens", [])
        if not top_tokens:
            return "<p>No text explanation available.</p>"

        html_parts = ['<div style="font-family: monospace; line-height: 1.8;">']
        for token_info in top_tokens:
            token = token_info["token"]
            weight = token_info["weight"]
            if weight > 0:
                opacity = min(abs(weight) * 5, 0.8)
                color = f"rgba(239, 68, 68, {opacity})"
            else:
                opacity = min(abs(weight) * 5, 0.8)
                color = f"rgba(16, 185, 129, {opacity})"

            html_parts.append(
                f'<span style="background: {color}; padding: 2px 4px; '
                f'margin: 1px; border-radius: 3px;">{token} '
                f'({weight:+.3f})</span> '
            )

        html_parts.append("</div>")
        return "".join(html_parts)


# ═════════════════════════════════════════════════════════════════════════════
# Optuna Ensemble Weight Tuning
# ═════════════════════════════════════════════════════════════════════════════

def tune_ensemble_weights(
    bert_probs: np.ndarray,
    xgb_probs: np.ndarray,
    ai_scores: np.ndarray,
    y_true: np.ndarray,
    n_trials: int = 100,
    fpr_threshold: float = 0.05,
) -> Dict:
    """
    Use Optuna to find optimal ensemble weights.

    Objective: maximise F1 while keeping FPR below threshold.

    Parameters
    ----------
    bert_probs : np.ndarray
        DistilBERT predicted probabilities.
    xgb_probs : np.ndarray
        XGBoost predicted probabilities.
    ai_scores : np.ndarray
        AI-generated detector scores.
    y_true : np.ndarray
        Ground truth labels.
    n_trials : int
        Number of Optuna trials.
    fpr_threshold : float
        Maximum acceptable false positive rate.

    Returns
    -------
    dict
        Optimal weights and performance metrics.
    """
    import optuna
    from sklearn.metrics import f1_score, confusion_matrix

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n  🔧 Tuning ensemble weights ({n_trials} trials)...")

    def objective(trial):
        w_bert = trial.suggest_float("w_bert", 0.1, 0.8)
        w_xgb = trial.suggest_float("w_xgb", 0.1, 0.8)
        w_ai = trial.suggest_float("w_ai", 0.0, 0.3)
        threshold = trial.suggest_float("threshold", 0.3, 0.7)

        # Normalise weights
        total = w_bert + w_xgb + w_ai
        w_bert /= total
        w_xgb /= total
        w_ai /= total

        # Weighted ensemble
        combined = w_bert * bert_probs + w_xgb * xgb_probs + w_ai * ai_scores
        y_pred = (combined > threshold).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Penalise high FPR
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / max(fp + tn, 1)
            if fpr > fpr_threshold:
                f1 *= 0.5  # heavy penalty

        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    total = best["w_bert"] + best["w_xgb"] + best["w_ai"]

    result = {
        "w_bert": round(best["w_bert"] / total, 4),
        "w_xgb": round(best["w_xgb"] / total, 4),
        "w_ai": round(best["w_ai"] / total, 4),
        "threshold": round(best["threshold"], 4),
        "best_f1": round(study.best_value, 4),
    }

    print(f"  ✅ Optimal weights: BERT={result['w_bert']}, "
          f"XGB={result['w_xgb']}, AI={result['w_ai']}")
    print(f"  ✅ Threshold: {result['threshold']}, Best F1: {result['best_f1']}")

    return result
