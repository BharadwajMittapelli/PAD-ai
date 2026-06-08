"""
app/streamlit_app.py
====================
PAD-ai Phase 2 — Phishing Attack Detection Dashboard

Multi-modal detection: URL analysis + Email text + AI-generated detection
with explainability (SHAP/LIME) and model comparison.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

# ── Path setup so 'src' is importable ───────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.utils import extract_url_features, get_feature_names, get_feature_keys
from src.preprocessing import (
    extract_url_features_extended,
    extended_features_to_array,
    get_extended_feature_names,
    get_extended_feature_keys,
    clean_text,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAD-ai v2 — Hybrid Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit header chrome */
    #MainMenu, footer { visibility: hidden; }

    .pad-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .pad-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-danger {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-danger h2 { color: #ef4444; }
    .result-safe  h2  { color: #10b981; }
    .ai-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .ai-badge-positive {
        background: rgba(168, 85, 247, 0.2);
        border: 1px solid #a855f7;
        color: #a855f7;
    }
    .ai-badge-negative {
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid #64748b;
        color: #94a3b8;
    }
    .score-card {
        background: rgba(30, 30, 60, 0.5);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid rgba(100, 100, 150, 0.3);
    }
    .score-card h4 { margin: 0; color: #94a3b8; font-size: 0.8rem; }
    .score-card .score-value { font-size: 1.6rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached across reruns) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection pipeline...")
def load_pipeline():
    """Load PhishGuardPredictor -- falls back to rule-based on failure."""
    import io
    try:
        # Suppress print output during loading to avoid charmap encoding
        # errors on Windows with non-ASCII paths (e.g. Vietnamese "Tai lieu")
        old_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
        try:
            from src.pipeline import PhishGuardPredictor
            pipeline = PhishGuardPredictor()
        finally:
            sys.stdout = old_stdout
        return pipeline
    except Exception:
        st.warning("Pipeline unavailable. Using rule-based fallback.")
        return None


@st.cache_resource(show_spinner="Loading Phase 1 detector...")
def load_detector_v1():
    """Load Phase 1 PhishingDetector as fallback."""
    try:
        from src.ml.model import PhishingDetector
        detector = PhishingDetector()
        if not detector.is_trained:
            detector._train_seed()
        return detector
    except Exception:
        return None


def rule_based_predict(features: dict):
    """
    Simple rule-based fallback when no ML model is available.
    Returns (is_phishing: bool, confidence: float).
    """
    score = (
        features["has_ip_address"]          * 0.35 +
        features["has_at_symbol"]            * 0.20 +
        min(features["suspicious_keyword_count"] * 0.08, 0.25) +
        features["has_double_slash"]         * 0.10 +
        (0.0 if features["is_https"] else 0.15) +
        min(features["subdomain_count"] * 0.05, 0.15)
    )
    is_phish = score > 0.30
    confidence = min(score + 0.35, 0.97) if is_phish else max(1.0 - score, 0.55)
    return is_phish, round(confidence, 4)


# ── Chart helpers ────────────────────────────────────────────────────────────
def make_gauge(confidence: float, is_phishing: bool) -> go.Figure:
    colour = "#ef4444" if is_phishing else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": "Threat Confidence", "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar":  {"color": colour},
            "steps": [
                {"range": [0,  35], "color": "rgba(16,185,129,0.15)"},
                {"range": [35, 65], "color": "rgba(234,179,8,0.15)"},
                {"range": [65, 100],"color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": colour, "width": 4},
                "thickness": 0.8,
                "value": confidence * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        height=260,
        margin=dict(t=60, b=10, l=20, r=20),
    )
    return fig


def make_feature_bar(features: dict) -> go.Figure:
    """Bar chart showing the most interpretable continuous features."""
    display = {
        "URL Length":            features.get("url_length", 0),
        "Domain Length":         features.get("domain_length", 0),
        "Dot Count":             features.get("num_dots", 0),
        "Hyphen Count":          features.get("num_hyphens", 0),
        "Subdomain Count":       features.get("subdomain_count", 0),
        "Path Length":           features.get("path_length", 0),
        "Path Depth":            features.get("path_depth", 0),
        "Suspicious Keywords":   features.get("suspicious_keyword_count", 0),
        "Digit Ratio ×10":       round(features.get("digit_ratio", 0) * 10, 2),
        "URL Entropy ×10":       round(features.get("url_entropy", 0) * 10, 2),
        "Special Char Ratio ×10": round(features.get("special_char_ratio", 0) * 10, 2),
        "Query Params":          features.get("num_query_params", 0),
    }
    labels = list(display.keys())
    values = list(display.values())

    fig = px.bar(
        x=labels, y=values,
        color=values,
        color_continuous_scale=["#10b981", "#eab308", "#ef4444"],
        labels={"x": "", "y": "Value", "color": "Value"},
        title="Feature Breakdown (40 URL Features)",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,40,0.5)",
        font_color="#ffffff",
        title_font_color="#94a3b8",
        height=320,
        margin=dict(t=50, b=10, l=10, r=10),
        xaxis_tickangle=-35,
        coloraxis_showscale=False,
    )
    return fig


def make_score_comparison(text_score: float, url_score: float, ai_score: float) -> go.Figure:
    """Radar/bar chart comparing sub-model scores."""
    categories = ["Text Model\n(DistilBERT)", "URL Model\n(XGBoost)", "AI-Generated\nDetector"]
    scores = [text_score * 100, url_score * 100, ai_score * 100]
    colours = ["#6366f1", "#f59e0b", "#a855f7"]

    fig = go.Figure(go.Bar(
        x=categories, y=scores,
        marker_color=colours,
        text=[f"{s:.1f}%" for s in scores],
        textposition="auto",
        textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        title="Sub-Model Phishing Scores",
        yaxis_title="Phishing Probability (%)",
        yaxis_range=[0, 100],
        template="plotly_dark",
        height=300,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,40,0.5)",
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ PAD-ai v2")
        st.markdown("*Hybrid Phishing Detection System*")
        st.divider()

        st.markdown("### Analysis Mode")
        mode = st.selectbox(
            "Select mode",
            ["🔗 URL Only", "📧 Email Only", "🔗+📧 Combined (Hybrid)"],
            index=2,
            label_visibility="collapsed",
        )
        st.divider()

        st.markdown("### How it works")
        st.markdown("""
1. 🔗 Paste URL + 📧 Email text
2. 🔬 40 URL features extracted
3. 🤖 DistilBERT analyses text
4. 🌳 XGBoost analyses URL
5. 🧠 AI-Generated detector runs
6. 🔗 Stacking ensemble fuses all
7. 📊 SHAP/LIME explains result
        """)
        st.divider()

        st.markdown("### Threat Indicators")
        indicators = [
            ("IP address in URL",    "🔴 High Risk"),
            ("No HTTPS",             "🔴 High Risk"),
            ("Suspicious keywords",  "🔴 High Risk"),
            ("AI-generated content", "🟣 Novel Risk"),
            ("High URL entropy",     "🟡 Medium"),
            ("Many subdomains",      "🟡 Medium"),
            ("@ symbol in URL",      "🔴 High Risk"),
        ]
        for label, risk in indicators:
            st.markdown(f"**{label}** — {risk}")

        st.divider()

        # Model status
        pipeline = load_pipeline()
        if pipeline:
            st.markdown("### Model Status")
            status = pipeline.available_models
            for name, loaded in status.items():
                icon = "✅" if loaded else "⬜"
                st.markdown(f"{icon} {name.replace('_', ' ').title()}")

        st.divider()
        st.caption("PAD-ai v2.0 · Phase 2 Hybrid · Educational use only")

    return mode


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    mode = render_sidebar()

    # Header
    st.markdown('<p class="pad-header">🛡️ PAD-ai v2</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pad-sub">Hybrid Phishing Detection — DistilBERT + XGBoost + AI-Generated Detector</p>',
        unsafe_allow_html=True,
    )

    # ── Input area ────────────────────────────────────────────────────────
    show_url = mode in ["🔗 URL Only", "🔗+📧 Combined (Hybrid)"]
    show_email = mode in ["📧 Email Only", "🔗+📧 Combined (Hybrid)"]

    col_input, col_gap = st.columns([4, 1])

    with col_input:
        url_input = ""
        email_input = ""

        if show_url:
            url_input = st.text_input(
                "URL to analyse",
                placeholder="https://example.com  or  http://suspicious-login-verify.com",
                label_visibility="collapsed",
                key="url_input",
            )

        if show_email:
            email_input = st.text_area(
                "Email body text",
                placeholder="Paste the email body here for text-based analysis...",
                height=120,
                label_visibility="collapsed",
                key="email_input",
            )

        analyse = st.button(
            "⚡ Run Security Audit", type="primary", use_container_width=True
        )

    # ── Session history init ─────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ── Analysis ─────────────────────────────────────────────────────────
    if analyse:
        has_url = url_input and url_input.strip()
        has_email = email_input and email_input.strip()

        if not has_url and not has_email:
            st.warning("Please enter a URL or email text first.")
            st.stop()

        with st.spinner("Analysing with hybrid pipeline..."):
            pipeline = load_pipeline()

            # Route to the appropriate prediction mode
            if has_url and has_email and pipeline:
                result = pipeline.predict_combined(email_input, url_input)
            elif has_url and pipeline:
                result = pipeline.predict_url(url_input)
            elif has_email and pipeline:
                result = pipeline.predict_email(email_input)
            elif has_url:
                # Full fallback
                features = extract_url_features(url_input)
                detector = load_detector_v1()
                if detector:
                    is_phish, conf, _ = detector.predict(url_input)
                else:
                    is_phish, conf = rule_based_predict(features)
                result = {
                    "label": "phishing" if is_phish else "safe",
                    "is_phishing": is_phish,
                    "confidence": conf,
                    "text_score": 0.0,
                    "url_score": conf if is_phish else 1.0 - conf,
                    "ai_score": 0.0,
                    "ai_generated": False,
                    "features": features,
                    "model_used": "phase1_fallback",
                }
            else:
                # Email-only heuristic fallback
                text_lower = email_input.lower()
                urgency_words = [
                    "urgent", "immediately", "suspend", "verify",
                    "confirm", "alert", "unauthorized", "expire",
                    "restricted", "security", "update your",
                    "click here", "act now", "limited time",
                ]
                urgency_count = sum(1 for w in urgency_words if w in text_lower)
                score = min(urgency_count * 0.12, 0.95)
                is_phish = score > 0.35
                conf = max(score, 1.0 - score)

                result = {
                    "label": "phishing" if is_phish else "safe",
                    "is_phishing": is_phish,
                    "confidence": conf,
                    "text_score": score,
                    "url_score": 0.0,
                    "ai_score": 0.0,
                    "ai_generated": False,
                    "features": {},
                    "model_used": "keyword_heuristic",
                }

        # Extract values
        is_phish = result.get("is_phishing", False)
        confidence = result.get("confidence", 0.0)
        text_score = result.get("text_score", 0.0)
        url_score = result.get("url_score", 0.0)
        ai_score = result.get("ai_score", 0.0)
        ai_generated = result.get("ai_generated", False)
        features = result.get("features", {})
        model_used = result.get("model_used", "unknown")

        # If features is empty but we have URL, extract them
        if not features and has_url:
            features = extract_url_features_extended(url_input)

        # ── Result card ──────────────────────────────────────────────────
        st.markdown("---")
        left, right = st.columns([3, 2])

        with left:
            if is_phish:
                st.markdown(f"""
                <div class="result-danger">
                    <h2>⚠️ PHISHING DETECTED</h2>
                    <p>Suspicious patterns identified. <strong>Do not interact with this content.</strong></p>
                    <p>Confidence: <strong>{confidence*100:.1f}%</strong></p>
                    <p style="color: #94a3b8; font-size: 0.85rem;">Model: {model_used}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <h2>✅ APPEARS SAFE</h2>
                    <p>No significant phishing indicators detected.</p>
                    <p>Confidence: <strong>{confidence*100:.1f}%</strong></p>
                    <p style="color: #94a3b8; font-size: 0.85rem;">Model: {model_used}</p>
                </div>
                """, unsafe_allow_html=True)

            # AI-Generated badge
            if has_email:
                if ai_generated:
                    st.markdown(
                        '<span class="ai-badge ai-badge-positive">'
                        f'🤖 Likely AI-Generated ({ai_score*100:.0f}%)</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="ai-badge ai-badge-negative">'
                        f'👤 Likely Human-Written ({(1-ai_score)*100:.0f}%)</span>',
                        unsafe_allow_html=True,
                    )

            st.markdown("")

            # ── Sub-model score cards ─────────────────────────────────────
            if mode == "🔗+📧 Combined (Hybrid)":
                st.markdown("#### 📊 Sub-Model Scores")
                st.plotly_chart(
                    make_score_comparison(text_score, url_score, ai_score),
                    use_container_width=True,
                )

            # ── Key binary flags grid ─────────────────────────────────────
            if features:
                st.markdown("#### 🔬 Key Flags")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("HTTPS",        "✅ Yes" if features.get("is_https", 0)       else "❌ No")
                    st.metric("IP in URL",    "⚠️ Yes" if features.get("has_ip_address", 0) else "✅ No")
                with c2:
                    st.metric("@ Symbol",     "⚠️ Yes" if features.get("has_at_symbol", 0)  else "✅ No")
                    st.metric("Shortened URL","⚠️ Yes" if features.get("is_shortened_url", 0) else "✅ No")
                with c3:
                    st.metric("Subdomains",          int(features.get("subdomain_count", 0)))
                    st.metric("Suspicious Keywords", int(features.get("suspicious_keyword_count", 0)))

                # ── Feature bar chart ─────────────────────────────────────
                st.plotly_chart(make_feature_bar(features), use_container_width=True)

                # ── Raw features expander ─────────────────────────────────
                with st.expander("🔩 Raw Feature Vector (all 40)"):
                    ext_names = get_extended_feature_names()
                    ext_keys = get_extended_feature_keys()
                    raw_df = pd.DataFrame({
                        "Feature":  ext_names,
                        "Key":      ext_keys,
                        "Value":    [
                            round(features.get(k, 0), 4)
                            for k in ext_keys
                        ],
                    })
                    st.dataframe(raw_df, use_container_width=True, hide_index=True)

        with right:
            # ── Confidence gauge ──────────────────────────────────────────
            st.plotly_chart(make_gauge(confidence, is_phish), use_container_width=True)

            # ── Explainability section ────────────────────────────────────
            if pipeline and has_url:
                with st.expander("🧠 Explainability (SHAP)", expanded=False):
                    try:
                        explanation = pipeline.explain_prediction(
                            email=email_input if has_email else None,
                            url=url_input,
                        )
                        exp_text = explanation.get("explanation_text", "")
                        if exp_text:
                            st.markdown(exp_text)

                        url_exp = explanation.get("url_explanation", {})
                        top_feats = url_exp.get("top_features", [])
                        if top_feats:
                            from src.ml.explainability import PredictionExplainer
                            explainer = PredictionExplainer()
                            fig = explainer.plot_shap_waterfall(url_exp)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("SHAP analysis requires trained XGBoost model.")
                    except Exception as e:
                        st.info(f"Explainability not available: {e}")

        # ── Update history ───────────────────────────────────────────────
        entry = {
            "Input": "",
            "Result": "⚠️ Phishing" if is_phish else "✅ Safe",
            "Confidence": f"{confidence*100:.1f}%",
            "Model": model_used,
        }
        if has_url:
            entry["Input"] = url_input[:50] + ("…" if len(url_input) > 50 else "")
        elif has_email:
            entry["Input"] = email_input[:50] + ("…" if len(email_input) > 50 else "")

        if has_email:
            entry["AI-Gen"] = "🤖 Yes" if ai_generated else "👤 No"

        st.session_state["history"].append(entry)

    # ── Scan history ─────────────────────────────────────────────────────
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("### 📋 Scan History")
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History"):
            st.session_state["history"] = []
            st.rerun()


if __name__ == "__main__":
    main()
