"""
app/streamlit_app.py
====================
PAD.ai — Phishing Attack Detection Dashboard

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
import plotly.graph_objects as go
import plotly.express as px

from src.utils import extract_url_features, get_feature_names, get_feature_keys

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAD.ai — Phishing Detection",
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
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached across reruns) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model…")
def load_detector():
    """Load PhishingDetector — falls back to rule-based scoring on failure."""
    try:
        from src.ml.model import PhishingDetector
        detector = PhishingDetector()
        if not detector.is_trained:
            detector._train_seed()
        return detector
    except Exception as exc:
        st.warning(f"ML model unavailable ({exc}). Using rule-based fallback.")
        return None


def rule_based_predict(features: dict):
    """
    Simple rule-based fallback when the ML model is not available.
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
        "URL Length":            features["url_length"],
        "Domain Length":         features["domain_length"],
        "Dot Count":             features["num_dots"],
        "Hyphen Count":          features["num_hyphens"],
        "Subdomain Count":       features["subdomain_count"],
        "Path Length":           features["path_length"],
        "Suspicious Keywords":   features["suspicious_keyword_count"],
        "Digit Ratio ×10":       round(features["digit_ratio"] * 10, 2),
        "URL Entropy ×10":       round(features["url_entropy"] * 10, 2),
        "Query Params":          features["num_query_params"],
    }
    labels = list(display.keys())
    values = list(display.values())

    fig = px.bar(
        x=labels, y=values,
        color=values,
        color_continuous_scale=["#10b981", "#eab308", "#ef4444"],
        labels={"x": "", "y": "Value", "color": "Value"},
        title="Feature Breakdown",
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


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ PAD.ai")
        st.markdown("*Phishing Attack Detection using AI*")
        st.divider()

        st.markdown("### How it works")
        st.markdown("""
1. 🔗 Paste a URL
2. 🔬 25 features extracted
3. 🤖 Random Forest classifies
4. ⚡ Result in milliseconds
        """)
        st.divider()

        st.markdown("### Threat Indicators")
        indicators = [
            ("IP address in URL",    "🔴 High Risk"),
            ("No HTTPS",             "🔴 High Risk"),
            ("Suspicious keywords",  "🔴 High Risk"),
            ("High URL entropy",     "🟡 Medium"),
            ("Many subdomains",      "🟡 Medium"),
            ("@ symbol in URL",      "🔴 High Risk"),
        ]
        for label, risk in indicators:
            st.markdown(f"**{label}** — {risk}")

        st.divider()
        st.caption("PAD.ai v1.0 · Educational use only")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Header
    st.markdown('<p class="pad-header">🛡️ PAD.ai</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pad-sub">Phishing Attack Detection using Artificial Intelligence</p>',
        unsafe_allow_html=True,
    )

    # ── Input area ────────────────────────────────────────────────────────────
    col_input, col_gap = st.columns([3, 1])
    with col_input:
        url_input = st.text_input(
            "URL to analyse",
            placeholder="https://example.com  or  http://suspicious-login-verify.com",
            label_visibility="collapsed",
        )
        analyse = st.button("⚡ Run Security Audit", type="primary", use_container_width=True)

    # ── Session history init ───────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyse:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
            st.stop()

        with st.spinner("Analysing URL…"):
            features   = extract_url_features(url_input)
            detector   = load_detector()

            if detector is not None:
                is_phish, confidence, _ = detector.predict(url_input)
            else:
                is_phish, confidence = rule_based_predict(features)

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown("---")
        left, right = st.columns([3, 2])

        with left:
            if is_phish:
                st.markdown(f"""
                <div class="result-danger">
                    <h2>⚠️ PHISHING DETECTED</h2>
                    <p>Suspicious patterns identified. <strong>Do not visit this URL.</strong></p>
                    <p>Confidence: <strong>{confidence*100:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <h2>✅ URL APPEARS SAFE</h2>
                    <p>No significant phishing indicators detected.</p>
                    <p>Confidence: <strong>{confidence*100:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # ── Key binary flags grid ──────────────────────────────────────
            st.markdown("#### 🔬 Key Flags")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("HTTPS",        "✅ Yes" if features["is_https"]       else "❌ No")
                st.metric("IP in URL",    "⚠️ Yes" if features["has_ip_address"] else "✅ No")
            with c2:
                st.metric("@ Symbol",     "⚠️ Yes" if features["has_at_symbol"]  else "✅ No")
                st.metric("Double Slash", "⚠️ Yes" if features["has_double_slash"] else "✅ No")
            with c3:
                st.metric("Subdomains",          int(features["subdomain_count"]))
                st.metric("Suspicious Keywords", int(features["suspicious_keyword_count"]))

            # ── Feature bar chart ──────────────────────────────────────────
            st.plotly_chart(make_feature_bar(features), use_container_width=True)

            # ── Raw features expander ──────────────────────────────────────
            with st.expander("🔩 Raw Feature Vector (all 25)"):
                raw_df = pd.DataFrame({
                    "Feature":  get_feature_names(),
                    "Key":      get_feature_keys(),
                    "Value":    [
                        round(features.get(k, 0), 4)
                        for k in get_feature_keys()
                    ],
                })
                st.dataframe(raw_df, use_container_width=True, hide_index=True)

        with right:
            # ── Confidence gauge ───────────────────────────────────────────
            st.plotly_chart(make_gauge(confidence, is_phish), use_container_width=True)

        # ── Update history ─────────────────────────────────────────────────
        st.session_state["history"].append({
            "URL":        url_input[:55] + ("…" if len(url_input) > 55 else ""),
            "Result":     "⚠️ Phishing" if is_phish else "✅ Safe",
            "Confidence": f"{confidence*100:.1f}%",
        })

    # ── Scan history ──────────────────────────────────────────────────────────
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
