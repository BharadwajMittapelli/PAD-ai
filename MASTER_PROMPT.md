# 🛡️ PAD.ai — Master Build Prompt

## Role
You are an expert Python developer and cybersecurity engineer.
Build a complete, modular, production-grade **Phishing Attack Detection (PAD.ai)** system.

---

## Project Goal
Build a well-structured ML-powered phishing detection system with:
- **25+ engineered URL features** extracted in `src/utils.py`
- **Random Forest classifier** encapsulated in `src/ml/model.py`
- **Streamlit interactive dashboard** in `app/streamlit_app.py`

---

## Architecture Rules (Non-Negotiable)
1. NEVER mix ML logic into the UI layer
2. All feature extraction lives **only** in `src/utils.py`
3. Model class in `src/ml/model.py` — zero UI code there
4. Streamlit app only imports from `src/` — no raw scikit-learn calls
5. All paths use `os.path` — no hardcoded strings

---

## Directory Structure to Build

```
PAD.ai/
├── app/
│   └── streamlit_app.py        ← Streamlit dashboard (main entry)
├── src/
│   ├── __init__.py
│   ├── utils.py                ← URL feature extraction & utilities
│   └── ml/
│       ├── __init__.py
│       └── model.py            ← PhishingDetector class
├── data/
│   └── .gitkeep               ← Placeholder for training CSVs
├── models/
│   └── .gitkeep               ← Saved model files (joblib)
├── tests/
│   └── test_ml.py             ← Unit tests
├── requirements.txt
├── README.md
└── .gitignore
```

---

## File-by-File Specification

### `src/utils.py`
Extract these 25 features from any URL string:

| Feature | Type | Description |
|---------|------|-------------|
| `url_length` | int | Total character length of URL |
| `is_https` | 0/1 | Protocol is HTTPS |
| `domain_length` | int | Length of domain part |
| `num_dots` | int | Count of '.' in URL |
| `num_hyphens` | int | Count of '-' in URL |
| `num_underscores` | int | Count of '_' in URL |
| `num_slashes` | int | Count of '/' in URL |
| `num_at_symbols` | int | Count of '@' symbols |
| `has_at_symbol` | 0/1 | '@' present (redirect trick) |
| `has_double_slash` | 0/1 | '//' outside protocol |
| `has_ip_address` | 0/1 | IPv4 pattern in domain |
| `subdomain_count` | int | Number of subdomains |
| `tld_in_subdomain` | 0/1 | TLD appears in subdomain |
| `path_length` | int | Length of URL path |
| `num_query_params` | int | Number of '?key=val' params |
| `num_percent` | int | Count of '%' (encoded chars) |
| `num_equals` | int | Count of '=' |
| `num_ampersand` | int | Count of '&' |
| `num_question_mark` | int | Count of '?' |
| `num_hash` | int | Count of '#' |
| `suspicious_keyword_count` | int | Matches against phishing word list |
| `has_suspicious_keyword` | 0/1 | Any suspicious keyword found |
| `digit_ratio` | float | Proportion of digits in URL |
| `url_entropy` | float | Shannon entropy — measures randomness |
| `has_prefix_suffix_hyphen` | 0/1 | Domain starts/ends with '-' |

**Required functions:**
```
extract_url_features(url: str) -> dict
features_to_array(features: dict) -> list   # ordered for sklearn
get_feature_names() -> list                 # for UI display
_calculate_entropy(text: str) -> float      # internal helper
```

**Suspicious keyword list must include:**
`login, signin, verify, secure, update, confirm, account, banking,
password, credential, paypal, ebay, amazon, apple, microsoft,
google, support, alert, suspended, unusual, activity`

---

### `src/ml/model.py`
Build a `PhishingDetector` class with:

```
class PhishingDetector:
    - __init__(): load saved model or mark as untrained
    - _train_seed(): train on hardcoded seed dataset (list of known safe/phishing URLs)
    - train(X, y): train on custom dataset
    - predict(url: str) -> tuple[bool, float, dict]:
        returns (is_phishing, confidence_score, feature_dict)
    - save(path): persist model with joblib
    - load(path): restore model from disk
    - property is_trained -> bool
```

**Seed training data must include at minimum:**
- 10 known safe URLs (google.com, github.com, etc.)
- 10 known phishing-pattern URLs (IP-based, hyphenated login pages, etc.)

**Model choice:** `RandomForestClassifier(n_estimators=100, random_state=42)`

---

### `app/streamlit_app.py`
Build a full Streamlit dashboard with these sections:

**Layout:**
- `st.set_page_config` — wide layout, shield icon, "PAD.ai" title
- Gradient CSS header via `st.markdown(unsafe_allow_html=True)`
- Sidebar with: logo text, About, How it works, Threat Indicators legend

**Main Panel:**
- Text input for URL
- "⚡ Run Security Audit" primary button
- On click:
  1. Call `extract_url_features(url)`
  2. Call `detector.predict(url)`
  3. Show result card (red danger / green safe with confidence %)
  4. Show Plotly **gauge chart** for confidence (0–100%)
  5. Show Plotly **bar chart** for top URL features
  6. Show **3-column metric grid** for key binary flags
  7. Show **expandable raw features table** (all 25 features)

**Session State:**
- Maintain scan history in `st.session_state.history`
- Show history table at bottom
- "🗑️ Clear History" button

---

### `requirements.txt`
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
streamlit>=1.28.0
plotly>=5.17.0
tldextract>=3.4.0
requests>=2.31.0
pytest>=7.4.0
```

---

## Code Quality Requirements
- Type hints on all function signatures
- Docstrings on all public methods
- `try/except` for all URL parsing (malformed inputs are common)
- Graceful fallback if `tldextract` is unavailable
- Never call `st.experimental_*` — use stable Streamlit API only
- No `print()` statements in production code — use `st.write()` or logging

## DO NOT
- Use FastAPI (project switches fully to Streamlit)
- Return mock/hardcoded predictions — build real ML pipeline
- Skip feature engineering — the 25 features ARE the intelligence
- Mix UI imports into `src/` modules
- Use `st.cache` — use `@st.cache_resource` for the model loader
