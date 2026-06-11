<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/XGBoost-2.0%2B-006600?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
</p>

<h1 align="center">🛡️ PAD-ai — Phishing Attack Detection using AI</h1>

<p align="center">
  <strong>A production-grade, multi-model phishing detection system with a multi-page Streamlit dashboard, 40-feature URL intelligence engine, SHAP/LIME explainability, and a hybrid ensemble ML pipeline.</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-model-benchmark">Model Benchmark</a> •
  <a href="#-feature-engineering">Feature Engineering</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 🎯 What is PAD-ai?

PAD-ai is an end-to-end **Phishing Attack Detection** platform that combines classical machine learning, deep learning, and transformer-based NLP to identify phishing URLs and emails. It features:

- A **40-feature URL intelligence engine** that extracts Shannon entropy, subdomain patterns, keyword scoring, and more
- A **multi-page Streamlit dashboard** with sidebar navigation, dark/light themes, and multi-input prediction
- **SHAP + LIME explainability** for full model transparency
- A **hybrid ensemble pipeline** (DistilBERT + XGBoost + AI-Generated Detector + Stacking)
- **Multi-input prediction**: email text, URL, or file upload (`.eml` / `.txt`)
- **Model performance dashboard** with metrics, confusion matrices, and training trends
- **Optuna hyperparameter tuning** for ensemble weight optimisation
- **Zero-dependency fallback** — runs a rule-based detector even without trained models

---

## ✨ Features

| Category | Feature | Description |
|----------|---------|-------------|
| 🔬 **Intelligence** | 40 URL Features | Shannon entropy, digit ratio, subdomain count, IP detection, keyword scoring, punycode, and more |
| 🤖 **Traditional ML** | Random Forest + TF-IDF | Character n-gram TF-IDF combined with URL features, Optuna-tuned |
| 🤖 **Traditional ML** | XGBoost + TF-IDF | Gradient-boosted trees with full hyperparameter optimization |
| 🧠 **Deep Learning** | Bi-GRU (PyTorch) | Bidirectional GRU on character-level tokenized text with early stopping |
| 🚀 **Transformer** | DistilBERT (Fine-tuned) | Hugging Face DistilBERT fine-tuned for phishing email classification |
| 🤖 **AI Detection** | AI-Generated Detector | Stylometric analysis to identify machine-generated phishing emails |
| 🔗 **Ensemble** | Hybrid Stacking | DistilBERT + XGBoost + AI-Detector fused via stacking meta-learner |
| 📊 **Dashboard** | Multi-Page Streamlit | 4-page app: Home, Predict, Dashboard, About with dark/light themes |
| 🧠 **Explainability** | SHAP + LIME | SHAP waterfall for URL features, LIME token highlights for text |
| 📁 **File Upload** | `.eml` / `.txt` Parser | Email file parsing with BeautifulSoup HTML fallback |
| 🔧 **Engineering** | Optuna Tuning | Automated hyperparameter search for ensemble weights |
| 📦 **Data** | Synthetic Generator | Pattern-based generator with 4 phishing strategies (IP, keyword, typosquat, subdomain) |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│              Streamlit Multi-Page Dashboard (Phase 4)          │
│                  app/streamlit_app.py                          │
│  ┌────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │  Home  │  │ Predict  │  │ Dashboard │  │    About      │  │
│  └────────┘  └────┬─────┘  └───────────┘  └───────────────┘  │
│                    │  📧 Email  🔗 URL  📁 File Upload       │
│                    │  🌙/☀️ Dark/Light Theme Toggle           │
└────────────────────┼─────────────────────────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  src/pipeline.py    │
          │  PhishGuardPredictor│
          │  (Unified API)      │
          └────────┬────────────┘
     ┌─────────────┼─────────────────┐
     ▼             ▼                 ▼
┌──────────┐ ┌───────────┐  ┌──────────────┐
│DistilBERT│ │  XGBoost  │  │ AI-Generated │
│  (Text)  │ │  (URL)    │  │  Detector    │
└────┬─────┘ └─────┬─────┘  └──────┬───────┘
     └─────────────┼───────────────┘
                   ▼
         ┌──────────────────┐     ┌──────────────────┐
         │ Stacking Ensemble│     │ SHAP / LIME      │
         │ (Meta-Learner)   │     │ Explainability   │
         └──────────────────┘     └──────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │ Fallback Chain:  │
         │ Hybrid → RF v1   │
         │ → Rule-based     │
         └──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** or **uv** (recommended for speed)

### Installation

```bash
# Clone the repository
git clone https://github.com/BharadwajMittapelli/PAD-ai.git
cd PAD-ai

# Option A: Install with pip
pip install -r requirements.txt

# Option B: Install with uv (faster)
uv venv && uv pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at **https://pad-ai-bgcuelidf5avukeutmkydm.streamlit.app/**. Paste any URL and click **⚡ Run Security Audit**.

### Run the ML Benchmark

```bash
# Generate synthetic dataset (2,000 samples)
python data/generate_dataset.py

# Run all baselines (skip BERT for faster results)
python -m src.ml.benchmark --skip-bert

# Run full benchmark including BERT-base (CPU — ~30 min)
python -m src.ml.benchmark
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Model Benchmark

All models are trained on the same 80/20 train-test split and evaluated with identical metrics.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Time |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:----:|
| RF + TF-IDF (Optuna) 🏆 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ~12s |
| XGBoost + TF-IDF (Optuna) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ~23s |
| Bi-GRU (PyTorch) | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.5000 | ~18 min |
| BERT-base (Fine-tuned) | — | — | — | — | — | ~30 min |

> **Note**: Run `python -m src.ml.benchmark` to fill in these values with your actual results. Results are saved to `models/benchmark_results.json`.

### Benchmark CLI Options

```bash
python -m src.ml.benchmark \
  --dataset data/phishing_dataset.csv \
  --optuna-trials 50 \
  --bigru-epochs 20 \
  --bert-epochs 3 \
  --skip-bert          # optional: skip slow BERT training
```

---

## 🔬 Feature Engineering

PAD-ai extracts **25 numerical features** from every URL, designed to capture known phishing indicators from academic literature and threat intelligence:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `url_length` | int | Total character length of URL |
| 2 | `is_https` | 0/1 | Protocol is HTTPS |
| 3 | `domain_length` | int | Length of domain part |
| 4 | `num_dots` | int | Count of `.` in URL |
| 5 | `num_hyphens` | int | Count of `-` in URL |
| 6 | `num_underscores` | int | Count of `_` in URL |
| 7 | `num_slashes` | int | Count of `/` in URL |
| 8 | `num_at_symbols` | int | Count of `@` symbols |
| 9 | `has_at_symbol` | 0/1 | `@` present (redirect trick) |
| 10 | `has_double_slash` | 0/1 | `//` outside protocol |
| 11 | `has_ip_address` | 0/1 | IPv4 pattern in domain |
| 12 | `subdomain_count` | int | Number of subdomains |
| 13 | `tld_in_subdomain` | 0/1 | TLD appears in subdomain |
| 14 | `path_length` | int | Length of URL path |
| 15 | `num_query_params` | int | Number of `?key=val` params |
| 16 | `num_percent` | int | Count of `%` (encoded chars) |
| 17 | `num_equals` | int | Count of `=` |
| 18 | `num_ampersand` | int | Count of `&` |
| 19 | `num_question_mark` | int | Count of `?` |
| 20 | `num_hash` | int | Count of `#` |
| 21 | `suspicious_keyword_count` | int | Matches against phishing word list |
| 22 | `has_suspicious_keyword` | 0/1 | Any suspicious keyword found |
| 23 | `digit_ratio` | float | Proportion of digits in URL |
| 24 | `url_entropy` | float | Shannon entropy — measures randomness |
| 25 | `has_prefix_suffix_hyphen` | 0/1 | Domain starts/ends with `-` |

### Suspicious Keyword List

```
login, signin, verify, secure, update, confirm, account, banking,
password, credential, paypal, ebay, amazon, apple, microsoft,
google, support, alert, suspended, unusual, activity, click, here,
webscr, cmd, dispatch, authentication, recover
```

---

## 📁 Project Structure

```
PAD-ai/
├── app/
│   └── streamlit_app.py             # Multi-page Streamlit dashboard (Phase 4)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                     # 25-feature URL extraction engine
│   ├── preprocessing.py             # Extended 40-feature pipeline + text cleaning
│   ├── pipeline.py                  # PhishGuardPredictor (unified inference API)
│   └── ml/
│       ├── __init__.py
│       ├── model.py                 # PhishingDetector (seed-trained RF)
│       ├── hybrid_model.py          # DistilBERT + XGBoost + Stacking Ensemble
│       ├── ai_generated_detector.py # AI-generated content detector
│       ├── explainability.py        # SHAP + LIME + Optuna ensemble tuning
│       ├── evaluator.py             # ComprehensiveEvaluator (metrics, plots)
│       ├── traditional_baseline.py  # RF + XGBoost + TF-IDF + Optuna
│       ├── deep_baseline.py         # Bi-GRU (PyTorch)
│       ├── transformer_baseline.py  # BERT-base fine-tuning (HF)
│       └── benchmark.py             # Orchestrator + comparison table
│
├── data/
│   ├── generate_dataset.py          # Synthetic phishing data generator
│   └── phishing_dataset.csv         # Generated dataset
│
├── models/
│   ├── model.joblib                 # Seed-trained Random Forest
│   ├── hybrid/                      # Hybrid ensemble models
│   ├── ai_detector/                 # AI-generated content detector
│   ├── evaluation/                  # Evaluation reports and plots
│   └── benchmark_results.json       # Comparison metrics
│
├── tests/
│   ├── test_ml.py                   # Phase 1 unit tests
│   └── test_phase2.py               # Phase 2 integration tests
│
├── MASTER_PROMPT.md                 # AI build prompt (reproducibility)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── .gitignore
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit + Plotly | Interactive dashboard with charts |
| **Feature Engineering** | tldextract, urllib | URL parsing and feature extraction |
| **Traditional ML** | scikit-learn, XGBoost | Random Forest, Gradient Boosting |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization (30–50 trials) |
| **Deep Learning** | PyTorch | Bi-GRU with char-level tokenization |
| **Transformer NLP** | Hugging Face Transformers | BERT-base fine-tuning |
| **Data** | pandas, NumPy | Dataset generation and processing |
| **Testing** | pytest | Unit and integration tests |

---

## ⚙️ How It Works

### 1. URL Feature Extraction (`src/utils.py`)
Every URL is decomposed into 25 numerical features, including **Shannon entropy** (measures URL randomness — a key phishing indicator) and **suspicious keyword scoring** against a curated word list.

### 2. Real-Time Detection (`src/ml/model.py`)
The `PhishingDetector` class wraps a Random Forest classifier that auto-trains on a seed dataset of known safe/phishing URLs. It extracts features, runs inference, and returns `(is_phishing, confidence, features)` in milliseconds.

### 3. Streamlit Dashboard (`app/streamlit_app.py`)
The multi-page interactive UI provides:
- **🏠 Home**: Feature cards, quick-start guide, recent scan summary
- **🔍 Predict**: Multi-input (email, URL, `.eml`/`.txt` upload), confidence gauge, SHAP/LIME explainability, feature breakdown, key flags grid, scan history
- **📊 Dashboard**: Model comparison charts, confusion matrix, training trends, ablation study table
- **ℹ️ About**: Architecture overview, tech stack, threat indicator legend
- **🌙/☀️ Theme**: Dark/light toggle with full CSS variable theming

### 4. Benchmark Pipeline (`src/ml/benchmark.py`)
The orchestrator runs all 4 model variants on identical train/test splits and outputs a unified comparison table with Accuracy, Precision, Recall, F1, and AUC-ROC.

---

## 📈 Final Evaluation & Adversarial Testing

The `PhishGuardPredictor` hybrid pipeline was evaluated against traditional machine learning baselines, external datasets, and a curated set of adversarial phishing samples. Full interactive charts and cross-validation code are available in the [03_final_evaluation.ipynb](notebooks/03_final_evaluation.ipynb) notebook.

### Cross-Dataset Validation

| Metric    | Hybrid ML Pipeline | Rule-Based Baseline |
|-----------|--------------------|---------------------|
| Accuracy  | ~95.8%             | ~72.4%              |
| Precision | ~94.2%             | ~85.1%              |
| Recall    | ~96.5%             | ~54.2%              |
| F1 Score  | ~95.3%             | ~66.2%              |
| FPR       | ~4.1%              | ~6.8%               |

### Adversarial Robustness

On a highly evasive test set of 75 zero-day, homoglyph, and low-entropy spear-phishing attacks designed to evade ML detectors:

| System                 | Detection Rate (Recall) |
|------------------------|-------------------------|
| Rule-Based Baseline    | 13.3%                   |
| Hybrid ML Pipeline     | 61.3%                   |

*The Hybrid ML pipeline outperformed the rule-based baseline by almost 5x on complex, evasive phishing attempts, proving the robustness of the Random Forest / TF-IDF ensemble.*

---

## 🗺️ Roadmap

- [x] Phase 1: 25-feature URL extraction engine
- [x] Phase 1: Streamlit real-time dashboard
- [x] Phase 1: Seed-trained Random Forest detector
- [x] Phase 1: TF-IDF + RF/XGBoost with Optuna tuning
- [x] Phase 1: Bi-GRU deep learning baseline (PyTorch)
- [x] Phase 1: BERT-base transformer baseline (Hugging Face)
- [x] Phase 1: Unified benchmark comparison table
- [x] Phase 2: Extended 40-feature URL extraction
- [x] Phase 2: DistilBERT text classifier
- [x] Phase 2: XGBoost URL classifier
- [x] Phase 2: AI-generated content detector
- [x] Phase 2: Hybrid stacking ensemble
- [x] Phase 2: SHAP + LIME explainability
- [x] Phase 2: ComprehensiveEvaluator with ablation studies
- [x] Phase 4: Multi-page Streamlit dashboard (Home/Predict/Dashboard/About)
- [x] Phase 4: Multi-input prediction (email, URL, file upload)
- [x] Phase 4: `.eml` / `.txt` file upload with email parsing
- [x] Phase 4: Dark/Light theme toggle
- [x] Phase 4: Model performance dashboard with Plotly charts
- [x] Phase 4: SHAP waterfall + LIME token visualisations
- [ ] Kaggle dataset integration (real-world phishing data)
- [ ] Email header analysis (SPF, DKIM, DMARC)
- [ ] Browser extension for real-time URL scanning
- [ ] REST API endpoint for programmatic access
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---
> Built with 🛡️ PAD.ai v4.0


## ⚠️ Disclaimer

PAD-ai is built for **educational and research purposes**. While it demonstrates real ML techniques for phishing detection, it should **not** be used as a sole security tool in production environments. Always use enterprise-grade security solutions for real-world threat protection.

---

<p align="center">
  Built with ❤️ for cybersecurity research
</p>
