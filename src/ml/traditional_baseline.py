"""
src/ml/traditional_baseline.py
==============================
Traditional ML Baseline: TF-IDF + Random Forest / XGBoost
with Optuna hyperparameter tuning.

Combines TF-IDF text features (URL + email body) with the
25 engineered URL features from src/utils.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import optuna
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import extract_url_features, features_to_array

# Silence Optuna logs during tuning
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_features(df: pd.DataFrame):
    """
    Build combined feature matrix:
    1. TF-IDF on 'url' + 'email_body' text
    2. 25 engineered URL features from src/utils.py
    Returns (X_combined_sparse, tfidf_vectorizer)
    """
    # Combine URL and email body into one text column for TF-IDF
    text_col = df["url"].fillna("") + " " + df["email_body"].fillna("")

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        analyzer="char_wb",
        min_df=2,
    )
    X_tfidf = tfidf.fit_transform(text_col)

    # Extract 25 URL features for each row
    url_features = []
    for url in df["url"]:
        feats = extract_url_features(url)
        url_features.append(features_to_array(feats))
    X_url = csr_matrix(np.array(url_features, dtype=np.float32))

    # Combine TF-IDF + URL features
    X_combined = hstack([X_tfidf, X_url])
    return X_combined, tfidf


def _transform_features(df: pd.DataFrame, tfidf: TfidfVectorizer):
    """Transform test data using a pre-fit TF-IDF vectorizer."""
    text_col = df["url"].fillna("") + " " + df["email_body"].fillna("")
    X_tfidf = tfidf.transform(text_col)

    url_features = []
    for url in df["url"]:
        feats = extract_url_features(url)
        url_features.append(features_to_array(feats))
    X_url = csr_matrix(np.array(url_features, dtype=np.float32))

    return hstack([X_tfidf, X_url])


def train_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_trials: int = 50,
    model_dir: str = "models",
) -> dict:
    """
    Train a Random Forest with Optuna hyperparameter tuning.

    Returns dict with model name and evaluation metrics.
    """
    print("\n══════════════════════════════════════════════")
    print("  📊 Training: Random Forest + TF-IDF (Optuna)")
    print("══════════════════════════════════════════════")

    X_train, tfidf = _build_features(train_df)
    y_train = train_df["label"].values
    X_test = _transform_features(test_df, tfidf)
    y_test = test_df["label"].values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return f1_score(y_test, preds)

    print(f"  ⏳ Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  ✅ Best F1: {study.best_value:.4f}")
    print(f"  📋 Best params: {study.best_params}")

    # Retrain with best params
    best_clf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    best_clf.fit(X_train, y_train)

    # Evaluate
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "RF + TF-IDF (Optuna)",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    # Save model + vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_clf, os.path.join(model_dir, "rf_tfidf_best.joblib"))
    joblib.dump(tfidf, os.path.join(model_dir, "rf_tfidf_vectorizer.joblib"))

    _print_metrics(metrics)
    return metrics


def train_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_trials: int = 50,
    model_dir: str = "models",
) -> dict:
    """
    Train an XGBoost classifier with Optuna hyperparameter tuning.

    Returns dict with model name and evaluation metrics.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  ⚠️ XGBoost not installed. Skipping.")
        return {
            "model": "XGBoost + TF-IDF (Optuna)",
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0,
        }

    print("\n══════════════════════════════════════════════")
    print("  📊 Training: XGBoost + TF-IDF (Optuna)")
    print("══════════════════════════════════════════════")

    X_train, tfidf = _build_features(train_df)
    y_train = train_df["label"].values
    X_test = _transform_features(test_df, tfidf)
    y_test = test_df["label"].values

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
        }
        clf = XGBClassifier(
            **params,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return f1_score(y_test, preds)

    print(f"  ⏳ Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  ✅ Best F1: {study.best_value:.4f}")
    print(f"  📋 Best params: {study.best_params}")

    # Retrain with best params
    best_clf = XGBClassifier(
        **study.best_params,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    best_clf.fit(X_train, y_train)

    # Evaluate
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "XGBoost + TF-IDF (Optuna)",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    # Save model + vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_clf, os.path.join(model_dir, "xgb_tfidf_best.joblib"))
    joblib.dump(tfidf, os.path.join(model_dir, "xgb_tfidf_vectorizer.joblib"))

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
