"""
src/preprocessing.py
====================
PAD-ai Phase 2 — Advanced Feature Engineering Pipeline

Provides:
- Dataset loading and merging
- Text cleaning and normalisation
- Extended URL feature extraction (15 new features on top of the original 25)
- TF-IDF feature generation
- BERT/DistilBERT embedding extraction

Usage:
    from src.preprocessing import (
        load_and_merge_datasets,
        clean_text,
        extract_url_features_extended,
        create_tfidf_features,
        create_bert_embeddings,
    )
"""

import os
import re
import math
import string
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# ── Phase 1 imports (extend, don't replace) ─────────────────────────────────
from src.utils import extract_url_features, features_to_array


# ── Constants ────────────────────────────────────────────────────────────────

URL_SHORTENERS = frozenset([
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "short.io", "tiny.cc",
])

EXECUTABLE_EXTENSIONS = frozenset([
    ".exe", ".bat", ".cmd", ".scr", ".pif", ".msi", ".js", ".vbs",
    ".php", ".asp", ".aspx", ".cgi", ".pl", ".py", ".rb",
])

COMMON_TLDS = frozenset([
    "com", "org", "net", "edu", "gov", "io", "co", "uk", "de", "fr",
])

# Extended feature order (15 new features appended after the original 25)
EXTENDED_FEATURE_ORDER: List[str] = [
    "path_depth", "longest_subdomain_length", "vowel_consonant_ratio",
    "consecutive_char_repeat_max", "special_char_ratio", "has_port_number",
    "is_shortened_url", "domain_token_count", "avg_domain_token_length",
    "has_www_prefix", "tld_length", "path_has_extension",
    "fragment_length", "has_punycode", "domain_digit_ratio",
]

EXTENDED_FEATURE_DISPLAY: List[str] = [
    "Path Depth", "Longest Subdomain Length", "Vowel/Consonant Ratio",
    "Max Consecutive Char Repeat", "Special Char Ratio", "Has Port Number",
    "Is Shortened URL", "Domain Token Count", "Avg Domain Token Length",
    "Has www Prefix", "TLD Length", "Path Has Extension",
    "Fragment Length", "Has Punycode", "Domain Digit Ratio",
]


# ── 1. Dataset Loading ──────────────────────────────────────────────────────

def load_and_merge_datasets(
    paths: Optional[List[str]] = None,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load and merge one or more phishing dataset CSVs.

    Parameters
    ----------
    paths : list of str, optional
        Explicit CSV paths.  If None, loads all ``*.csv`` in *data_dir*.
    data_dir : str
        Directory to scan when *paths* is not given.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with columns: url, email_body, label
        (and optionally is_ai_generated).
    """
    if paths is None:
        paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        ]

    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # Validate required columns
        required = {"url", "email_body", "label"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise ValueError(f"{p} missing columns: {missing}")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Deduplicate on url + email_body
    before = len(merged)
    merged = merged.drop_duplicates(subset=["url", "email_body"], keep="first")
    after = len(merged)
    if before != after:
        print(f"  ℹ️  Removed {before - after} duplicate rows.")

    # Fill NaNs
    merged["url"] = merged["url"].fillna("")
    merged["email_body"] = merged["email_body"].fillna("")

    print(
        f"  📂 Loaded {len(merged)} samples "
        f"({merged['label'].sum()} phishing, "
        f"{(merged['label'] == 0).sum()} safe)"
    )
    return merged


# ── 2. Text Cleaning ────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str, remove_urls: bool = False) -> str:
    """
    Clean and normalise text for NLP processing.

    Steps:
        1. Lowercase
        2. Strip HTML tags
        3. Optionally remove embedded URLs
        4. Normalise whitespace
        5. Strip leading/trailing whitespace

    Parameters
    ----------
    text : str
        Raw text to clean.
    remove_urls : bool
        If True, strip embedded URLs from the text body.

    Returns
    -------
    str
        Cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _HTML_TAG_RE.sub(" ", text)

    if remove_urls:
        text = _URL_RE.sub(" ", text)

    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ── 3. Extended URL Feature Extraction ──────────────────────────────────────

def extract_url_features_extended(url: str) -> Dict[str, float]:
    """
    Extract 40 features from a URL string.

    Calls the original ``extract_url_features()`` for the first 25,
    then appends 15 new features.

    Parameters
    ----------
    url : str
        Raw URL string.

    Returns
    -------
    dict
        40-feature dictionary (25 original + 15 extended).
    """
    # Start with the original 25 features
    features = extract_url_features(url)

    # ── Parse URL components ─────────────────────────────────────────────
    raw_url = url.strip()
    normalised = raw_url if raw_url.startswith(("http://", "https://")) else "http://" + raw_url

    try:
        parsed = urlparse(normalised)
        domain = parsed.netloc
        path = parsed.path
        fragment = parsed.fragment
    except Exception:
        domain, path, fragment = "", "", ""

    # Strip port from domain for analysis
    domain_no_port = domain.split(":")[0] if domain else ""

    # ── 26. Path depth (number of path segments) ─────────────────────────
    segments = [s for s in path.split("/") if s]
    features["path_depth"] = len(segments)

    # ── 27. Longest subdomain part length ────────────────────────────────
    domain_parts = domain_no_port.split(".")
    if len(domain_parts) > 2:
        subdomain_parts = domain_parts[:-2]
        features["longest_subdomain_length"] = max(
            (len(p) for p in subdomain_parts), default=0
        )
    else:
        features["longest_subdomain_length"] = 0

    # ── 28. Vowel-to-consonant ratio in domain ──────────────────────────
    vowels = sum(1 for c in domain_no_port.lower() if c in "aeiou")
    consonants = sum(1 for c in domain_no_port.lower() if c.isalpha() and c not in "aeiou")
    features["vowel_consonant_ratio"] = round(
        vowels / max(consonants, 1), 4
    )

    # ── 29. Maximum consecutive character repetition ─────────────────────
    max_repeat = 0
    if raw_url:
        current_char, current_count = raw_url[0], 1
        for ch in raw_url[1:]:
            if ch == current_char:
                current_count += 1
                max_repeat = max(max_repeat, current_count)
            else:
                current_char = ch
                current_count = 1
        max_repeat = max(max_repeat, current_count)
    features["consecutive_char_repeat_max"] = max_repeat

    # ── 30. Special character ratio ──────────────────────────────────────
    n = len(raw_url)
    special_count = sum(
        1 for c in raw_url if not c.isalnum() and c not in (".", "/", ":", "-")
    )
    features["special_char_ratio"] = round(special_count / max(n, 1), 4)

    # ── 31. Has port number ──────────────────────────────────────────────
    has_port = 0
    if ":" in domain:
        port_str = domain.split(":")[-1]
        if port_str.isdigit() and int(port_str) not in (80, 443):
            has_port = 1
    features["has_port_number"] = has_port

    # ── 32. Is shortened URL ─────────────────────────────────────────────
    features["is_shortened_url"] = (
        1 if domain_no_port.lower() in URL_SHORTENERS else 0
    )

    # ── 33. Domain token count (split by hyphens and dots) ───────────────
    domain_tokens = re.split(r"[.\-]", domain_no_port)
    domain_tokens = [t for t in domain_tokens if t]
    features["domain_token_count"] = len(domain_tokens)

    # ── 34. Average domain token length ──────────────────────────────────
    if domain_tokens:
        features["avg_domain_token_length"] = round(
            sum(len(t) for t in domain_tokens) / len(domain_tokens), 2
        )
    else:
        features["avg_domain_token_length"] = 0

    # ── 35. Has www prefix ───────────────────────────────────────────────
    features["has_www_prefix"] = (
        1 if domain_no_port.lower().startswith("www.") else 0
    )

    # ── 36. TLD length ───────────────────────────────────────────────────
    tld = domain_parts[-1] if domain_parts else ""
    features["tld_length"] = len(tld)

    # ── 37. Path has executable/dangerous extension ──────────────────────
    path_lower = path.lower()
    features["path_has_extension"] = (
        1 if any(path_lower.endswith(ext) for ext in EXECUTABLE_EXTENSIONS) else 0
    )

    # ── 38. Fragment length ──────────────────────────────────────────────
    features["fragment_length"] = len(fragment)

    # ── 39. Has Punycode (internationalized domain attack) ───────────────
    features["has_punycode"] = 1 if "xn--" in domain_no_port.lower() else 0

    # ── 40. Domain digit ratio ───────────────────────────────────────────
    domain_len = len(domain_no_port)
    domain_digits = sum(c.isdigit() for c in domain_no_port)
    features["domain_digit_ratio"] = round(
        domain_digits / max(domain_len, 1), 4
    )

    return features


def extended_features_to_array(features: Dict[str, float]) -> List[float]:
    """
    Convert an extended feature dict to a 40-element ordered list.

    Uses the original 25 features in their canonical order,
    followed by the 15 extended features.

    Parameters
    ----------
    features : dict
        Output of ``extract_url_features_extended()``.

    Returns
    -------
    list of float
        40 values in canonical order.
    """
    from src.utils import get_feature_keys

    base_order = get_feature_keys()
    full_order = base_order + EXTENDED_FEATURE_ORDER
    return [float(features.get(f, 0)) for f in full_order]


def get_extended_feature_names() -> List[str]:
    """Return human-readable names for all 40 features."""
    from src.utils import get_feature_names

    return get_feature_names() + EXTENDED_FEATURE_DISPLAY.copy()


def get_extended_feature_keys() -> List[str]:
    """Return raw feature keys for all 40 features."""
    from src.utils import get_feature_keys

    return get_feature_keys() + EXTENDED_FEATURE_ORDER.copy()


# ── 4. TF-IDF Feature Generation ────────────────────────────────────────────

def create_tfidf_features(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 3),
    vectorizer=None,
) -> Tuple:
    """
    Create TF-IDF features from text data.

    Parameters
    ----------
    texts : list of str
        Text documents (e.g., cleaned email bodies).
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple
        N-gram range for character-level features.
    vectorizer : TfidfVectorizer or None
        Pre-fit vectorizer for transform-only (test data).

    Returns
    -------
    (sparse_matrix, TfidfVectorizer)
        Sparse TF-IDF matrix and the fitted vectorizer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer="char_wb",
            min_df=2,
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)

    return X, vectorizer


# ── 5. BERT Embedding Extraction ────────────────────────────────────────────

def create_bert_embeddings(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate [CLS] token embeddings using a pretrained transformer.

    Parameters
    ----------
    texts : list of str
        Input texts to embed.
    model_name : str
        Hugging Face model identifier.
    batch_size : int
        Batch size for inference.
    max_length : int
        Maximum token sequence length.
    device : str
        Device for inference ('cpu' or 'cuda').

    Returns
    -------
    np.ndarray
        Shape (n_samples, hidden_dim) embeddings — 768 for DistilBERT.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  ⏳ Loading {model_name} for embedding extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []
    n_batches = math.ceil(len(texts) / batch_size)

    for i in range(n_batches):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # Extract [CLS] token embedding (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"    Batch {i+1}/{n_batches} processed")

    embeddings = np.vstack(all_embeddings)
    print(f"  ✅ Generated embeddings: {embeddings.shape}")
    return embeddings


# ── 6. Batch Feature Engineering ────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    include_tfidf: bool = True,
    include_url_features: bool = True,
    tfidf_vectorizer=None,
) -> Tuple:
    """
    Build a combined feature matrix from a DataFrame.

    Combines TF-IDF text features with extended URL features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'url' and 'email_body' columns.
    include_tfidf : bool
        Whether to include TF-IDF features.
    include_url_features : bool
        Whether to include the 40 URL features.
    tfidf_vectorizer : TfidfVectorizer or None
        Pre-fit vectorizer (use for test data).

    Returns
    -------
    (feature_matrix, tfidf_vectorizer)
        Combined feature matrix and the vectorizer.
    """
    from scipy.sparse import hstack

    matrices = []

    # TF-IDF on combined text
    if include_tfidf:
        texts = (
            df["url"].fillna("") + " " + df["email_body"].fillna("")
        ).apply(clean_text).tolist()
        X_tfidf, tfidf_vectorizer = create_tfidf_features(
            texts, vectorizer=tfidf_vectorizer
        )
        matrices.append(X_tfidf)

    # Extended URL features
    if include_url_features:
        url_features = []
        for url in df["url"].fillna(""):
            feats = extract_url_features_extended(url)
            url_features.append(extended_features_to_array(feats))
        X_url = csr_matrix(np.array(url_features, dtype=np.float32))
        matrices.append(X_url)

    if not matrices:
        raise ValueError("At least one feature type must be enabled.")

    X_combined = hstack(matrices) if len(matrices) > 1 else matrices[0]
    return X_combined, tfidf_vectorizer
