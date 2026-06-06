"""
src/utils.py
============
PAD.ai — URL Feature Extraction Utilities

Extracts 25 numerical features from any URL string for ML classification.
All features are designed to capture phishing-indicator patterns documented
in academic literature and real-world threat intelligence.

Usage:
    from src.utils import extract_url_features, features_to_array

    features = extract_url_features("http://paypal-login-verify.com/account")
    vector   = features_to_array(features)  # pass to ML model
"""

import re
import math
from urllib.parse import urlparse, parse_qs
from typing import Dict, List

# ── Optional dependency: tldextract for precise subdomain parsing ────────────
try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False

# ── Phishing keyword list ────────────────────────────────────────────────────
SUSPICIOUS_KEYWORDS: List[str] = [
    "login", "signin", "verify", "secure", "update", "confirm",
    "account", "banking", "password", "credential", "paypal",
    "ebay", "amazon", "apple", "microsoft", "google", "support",
    "alert", "suspended", "unusual", "activity", "click", "here",
    "webscr", "cmd", "dispatch", "authentication", "recover",
]

# ── Canonical feature order (must stay fixed — matches ML training) ──────────
_FEATURE_ORDER: List[str] = [
    "url_length", "is_https", "domain_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_at_symbols", "has_at_symbol",
    "has_double_slash", "has_ip_address", "subdomain_count", "tld_in_subdomain",
    "path_length", "num_query_params", "num_percent", "num_equals",
    "num_ampersand", "num_question_mark", "num_hash", "suspicious_keyword_count",
    "has_suspicious_keyword", "digit_ratio", "url_entropy",
    "has_prefix_suffix_hyphen",
]

_FEATURE_DISPLAY_NAMES: List[str] = [
    "URL Length", "Is HTTPS", "Domain Length", "Dot Count", "Hyphen Count",
    "Underscore Count", "Slash Count", "At Symbol Count", "Has @ Symbol",
    "Has Double Slash", "Has IP Address", "Subdomain Count", "TLD in Subdomain",
    "Path Length", "Query Param Count", "Percent Count", "Equals Count",
    "Ampersand Count", "Question Mark Count", "Hash Count",
    "Suspicious Keyword Count", "Has Suspicious Keyword",
    "Digit Ratio", "URL Entropy", "Prefix/Suffix Hyphen",
]


# ── Public API ───────────────────────────────────────────────────────────────

def extract_url_features(url: str) -> Dict[str, float]:
    """
    Extract 25 phishing-indicator features from a URL string.

    Parameters
    ----------
    url : str
        Raw URL (with or without protocol prefix).

    Returns
    -------
    dict
        Feature name → numerical value mapping.
        Binary flags are encoded as 0 (absent) or 1 (present).
        All values are JSON-serialisable floats/ints.

    Notes
    -----
    Never raises — malformed URLs fall back to safe defaults
    so the model always receives a complete 25-feature vector.
    """
    features: Dict[str, float] = {}

    # ── Normalise URL ────────────────────────────────────────────────────────
    raw_url = url.strip()
    normalised = raw_url if raw_url.startswith(("http://", "https://")) else "http://" + raw_url

    # ── Parse components ─────────────────────────────────────────────────────
    try:
        parsed   = urlparse(normalised)
        domain   = parsed.netloc
        path     = parsed.path
        query    = parsed.query
    except Exception:
        domain, path, query = "", "", ""

    # ── 1. Length features ───────────────────────────────────────────────────
    features["url_length"]    = len(raw_url)
    features["domain_length"] = len(domain)
    features["path_length"]   = len(path)

    # ── 2. Protocol ──────────────────────────────────────────────────────────
    features["is_https"] = 1 if raw_url.lower().startswith("https") else 0

    # ── 3. Character count features ──────────────────────────────────────────
    features["num_dots"]           = raw_url.count(".")
    features["num_hyphens"]        = raw_url.count("-")
    features["num_underscores"]    = raw_url.count("_")
    features["num_slashes"]        = raw_url.count("/")
    features["num_at_symbols"]     = raw_url.count("@")
    features["has_at_symbol"]      = 1 if "@" in raw_url else 0
    features["num_percent"]        = raw_url.count("%")
    features["num_equals"]         = raw_url.count("=")
    features["num_ampersand"]      = raw_url.count("&")
    features["num_question_mark"]  = raw_url.count("?")
    features["num_hash"]           = raw_url.count("#")

    # ── 4. Redirect tricks ───────────────────────────────────────────────────
    # '//' outside of the protocol (e.g. http://legit.com//evil.com)
    after_protocol = raw_url.split("://", 1)[-1] if "://" in raw_url else raw_url
    features["has_double_slash"] = 1 if "//" in after_protocol else 0

    # ── 5. IP address in domain ──────────────────────────────────────────────
    ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    features["has_ip_address"] = 1 if re.search(ip_pattern, domain) else 0

    # ── 6. Subdomain analysis ────────────────────────────────────────────────
    if _HAS_TLDEXTRACT:
        ext = tldextract.extract(raw_url)
        subdomain_parts          = ext.subdomain.split(".") if ext.subdomain else []
        features["subdomain_count"]    = len(subdomain_parts)
        features["tld_in_subdomain"]   = (
            1 if ext.suffix and ext.suffix in (ext.subdomain or "") else 0
        )
    else:
        # Fallback: rough heuristic (parts beyond root domain + TLD)
        parts                        = domain.replace("www.", "").split(".")
        features["subdomain_count"]  = max(0, len(parts) - 2)
        features["tld_in_subdomain"] = 0

    # ── 7. Domain prefix/suffix hyphen ───────────────────────────────────────
    clean_domain = domain.split(":")[0]  # strip port if present
    features["has_prefix_suffix_hyphen"] = (
        1 if clean_domain.startswith("-") or clean_domain.endswith("-") else 0
    )

    # ── 8. Query parameters ───────────────────────────────────────────────────
    try:
        features["num_query_params"] = len(parse_qs(query))
    except Exception:
        features["num_query_params"] = 0

    # ── 9. Suspicious keyword analysis ───────────────────────────────────────
    url_lower  = raw_url.lower()
    kw_count   = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    features["suspicious_keyword_count"] = kw_count
    features["has_suspicious_keyword"]   = 1 if kw_count > 0 else 0

    # ── 10. Statistical features ─────────────────────────────────────────────
    n = len(raw_url)
    digit_count           = sum(c.isdigit() for c in raw_url)
    features["digit_ratio"]   = round(digit_count / n, 4) if n > 0 else 0.0
    features["url_entropy"]   = _calculate_entropy(raw_url)

    return features


def features_to_array(features: Dict[str, float]) -> List[float]:
    """
    Convert a feature dict to an ordered list matching the ML model's
    expected input shape (25 values).

    Parameters
    ----------
    features : dict
        Output of ``extract_url_features()``.

    Returns
    -------
    list of float
        Values in canonical _FEATURE_ORDER.  Missing keys default to 0.
    """
    return [float(features.get(f, 0)) for f in _FEATURE_ORDER]


def get_feature_names() -> List[str]:
    """
    Return human-readable feature names in canonical order,
    suitable for chart axis labels or DataFrames.
    """
    return _FEATURE_DISPLAY_NAMES.copy()


def get_feature_keys() -> List[str]:
    """Return raw feature key names in canonical order."""
    return _FEATURE_ORDER.copy()


# ── Internal helpers ─────────────────────────────────────────────────────────

def _calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    High entropy (close to log2(charset_size)) indicates randomly generated
    or encoded content — a common obfuscation technique in phishing URLs.

    Parameters
    ----------
    text : str

    Returns
    -------
    float
        Entropy in bits, rounded to 4 decimal places.
        Returns 0.0 for empty strings.
    """
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    entropy = -sum((count / n) * math.log2(count / n) for count in freq.values())
    return round(entropy, 4)
