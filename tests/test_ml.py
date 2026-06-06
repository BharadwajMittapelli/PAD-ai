import pytest
import os
from src.ml.model import PhishingDetector

@pytest.fixture(scope="module")
def detector():
    """Fixture to provide a trained PhishingDetector for tests."""
    det = PhishingDetector(model_path="models/test_model.joblib")
    if not det.is_trained:
        det._train_seed()
    yield det
    # Cleanup after tests
    if os.path.exists("models/test_model.joblib"):
        os.remove("models/test_model.joblib")

def test_detector_initialization(detector):
    """Test that the detector initializes and trains correctly."""
    assert detector.is_trained is True
    assert detector.model is not None

@pytest.mark.parametrize("url, expected_is_phishing", [
    ("https://www.google.com", False),
    ("https://www.facebook.com", False),
    ("https://netflix.com", False),
    ("http://secure-login-attempt.com/verify-account", True),
    ("http://10.0.0.1/update/password", True),
    ("http://paypal-confirm-now.org", True)
])
def test_predict(detector, url, expected_is_phishing):
    """Test that known safe and phishing URLs are correctly classified."""
    is_phish, conf, features = detector.predict(url)
    
    # Assert correctness
    assert is_phish == expected_is_phishing
    
    # Assert output types
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
    assert isinstance(features, dict)
    assert len(features) >= 20  # Should have ~25 features
