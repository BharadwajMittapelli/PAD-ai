import joblib
import random

class PhishingDetector:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        """Dummy training method."""
        print("Training model (dummy)...")
        self.model = {"type": "dummy", "trained": True}

    def predict(self, url: str, email_content: str = None):
        """Predicts if the input is a phishing attempt."""
        return self._heuristic_predict(url, email_content)

    def _heuristic_predict(self, url: str, email_content: str = None):
        """Simple heuristic for demonstration purposes."""
        url_lower = url.lower()
        
        if any(kw in url_lower for kw in ["phishing", "suspicious", "malware"]):
            return True, 0.95
        
        if any(domain in url_lower for domain in ["google.com", "example.com"]):
            return False, 0.05
            
        return False, 0.2

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
