import os
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.utils import extract_url_features, features_to_array

class PhishingDetector:
    def __init__(self, model_path: str = "models/model.joblib"):
        """Initialize the detector, loading a saved model if available."""
        self.model_path = model_path
        self.model = None
        self._is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Try loading an existing model
        if os.path.exists(self.model_path):
            self.load(self.model_path)

    @property
    def is_trained(self) -> bool:
        """Returns whether the model is trained and ready."""
        return self._is_trained

    def _train_seed(self):
        """Trains the model on a hardcoded seed dataset of safe and phishing URLs."""
        # Minimum 10 safe URLs
        safe_urls = [
            "https://www.google.com",
            "https://github.com",
            "https://en.wikipedia.org",
            "https://www.python.org",
            "https://stackoverflow.com",
            "https://www.microsoft.com",
            "https://www.apple.com",
            "https://aws.amazon.com",
            "https://news.ycombinator.com",
            "https://www.netflix.com"
        ]
        
        # Minimum 10 phishing-pattern URLs
        phish_urls = [
            "http://192.168.1.100/login/secure",
            "http://paypal-update-account-info.com",
            "http://secure-login-attempt.com/verify-account",
            "http://appleid-confirm-now.org",
            "http://10.0.0.1/update/password",
            "https://banking-support-alert-suspended.net",
            "http://amazon-account-verify-credential.info",
            "http://google.com-login-verify.xyz",
            "http://secure-banking-verify-signin.com",
            "http://netflix-billing-update-now.com"
        ]
        
        # Build dataset
        X, y = [], []
        
        for url in safe_urls:
            features = extract_url_features(url)
            X.append(features_to_array(features))
            y.append(0)  # 0 = Safe
            
        for url in phish_urls:
            features = extract_url_features(url)
            X.append(features_to_array(features))
            y.append(1)  # 1 = Phishing
            
        print("Training Random Forest on seed data...")
        self.train(X, y)
        self.save(self.model_path)

    def train(self, X: list, y: list):
        """Train the Random Forest model on custom data."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, url: str) -> tuple[bool, float, dict]:
        """
        Predict whether a URL is phishing.
        Returns: (is_phishing: bool, confidence_score: float, features: dict)
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call _train_seed() or train().")
            
        # 1. Extract features
        features = extract_url_features(url)
        
        # 2. Convert to array for scikit-learn
        X_pred = [features_to_array(features)]
        
        # 3. Predict probability
        probs = self.model.predict_proba(X_pred)[0]
        
        # probs[1] is the probability of class 1 (phishing)
        confidence = float(probs[1])
        is_phishing = confidence > 0.5
        
        # Return probability of the predicted class as confidence
        final_confidence = confidence if is_phishing else float(probs[0])
        
        return is_phishing, final_confidence, features

    def save(self, path: str):
        """Persist model to disk using joblib."""
        if self.model is not None:
            joblib.dump(self.model, path)

    def load(self, path: str):
        """Restore model from disk."""
        self.model = joblib.load(path)
        self._is_trained = True
