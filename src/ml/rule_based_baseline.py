import re
from typing import Dict, Tuple

class RuleBasedDetector:
    """
    A simple baseline detector using regex patterns to identify
    phishing traits in emails and URLs.
    """
    def __init__(self):
        # Compiled regex patterns for speed
        self.urgency_words = re.compile(r'\b(urgent|immediate|suspend|verify|confirm|alert|action required)\b', re.IGNORECASE)
        self.url_shorteners = re.compile(r'(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|rebrand\.ly|cutt\.ly|short\.io|tiny\.cc)', re.IGNORECASE)
        self.ip_address = re.compile(r'https?://[0-9]{1,3}(?:\.[0-9]{1,3}){3}')
        self.exe_extensions = re.compile(r'\.(exe|bat|cmd|scr|pif|msi|js|vbs|php|asp|aspx|cgi|pl|py|rb)(\?|$)', re.IGNORECASE)
        self.suspicious_subdomains = re.compile(r'(login|secure|account|update|verify|billing)', re.IGNORECASE)

    def predict_combined(self, email_body: str, url: str) -> Dict:
        """
        Evaluate rules and return a prediction result matching the
        interface of PhishGuardPredictor.
        """
        score = 0.0
        triggered_rules = []
        
        email_body = str(email_body) if email_body else ""
        url = str(url) if url else ""

        # 1. Check for urgency words in email body
        if self.urgency_words.search(email_body):
            score += 0.3
            triggered_rules.append("urgency_keywords")

        # 2. Check for URL shorteners
        if self.url_shorteners.search(url):
            score += 0.4
            triggered_rules.append("url_shortener")

        # 3. Check for IP address in URL
        if self.ip_address.search(url):
            score += 0.6
            triggered_rules.append("ip_address_domain")

        # 4. Check for executable/script extensions
        if self.exe_extensions.search(url):
            score += 0.5
            triggered_rules.append("executable_extension")

        # 5. Suspicious keywords in subdomains/paths
        if self.suspicious_subdomains.search(url):
            score += 0.3
            triggered_rules.append("suspicious_url_keywords")

        # Cap score at 1.0
        final_score = min(score, 1.0)
        is_phishing = final_score >= 0.5

        return {
            "is_phishing": is_phishing,
            "confidence": final_score if is_phishing else 1.0 - final_score,
            "score": final_score,
            "triggered_rules": triggered_rules,
            "model_used": "rule_based_baseline"
        }
