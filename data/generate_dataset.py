"""
data/generate_dataset.py
========================
Generates a synthetic phishing detection dataset with realistic URL and
email body patterns for benchmarking ML models.

Usage:
    python data/generate_dataset.py

Output:
    data/phishing_dataset.csv  (columns: url, email_body, label)
"""

import os
import csv
import random
import string

random.seed(42)

# ── Domain / URL building blocks ────────────────────────────────────────────

SAFE_DOMAINS = [
    "google.com", "github.com", "stackoverflow.com", "wikipedia.org",
    "python.org", "microsoft.com", "apple.com", "amazon.com",
    "netflix.com", "linkedin.com", "twitter.com", "reddit.com",
    "medium.com", "nytimes.com", "bbc.co.uk", "cnn.com",
    "dropbox.com", "spotify.com", "zoom.us", "slack.com",
    "adobe.com", "oracle.com", "ibm.com", "intel.com",
    "mozilla.org", "cloudflare.com", "digitalocean.com",
    "heroku.com", "gitlab.com", "bitbucket.org",
]

SAFE_PATHS = [
    "/", "/about", "/help", "/docs", "/blog", "/pricing",
    "/contact", "/features", "/products", "/solutions",
    "/careers", "/news", "/press", "/legal/terms",
    "/support/faq", "/developer/api", "/resources",
]

SAFE_EMAIL_TEMPLATES = [
    "Thank you for your purchase! Your order #{order} has been confirmed.",
    "Welcome to {brand}! We're glad to have you on board.",
    "Your monthly newsletter from {brand} is here. Check out what's new.",
    "Meeting reminder: You have a scheduled meeting at {time} today.",
    "Hi there! Just a quick update on your recent inquiry with {brand}.",
    "Your subscription to {brand} has been renewed successfully.",
    "Weekly digest: Here are the top stories from {brand} this week.",
    "Your invoice #{order} is available. Thank you for choosing {brand}.",
    "Congratulations! You've earned a new badge on {brand}.",
    "Reminder: Your event with {brand} is coming up on {date}.",
    "Thanks for contacting {brand} support. We received your ticket #{order}.",
    "Here's your receipt for the recent transaction with {brand}.",
    "Good news! Your {brand} account has been upgraded.",
    "Project update: The latest build for {brand} is ready for review.",
    "Your {brand} trial is active. Enjoy exploring all the features!",
]

PHISH_DOMAINS_PATTERNS = [
    "paypal-secure-login.com", "apple-id-verify.net",
    "microsoft-account-update.org", "amazon-billing-alert.info",
    "netflix-payment-issue.com", "google-security-alert.xyz",
    "banking-verify-signin.com", "account-confirm-update.net",
    "secure-ebay-login.org", "credential-recovery-support.com",
    "login-verify-account.info", "update-password-now.xyz",
    "suspended-account-alert.com", "unusual-activity-detected.net",
    "confirm-identity-secure.org", "paypal-support-team.info",
    "apple-billing-update.com", "amazon-order-confirm.xyz",
    "netflix-account-locked.net", "google-signin-verify.org",
]

PHISH_PATHS = [
    "/login", "/signin", "/verify", "/secure/update",
    "/account/confirm", "/password/reset", "/credential/recovery",
    "/billing/update", "/payment/verify", "/identity/confirm",
    "/webscr?cmd=login", "/dispatch/auth", "/authentication",
    "/recover/account", "/support/alert",
]

PHISH_EMAIL_TEMPLATES = [
    "URGENT: Your {brand} account has been suspended. Click here to verify your identity immediately.",
    "We detected unusual activity on your {brand} account. Confirm your password now to avoid suspension.",
    "Your {brand} payment could not be processed. Update your billing information immediately.",
    "ALERT: Someone tried to access your {brand} account from an unknown device. Secure it now.",
    "Your {brand} account will be closed in 24 hours unless you verify your credentials.",
    "IMPORTANT: Your {brand} password has expired. Reset it immediately to maintain access.",
    "Dear Customer, we noticed suspicious login attempts on your {brand} account. Act now!",
    "WARNING: Your {brand} order #{order} cannot be delivered. Confirm your shipping details.",
    "Your {brand} subscription has been cancelled due to payment failure. Update your card now.",
    "Security Notice: Unusual sign-in activity detected. Verify your {brand} account here.",
    "Action Required: Your {brand} account has been temporarily limited. Restore access now.",
    "FINAL WARNING: Verify your {brand} account within 12 hours or it will be permanently deleted.",
    "Congratulations! You won a {brand} gift card worth $500. Click to claim your reward now.",
    "Your {brand} refund of $127.99 is pending. Confirm your bank details to receive it.",
    "IT Department: Your email storage is full. Click here to upgrade and avoid losing messages.",
]

BRANDS = [
    "PayPal", "Apple", "Microsoft", "Amazon", "Netflix", "Google",
    "eBay", "Chase Bank", "Wells Fargo", "Bank of America",
    "Spotify", "Dropbox", "LinkedIn", "Facebook", "Instagram",
]

IP_OCTETS = lambda: f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _random_string(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _fill_template(template: str) -> str:
    brand = random.choice(BRANDS)
    order = random.randint(100000, 999999)
    time = f"{random.randint(9,17)}:{random.choice(['00','15','30','45'])}"
    date = f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    return template.format(brand=brand, order=order, time=time, date=date)


def generate_safe_sample() -> dict:
    """Generate a single safe URL + email sample."""
    domain = random.choice(SAFE_DOMAINS)
    path = random.choice(SAFE_PATHS)
    protocol = random.choice(["https://", "https://www.", "https://"])
    url = f"{protocol}{domain}{path}"

    # Sometimes add query params
    if random.random() < 0.2:
        url += f"?ref={_random_string(5)}"

    email_body = _fill_template(random.choice(SAFE_EMAIL_TEMPLATES))
    return {"url": url, "email_body": email_body, "label": 0}


def generate_phish_sample() -> dict:
    """Generate a single phishing URL + email sample."""
    strategy = random.choice(["keyword_domain", "ip_based", "typosquat", "long_subdomain"])

    if strategy == "keyword_domain":
        domain = random.choice(PHISH_DOMAINS_PATTERNS)
        path = random.choice(PHISH_PATHS)
        protocol = random.choice(["http://", "https://"])
        url = f"{protocol}{domain}{path}"

    elif strategy == "ip_based":
        ip = IP_OCTETS()
        path = random.choice(PHISH_PATHS)
        protocol = "http://"
        url = f"{protocol}{ip}{path}"

    elif strategy == "typosquat":
        legit = random.choice(["paypal", "apple", "google", "amazon", "microsoft"])
        typo = legit[:3] + _random_string(3) + legit[3:]
        tld = random.choice([".com", ".net", ".org", ".xyz", ".info"])
        path = random.choice(PHISH_PATHS)
        url = f"http://{typo}{tld}{path}"

    elif strategy == "long_subdomain":
        base = random.choice(["com", "net", "org"])
        legit = random.choice(["paypal", "apple", "google", "amazon"])
        sub = f"{legit}.com.{_random_string(5)}.{_random_string(6)}"
        tld = random.choice([".xyz", ".info", ".tk"])
        path = random.choice(PHISH_PATHS)
        url = f"http://{sub}{tld}{path}"

    # Sometimes add @ redirect trick
    if random.random() < 0.15:
        url = url.replace("://", f"://{random.choice(BRANDS).lower()}@", 1)

    email_body = _fill_template(random.choice(PHISH_EMAIL_TEMPLATES))
    return {"url": url, "email_body": email_body, "label": 1}


def generate_dataset(n_safe: int = 1000, n_phish: int = 1000, output_path: str = None):
    """Generate and save the full dataset."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phishing_dataset.csv")

    samples = []
    for _ in range(n_safe):
        samples.append(generate_safe_sample())
    for _ in range(n_phish):
        samples.append(generate_phish_sample())

    random.shuffle(samples)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "email_body", "label"])
        writer.writeheader()
        writer.writerows(samples)

    print(f"[OK] Generated {len(samples)} samples -> {output_path}")
    print(f"     Safe: {n_safe} | Phishing: {n_phish}")
    return output_path


if __name__ == "__main__":
    generate_dataset()
