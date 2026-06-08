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

# ── AI-Generated phishing email templates ────────────────────────────────────
# These simulate LLM-generated phishing: more polished, uniform sentence length,
# consistent punctuation, fewer spelling errors, and predictable structure.
AI_PHISH_EMAIL_TEMPLATES = [
    "Dear valued customer, we have detected unauthorized access to your {brand} account. "
    "For your security, we require you to verify your identity. Please click the link below "
    "to confirm your account information and restore full access. Failure to act within "
    "24 hours may result in permanent account suspension. Thank you for your cooperation.",

    "We are writing to inform you that your {brand} account requires immediate attention. "
    "Our security systems have identified suspicious activity associated with your account. "
    "To protect your personal information, please verify your credentials through the secure "
    "link provided. Your prompt action is greatly appreciated.",

    "This is an automated notification from the {brand} Security Team. We have observed "
    "unusual login patterns on your account from an unrecognized device. To ensure the "
    "safety of your account, please complete the verification process. Click the secure "
    "link below to confirm your identity and update your security settings.",

    "Important notice regarding your {brand} subscription. Your recent payment of $49.99 "
    "could not be processed due to outdated billing information. Please update your payment "
    "details within 48 hours to avoid service interruption. Click below to access your "
    "account settings and resolve this issue promptly.",

    "Thank you for being a loyal {brand} customer. As part of our ongoing security "
    "enhancement program, we are requiring all users to re-verify their account credentials. "
    "This process helps us maintain the highest level of security for your personal data. "
    "Please follow the secure link to complete the verification process.",

    "We regret to inform you that your {brand} account has been temporarily restricted. "
    "This action was taken after our automated systems detected potential unauthorized "
    "access. To lift this restriction, please verify your account by clicking the link "
    "below. We apologize for any inconvenience this may cause.",

    "Our records indicate that your {brand} account password has not been updated in over "
    "90 days. For your protection, we strongly recommend updating your password immediately. "
    "Please click the secure link below to create a new password. This will help ensure "
    "the continued security of your account information.",

    "You have received a refund of $127.50 from {brand}. To process this refund to your "
    "original payment method, we need you to confirm your banking details. Please click "
    "the secure link below to verify your information. The refund will be processed within "
    "3 to 5 business days after verification is complete.",

    "As part of our annual security audit, {brand} is requesting all users to update their "
    "account security settings. This includes verifying your email address, phone number, "
    "and payment information. Please complete this process within 72 hours to maintain "
    "uninterrupted access to your account and all associated services.",

    "We noticed that your {brand} account was accessed from a new location. If this was not "
    "you, please secure your account immediately by clicking the verification link below. "
    "Our security team is available around the clock to assist you with any concerns. "
    "Your account security is our top priority at {brand}.",

    "Congratulations! You have been selected for an exclusive {brand} reward program. As a "
    "valued customer, you are eligible to receive a complimentary gift card worth $250. To "
    "claim your reward, please verify your identity through the secure link provided below. "
    "This offer is valid for the next 48 hours only.",

    "The {brand} IT department has detected that your email storage is approaching its "
    "maximum capacity. To prevent loss of important messages and attachments, please click "
    "the link below to upgrade your storage plan. This upgrade is free of charge for all "
    "existing customers and will be applied to your account automatically.",
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
    return {"url": url, "email_body": email_body, "label": 0, "is_ai_generated": 0}


def generate_phish_sample() -> dict:
    """Generate a single human-written phishing URL + email sample."""
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
    return {"url": url, "email_body": email_body, "label": 1, "is_ai_generated": 0}


def generate_ai_phish_sample() -> dict:
    """
    Generate a single AI-generated phishing URL + email sample.

    AI-generated samples differ from human-written phishing in their
    stylometric profile: more uniform sentence lengths, polished grammar,
    and predictable phrasing patterns.
    """
    # AI phishing still uses the same URL strategies
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
        typo = legit[:3] + _random_string(2) + legit[3:]
        tld = random.choice([".com", ".net", ".org", ".xyz"])
        path = random.choice(PHISH_PATHS)
        url = f"https://{typo}{tld}{path}"

    elif strategy == "long_subdomain":
        legit = random.choice(["paypal", "apple", "google", "amazon"])
        sub = f"secure.{legit}.com.{_random_string(4)}"
        tld = random.choice([".xyz", ".info", ".tk"])
        path = random.choice(PHISH_PATHS)
        url = f"https://{sub}{tld}{path}"

    # AI phishing uses the AI-specific templates
    email_body = _fill_template(random.choice(AI_PHISH_EMAIL_TEMPLATES))
    return {"url": url, "email_body": email_body, "label": 1, "is_ai_generated": 1}


def generate_dataset(
    n_safe: int = 1000,
    n_phish: int = 1000,
    n_ai_phish: int = 1000,
    output_path: str = None,
):
    """
    Generate and save the full dataset.

    Parameters
    ----------
    n_safe : int
        Number of safe (legitimate) samples.
    n_phish : int
        Number of human-written phishing samples.
    n_ai_phish : int
        Number of AI-generated phishing samples.
    output_path : str, optional
        Output CSV path.  Defaults to ``data/phishing_dataset.csv``.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phishing_dataset.csv")

    samples = []
    for _ in range(n_safe):
        samples.append(generate_safe_sample())
    for _ in range(n_phish):
        samples.append(generate_phish_sample())
    for _ in range(n_ai_phish):
        samples.append(generate_ai_phish_sample())

    random.shuffle(samples)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["url", "email_body", "label", "is_ai_generated"]
        )
        writer.writeheader()
        writer.writerows(samples)

    total = len(samples)
    print(f"[OK] Generated {total} samples -> {output_path}")
    print(f"     Safe: {n_safe} | Human Phishing: {n_phish} | AI Phishing: {n_ai_phish}")
    return output_path


if __name__ == "__main__":
    generate_dataset()

