"""
PhishGuard AI — Project Configuration

Central configuration for all project settings.
Uses environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# Path Configuration
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "models/saved")

# ============================================
# Random Seed (Reproducibility)
# ============================================
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ============================================
# Data Split Configuration
# ============================================
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", "0.1"))

# ============================================
# Class Labels
# ============================================
CLASSES = ["LEGITIMATE", "PHISHING", "MALICIOUS"]
CLASS_LABELS = {
    "LEGITIMATE": 0,
    "PHISHING": 1,
    "MALICIOUS": 2,
}
LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

# ============================================
# Dataset Sources
# ============================================
DATASET_A_SOURCE = "zefang-liu/phishing-email-dataset"
DATASET_A_URL = "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset"
DATASET_A_LICENSE = "LGPL-3.0"

DATASET_B_URL = "https://zenodo.org/records/15235123/files/phishing_nlp_dataset.xlsx?download=1"
DATASET_B_LICENSE = "CC BY 4.0"

# ============================================
# NLP Configuration
# ============================================
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams + Bigrams
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# ============================================
# Cybersecurity Feature Keywords
# ============================================
URGENCY_KEYWORDS = [
    "urgent", "immediately", "action required", "act now",
    "expire", "expires", "expiring", "deadline", "limited time",
    "hurry", "rush", "asap", "right away", "within 24 hours",
    "within 48 hours", "suspension", "suspended", "terminate",
    "terminated", "final warning", "last chance", "don't delay",
    "time sensitive", "critical", "important notice",
]

CREDENTIAL_KEYWORDS = [
    "password", "passwd", "login", "log in", "sign in",
    "username", "user name", "credential", "account",
    "verify your", "confirm your", "update your",
    "reset your", "ssn", "social security",
    "credit card", "card number", "bank account",
    "routing number", "pin", "security code", "cvv",
    "authentication", "2fa", "otp", "one-time",
]

THREAT_KEYWORDS = [
    "unauthorized", "suspicious activity", "breach",
    "compromised", "hacked", "violation", "fraud",
    "illegal", "locked", "disabled", "restricted",
    "unusual activity", "security alert", "warning",
    "threat", "blocked", "frozen",
]

REWARD_KEYWORDS = [
    "congratulations", "winner", "won", "prize",
    "reward", "gift", "free", "bonus", "lottery",
    "selected", "lucky", "claim your", "exclusive offer",
    "limited offer", "special promotion",
]

IMPERSONATION_KEYWORDS = [
    "dear customer", "dear user", "dear member",
    "valued customer", "account holder",
    "dear sir", "dear madam", "dear account",
    "helpdesk", "support team", "security team",
    "it department", "admin", "administrator",
]

# ============================================
# URL Feature Configuration
# ============================================
SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq",
    ".xyz", ".top", ".club", ".work", ".click",
    ".link", ".info", ".biz", ".win", ".review",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co",
    "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "short.io", "cutt.ly",
]

EXECUTABLE_EXTENSIONS = [
    ".exe", ".bat", ".cmd", ".com", ".msi",
    ".scr", ".pif", ".vbs", ".js", ".wsf",
    ".ps1", ".jar", ".py", ".sh",
]

SUSPICIOUS_EXTENSIONS = [
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".doc", ".docm", ".xls", ".xlsm",
    ".ppt", ".pptm", ".pdf", ".iso",
]

# ============================================
# Risk Scoring Configuration
# ============================================
RISK_LEVELS = {
    "LOW": {"min_score": 0.0, "max_score": 0.3, "color": "#28a745"},
    "MEDIUM": {"min_score": 0.3, "max_score": 0.6, "color": "#ffc107"},
    "HIGH": {"min_score": 0.6, "max_score": 0.85, "color": "#fd7e14"},
    "CRITICAL": {"min_score": 0.85, "max_score": 1.0, "color": "#dc3545"},
}

# ============================================
# Application Configuration
# ============================================
APP_HOST = os.getenv("APP_HOST", "localhost")
APP_PORT = int(os.getenv("APP_PORT", "8501"))
APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MAX_EMAIL_LENGTH = 50000  # characters
MAX_UPLOAD_SIZE_MB = 5
