"""
Mail-Lens AI — Text Preprocessor

NLP preprocessing pipeline for email text:
1. HTML tag removal
2. URL extraction and normalization
3. Email address normalization
4. Unicode normalization
5. Lowercasing
6. Whitespace normalization
7. Optional stopword removal (configurable — tested both ways)
8. Tokenization
9. Optional lemmatization (configurable — tested both ways)

Design Decisions:
- Stopword removal is OFF by default. Testing showed that phishing-relevant
  words like "your", "you", "please" can be informative for detection.
  The final decision is documented after experimentation.
- Lemmatization is ON by default to normalize verb forms while preserving
  semantic meaning.
- URLs and email addresses are replaced with placeholders rather than
  removed, because their PRESENCE is an important signal even after
  the URL features are extracted separately.
"""

import html
import logging
import re
import unicodedata
from typing import List, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ============================================
# Regex Patterns (compiled once for performance)
# ============================================

# URL pattern — matches http(s), ftp, and common URL forms
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"'\)]+|"
    r"ftp://[^\s<>\"'\)]+|"
    r"www\.[^\s<>\"'\)]+",
    re.IGNORECASE,
)

# Email address pattern
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Multiple whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")

# Non-alphanumeric (for aggressive cleaning)
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")

# HTML entities
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")


class TextPreprocessor:
    """
    Configurable text preprocessing pipeline for email content.

    Usage:
        preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=True)
        clean_text = preprocessor.preprocess("Raw email <b>content</b> here")
    """

    def __init__(
        self,
        remove_stopwords: bool = False,
        lemmatize: bool = True,
        lowercase: bool = True,
        remove_html: bool = True,
        normalize_urls: bool = True,
        normalize_emails: bool = True,
        min_token_length: int = 2,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.normalize_urls = normalize_urls
        self.normalize_emails = normalize_emails
        self.min_token_length = min_token_length

        # Lazy-load NLTK resources
        self._stopwords = None
        self._lemmatizer = None
        self._nltk_initialized = False

    def _init_nltk(self):
        """Initialize NLTK resources on first use."""
        if self._nltk_initialized:
            return

        import nltk

        for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        if self.remove_stopwords:
            from nltk.corpus import stopwords
            self._stopwords = set(stopwords.words("english"))

        if self.lemmatize:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()

        self._nltk_initialized = True

    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text (before normalization)."""
        return URL_PATTERN.findall(text)

    def extract_emails(self, text: str) -> List[str]:
        """Extract all email addresses from text."""
        return EMAIL_PATTERN.findall(text)

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        if "<" in text and ">" in text:
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)

        # Decode HTML entities
        text = html.unescape(text)

        return text


    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to ASCII-compatible form."""
        text = unicodedata.normalize("NFKD", text)
        # Remove combining characters (accents, etc.) but keep base chars
        text = "".join(
            c for c in text if not unicodedata.combining(c)
        )
        return text

    def preprocess(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to a text string.

        Pipeline order:
        1. HTML removal
        2. URL normalization (replace with placeholder)
        3. Email normalization (replace with placeholder)
        4. Unicode normalization
        5. Lowercasing
        6. Whitespace normalization
        7. Stopword removal (optional)
        8. Lemmatization (optional)
        9. Min token length filtering
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. HTML removal
        if self.remove_html:
            text = self.remove_html_tags(text)

        # 2. URL normalization
        if self.normalize_urls:
            text = URL_PATTERN.sub(" urlplaceholder ", text)

        # 3. Email normalization
        if self.normalize_emails:
            text = EMAIL_PATTERN.sub(" emailplaceholder ", text)

        # 4. Unicode normalization
        text = self.normalize_unicode(text)

        # 5. Lowercasing
        if self.lowercase:
            text = text.lower()

        # 6. Remove remaining special characters but keep spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # 7. Whitespace normalization
        text = WHITESPACE_PATTERN.sub(" ", text).strip()

        # Tokenize for stopword/lemma processing
        tokens = text.split()

        # 8. Stopword removal (optional)
        if self.remove_stopwords:
            self._init_nltk()
            tokens = [t for t in tokens if t not in self._stopwords]

        # 9. Lemmatization (optional)
        if self.lemmatize:
            self._init_nltk()
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        # 10. Min token length
        if self.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts."""
        return [self.preprocess(t) for t in texts]


# ============================================
# Module-level convenience function
# ============================================

_default_preprocessor = None


def preprocess_text(text: str) -> str:
    """Preprocess a single text using the default preprocessor."""
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = TextPreprocessor()
    return _default_preprocessor.preprocess(text)
