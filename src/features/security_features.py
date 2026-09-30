"""
Mail-Lens AI — Cybersecurity Feature Extractor

Extracts security-specific features from email text for the ML pipeline.

Feature Groups:
1. Linguistic Security Indicators (urgency, credential requests, threats, etc.)
2. URL Features (count, HTTP vs HTTPS, suspicious patterns)
3. Attachment Indicators (mentioned executable extensions, archive keywords)

IMPORTANT: All analysis is STATIC — no URLs are visited, no files are executed.

These features are designed to complement the NLP/TF-IDF features by
capturing domain-specific security signals that bag-of-words models
might not emphasize strongly enough on their own.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List
from urllib.parse import urlparse

from src.utils.config import (
    CREDENTIAL_KEYWORDS,
    EXECUTABLE_EXTENSIONS,
    IMPERSONATION_KEYWORDS,
    KNOWN_BRANDS,
    REWARD_KEYWORDS,
    SIMULATION_KEYWORDS,
    SUSPICIOUS_EXTENSIONS,
    SUSPICIOUS_TLDS,
    THREAT_KEYWORDS,
    URGENCY_KEYWORDS,
    URL_SHORTENERS,
)

logger = logging.getLogger(__name__)

# ============================================
# Compiled Patterns
# ============================================
URL_EXTRACT_PATTERN = re.compile(
    r"https?://[^\s<>\"'\)]+|"
    r"ftp://[^\s<>\"'\)]+|"
    r"www\.[^\s<>\"'\)]+",
    re.IGNORECASE,
)

IP_URL_PATTERN = re.compile(
    r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    re.IGNORECASE,
)


@dataclass
class SecurityFeatures:
    """Container for all extracted security features."""

    # Linguistic indicators (normalized scores 0-1 based on keyword density)
    urgency_score: float = 0.0
    credential_score: float = 0.0
    threat_score: float = 0.0
    reward_score: float = 0.0
    impersonation_score: float = 0.0

    # Raw counts for indicators
    urgency_count: int = 0
    credential_count: int = 0
    threat_count: int = 0
    reward_count: int = 0
    impersonation_count: int = 0

    # URL features
    url_count: int = 0
    has_http_url: int = 0       # HTTP without S
    has_https_url: int = 0
    has_ip_url: int = 0         # IP-address-based URL
    has_url_shortener: int = 0
    suspicious_tld_count: int = 0
    avg_url_length: float = 0.0
    max_url_length: int = 0

    # Attachment indicators
    has_executable_mention: int = 0
    has_archive_mention: int = 0
    attachment_keyword_count: int = 0

    # Brand impersonation / typosquatting
    brand_similarity_score: float = 0.0
    is_typosquat: int = 0

    # Simulation cloaking
    simulation_cloak_score: float = 0.0

    # Combined indicator count
    total_indicators: int = 0

    # Detected indicators (human-readable list)
    detected_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for ML feature vector."""
        return {
            "urgency_score": self.urgency_score,
            "credential_score": self.credential_score,
            "threat_score": self.threat_score,
            "reward_score": self.reward_score,
            "impersonation_score": self.impersonation_score,
            "url_count": self.url_count,
            "has_http_url": self.has_http_url,
            "has_https_url": self.has_https_url,
            "has_ip_url": self.has_ip_url,
            "has_url_shortener": self.has_url_shortener,
            "suspicious_tld_count": self.suspicious_tld_count,
            "avg_url_length": self.avg_url_length,
            "max_url_length": self.max_url_length,
            "has_executable_mention": self.has_executable_mention,
            "has_archive_mention": self.has_archive_mention,
            "attachment_keyword_count": self.attachment_keyword_count,
            "brand_similarity_score": self.brand_similarity_score,
            "is_typosquat": self.is_typosquat,
            "simulation_cloak_score": self.simulation_cloak_score,
            "total_indicators": self.total_indicators,
        }


class SecurityFeatureExtractor:
    """
    Extracts cybersecurity-specific features from raw email text.

    Usage:
        extractor = SecurityFeatureExtractor()
        features = extractor.extract("Dear customer, verify your account immediately...")
        feature_dict = features.to_dict()
    """

    def __init__(self):
        # Pre-compile keyword patterns for faster matching
        self._urgency_patterns = self._compile_keywords(URGENCY_KEYWORDS)
        self._credential_patterns = self._compile_keywords(CREDENTIAL_KEYWORDS)
        self._threat_patterns = self._compile_keywords(THREAT_KEYWORDS)
        self._reward_patterns = self._compile_keywords(REWARD_KEYWORDS)
        self._impersonation_patterns = self._compile_keywords(IMPERSONATION_KEYWORDS)
        self._simulation_patterns = self._compile_keywords(SIMULATION_KEYWORDS)

    @staticmethod
    def _compile_keywords(keywords: List[str]) -> List[re.Pattern]:
        """Compile keyword list into regex patterns for word-boundary matching."""
        return [
            re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for kw in keywords
        ]

    @staticmethod
    def _count_matches(text: str, patterns: List[re.Pattern]) -> int:
        """Count total keyword matches in text."""
        count = 0
        for pattern in patterns:
            count += len(pattern.findall(text))
        return count

    @staticmethod
    def _get_matched_keywords(text: str, patterns: List[re.Pattern], keywords: List[str]) -> List[str]:
        """Return list of matched keyword names."""
        matched = []
        for pattern, keyword in zip(patterns, keywords):
            if pattern.search(text):
                matched.append(keyword)
        return matched

    def _compute_score(self, count: int, max_expected: int = 5) -> float:
        """Normalize count to 0-1 score with diminishing returns."""
        if count == 0:
            return 0.0
        return min(1.0, count / max_expected)

    def extract_url_features(self, text: str) -> dict:
        """Extract URL-related features from raw text (STATIC analysis only)."""
        urls = URL_EXTRACT_PATTERN.findall(text)

        features = {
            "url_count": len(urls),
            "has_http_url": 0,
            "has_https_url": 0,
            "has_ip_url": 0,
            "has_url_shortener": 0,
            "suspicious_tld_count": 0,
            "avg_url_length": 0.0,
            "max_url_length": 0,
        }

        if not urls:
            return features

        features["avg_url_length"] = sum(len(u) for u in urls) / len(urls)
        features["max_url_length"] = max(len(u) for u in urls)

        for url in urls:
            url_lower = url.lower()

            # HTTP vs HTTPS
            if url_lower.startswith("http://"):
                features["has_http_url"] = 1
            elif url_lower.startswith("https://"):
                features["has_https_url"] = 1

            # IP-based URL
            if IP_URL_PATTERN.search(url):
                features["has_ip_url"] = 1

            # URL shortener
            for shortener in URL_SHORTENERS:
                if shortener in url_lower:
                    features["has_url_shortener"] = 1
                    break

            # Suspicious TLD
            try:
                parsed = urlparse(url if "://" in url else f"http://{url}")
                hostname = parsed.hostname or ""
                for tld in SUSPICIOUS_TLDS:
                    if hostname.endswith(tld):
                        features["suspicious_tld_count"] += 1
                        break
            except (ValueError, AttributeError) as e:
                logger.debug(f"URL parsing failed for TLD analysis on '{url}': {e}")

        return features

    def extract_attachment_features(self, text: str) -> dict:
        """Extract attachment-related features from text mentions."""
        text_lower = text.lower()

        has_executable = 0
        has_archive = 0
        attachment_count = 0

        for ext in EXECUTABLE_EXTENSIONS:
            if ext in text_lower:
                has_executable = 1
                attachment_count += 1

        for ext in SUSPICIOUS_EXTENSIONS:
            if ext in text_lower:
                has_archive = 1
                attachment_count += 1

        return {
            "has_executable_mention": has_executable,
            "has_archive_mention": has_archive,
            "attachment_keyword_count": attachment_count,
        }

    def extract(self, text: str) -> SecurityFeatures:
        """
        Extract all security features from raw email text.

        Args:
            text: Raw (un-preprocessed) email text

        Returns:
            SecurityFeatures dataclass with all extracted features
        """
        if not text or not isinstance(text, str):
            return SecurityFeatures()

        features = SecurityFeatures()
        indicators = []

        # --- Linguistic indicators ---
        features.urgency_count = self._count_matches(text, self._urgency_patterns)
        features.credential_count = self._count_matches(text, self._credential_patterns)
        features.threat_count = self._count_matches(text, self._threat_patterns)
        features.reward_count = self._count_matches(text, self._reward_patterns)
        features.impersonation_count = self._count_matches(text, self._impersonation_patterns)

        features.urgency_score = self._compute_score(features.urgency_count)
        features.credential_score = self._compute_score(features.credential_count)
        features.threat_score = self._compute_score(features.threat_count)
        features.reward_score = self._compute_score(features.reward_count)
        features.impersonation_score = self._compute_score(features.impersonation_count)

        if features.urgency_count > 0:
            indicators.append("Urgency language detected")
        if features.credential_count > 0:
            indicators.append("Credential/account request language")
        if features.threat_count > 0:
            indicators.append("Threat/fear language detected")
        if features.reward_count > 0:
            indicators.append("Reward/prize language detected")
        if features.impersonation_count > 0:
            indicators.append("Impersonation indicators detected")

        # --- URL features ---
        url_feats = self.extract_url_features(text)
        features.url_count = url_feats["url_count"]
        features.has_http_url = url_feats["has_http_url"]
        features.has_https_url = url_feats["has_https_url"]
        features.has_ip_url = url_feats["has_ip_url"]
        features.has_url_shortener = url_feats["has_url_shortener"]
        features.suspicious_tld_count = url_feats["suspicious_tld_count"]
        features.avg_url_length = url_feats["avg_url_length"]
        features.max_url_length = url_feats["max_url_length"]

        if features.url_count > 0:
            indicators.append(f"{features.url_count} URL(s) found")
        if features.has_http_url:
            indicators.append("HTTP (non-encrypted) URL detected")
        if features.has_ip_url:
            indicators.append("IP-address-based URL detected")
        if features.has_url_shortener:
            indicators.append("URL shortener detected")
        if features.suspicious_tld_count > 0:
            indicators.append("Suspicious TLD detected")

        # --- Attachment features ---
        attach_feats = self.extract_attachment_features(text)
        features.has_executable_mention = attach_feats["has_executable_mention"]
        features.has_archive_mention = attach_feats["has_archive_mention"]
        features.attachment_keyword_count = attach_feats["attachment_keyword_count"]

        if features.has_executable_mention:
            indicators.append("Executable file type mentioned")
        if features.has_archive_mention:
            indicators.append("Archive/document file type mentioned")

        # --- Typosquatting / Brand impersonation ---
        brand_feats = self._extract_brand_impersonation(text)
        features.brand_similarity_score = brand_feats["brand_similarity_score"]
        features.is_typosquat = brand_feats["is_typosquat"]

        if features.is_typosquat:
            indicators.append(
                f"Possible brand impersonation detected (similarity: {features.brand_similarity_score:.0%})"
            )

        # --- Simulation cloaking ---
        sim_count = self._count_matches(text, self._simulation_patterns)
        features.simulation_cloak_score = self._compute_score(sim_count, max_expected=3)
        if sim_count > 0 and (features.credential_count > 0 or features.threat_count > 0):
            indicators.append("Simulation/training language with threat indicators (possible cloaking)")

        # --- Total indicators ---
        features.total_indicators = len(indicators)
        features.detected_indicators = indicators

        return features

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return SecurityFeatureExtractor._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def _extract_brand_impersonation(self, text: str) -> dict:
        """Detect typosquatting and brand impersonation in domains."""
        result = {"brand_similarity_score": 0.0, "is_typosquat": 0}

        # Extract all domain-like patterns from text
        domain_pattern = re.compile(
            r'[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}',
            re.IGNORECASE,
        )
        domains_in_text = domain_pattern.findall(text.lower())

        max_similarity = 0.0
        for domain in domains_in_text:
            for brand in KNOWN_BRANDS:
                if domain == brand:
                    continue  # Exact match = legitimate
                dist = self._levenshtein_distance(domain, brand)
                max_len = max(len(domain), len(brand))
                if max_len == 0:
                    continue
                similarity = 1.0 - (dist / max_len)
                if similarity > max_similarity:
                    max_similarity = similarity
                # Typosquat: very close but not exact (edit distance 1-2)
                if 0 < dist <= 2 and len(domain) >= 5:
                    result["is_typosquat"] = 1

        result["brand_similarity_score"] = round(max_similarity, 4)
        return result
