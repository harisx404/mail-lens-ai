"""
PhishGuard AI — Test Suite

Tests for:
  - Text preprocessing
  - Security feature extraction
  - Data pipeline validation
  - Model prediction shape/labels
  - Edge cases
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# Text Preprocessing Tests
# ============================================
class TestTextPreprocessor:
    """Tests for NLP preprocessing pipeline."""

    def setup_method(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        self.preprocessor = TextPreprocessor(
            remove_stopwords=False, lemmatize=True
        )

    def test_empty_text(self):
        assert self.preprocessor.preprocess("") == ""

    def test_none_text(self):
        assert self.preprocessor.preprocess(None) == ""

    def test_html_removal(self):
        result = self.preprocessor.preprocess("<b>Hello</b> <i>World</i>")
        assert "<b>" not in result
        assert "<i>" not in result
        assert "hello" in result

    def test_url_normalization(self):
        text = "Visit https://example.com/login for details"
        result = self.preprocessor.preprocess(text)
        assert "urlplaceholder" in result
        assert "https" not in result

    def test_email_normalization(self):
        text = "Contact admin@company.com for help"
        result = self.preprocessor.preprocess(text)
        assert "emailplaceholder" in result
        assert "@" not in result

    def test_unicode_handling(self):
        text = "Héllo Wörld café"
        result = self.preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_special_characters(self):
        text = "Price: $100!! @#$% *** —— $$"
        result = self.preprocessor.preprocess(text)
        assert "$" not in result
        assert "#" not in result

    def test_whitespace_normalization(self):
        text = "Hello    World\n\n\tTest"
        result = self.preprocessor.preprocess(text)
        assert "  " not in result

    def test_lowercasing(self):
        text = "URGENT Account Verification REQUIRED"
        result = self.preprocessor.preprocess(text)
        assert result == result.lower()

    def test_very_short_text(self):
        result = self.preprocessor.preprocess("hi")
        assert isinstance(result, str)

    def test_url_extraction(self):
        text = "Visit https://example.com and http://test.org"
        urls = self.preprocessor.extract_urls(text)
        assert len(urls) == 2

    def test_email_extraction(self):
        text = "Send to user@domain.com and admin@test.org"
        emails = self.preprocessor.extract_emails(text)
        assert len(emails) == 2


# ============================================
# Security Feature Tests
# ============================================
class TestSecurityFeatures:
    """Tests for cybersecurity feature extraction."""

    def setup_method(self):
        from src.features.security_features import SecurityFeatureExtractor
        self.extractor = SecurityFeatureExtractor()

    def test_empty_text(self):
        result = self.extractor.extract("")
        assert result.urgency_score == 0.0
        assert result.total_indicators == 0

    def test_urgency_detection(self):
        text = "URGENT: Your account will be suspended immediately. Act now!"
        result = self.extractor.extract(text)
        assert result.urgency_count > 0
        assert result.urgency_score > 0.0

    def test_credential_detection(self):
        text = "Please verify your password and update your account credentials."
        result = self.extractor.extract(text)
        assert result.credential_count > 0
        assert result.credential_score > 0.0

    def test_threat_detection(self):
        text = "Unauthorized access detected. Your account has been compromised."
        result = self.extractor.extract(text)
        assert result.threat_count > 0

    def test_reward_detection(self):
        text = "Congratulations! You have won a $1000 prize. Claim your reward!"
        result = self.extractor.extract(text)
        assert result.reward_count > 0

    def test_url_features(self):
        text = "Click here: http://192.168.1.1/login and https://bit.ly/abc123"
        result = self.extractor.extract(text)
        assert result.url_count >= 2
        assert result.has_ip_url == 1
        assert result.has_url_shortener == 1

    def test_executable_mention(self):
        text = "Please download the attached file update.exe"
        result = self.extractor.extract(text)
        assert result.has_executable_mention == 1

    def test_legitimate_email_low_indicators(self):
        text = (
            "Hi John, just wanted to follow up on our meeting yesterday. "
            "The quarterly report is ready for review. Let me know if you "
            "have any questions. Best regards, Sarah."
        )
        result = self.extractor.extract(text)
        assert result.total_indicators <= 1  # May detect minor things

    def test_phishing_email_high_indicators(self):
        text = (
            "Dear customer, your account has been compromised. "
            "Verify your password immediately at http://192.168.1.1/verify "
            "or your account will be suspended within 24 hours. "
            "Click here to update your credentials now."
        )
        result = self.extractor.extract(text)
        assert result.total_indicators >= 3

    def test_to_dict(self):
        result = self.extractor.extract("Test text")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "urgency_score" in d
        assert "url_count" in d


# ============================================
# Data Pipeline Tests
# ============================================
class TestDataPipeline:
    """Tests for data loading and pipeline functions."""

    def test_label_mapping_dataset_a(self):
        import pandas as pd
        from src.data.pipeline import map_labels

        df = pd.DataFrame({
            "text": ["hello", "world"],
            "label_original": ["Safe Email", "Phishing Email"],
            "source": ["dataset_a", "dataset_a"],
        })
        result = map_labels(df, "dataset_a")
        assert list(result["label"]) == ["LEGITIMATE", "PHISHING"]

    def test_label_mapping_dataset_b(self):
        import pandas as pd
        from src.data.pipeline import map_labels

        df = pd.DataFrame({
            "text": ["a", "b", "c", "d", "e", "f"],
            "label_original": [
                "NOT-Malicious", "Phishing", "Baiting",
                "Pretexting", "Malware", "Scareware",
            ],
            "source": ["dataset_b"] * 6,
        })
        result = map_labels(df, "dataset_b")
        expected = ["LEGITIMATE", "PHISHING", "PHISHING", "PHISHING", "MALICIOUS", "MALICIOUS"]
        assert list(result["label"]) == expected

    def test_clean_data_removes_empty(self):
        import pandas as pd
        from src.data.pipeline import clean_data

        df = pd.DataFrame({
            "text": ["valid text", "", "   ", "nan", "another valid"],
            "label": ["PHISHING", "PHISHING", "LEGITIMATE", "LEGITIMATE", "MALICIOUS"],
        })
        result = clean_data(df)
        assert len(result) == 2  # Only valid rows remain

    def test_deduplication(self):
        import pandas as pd
        from src.data.pipeline import deduplicate

        df = pd.DataFrame({
            "text": ["same text", "same text", "different text"],
            "label": ["PHISHING", "LEGITIMATE", "MALICIOUS"],
        })
        result = deduplicate(df)
        assert len(result) == 2


# ============================================
# Configuration Tests
# ============================================
class TestConfiguration:
    """Tests for project configuration."""

    def test_classes_defined(self):
        from src.utils.config import CLASSES, CLASS_LABELS
        assert len(CLASSES) == 3
        assert "LEGITIMATE" in CLASSES
        assert "PHISHING" in CLASSES
        assert "MALICIOUS" in CLASSES
        assert len(CLASS_LABELS) == 3

    def test_keyword_lists_not_empty(self):
        from src.utils.config import (
            URGENCY_KEYWORDS, CREDENTIAL_KEYWORDS,
            THREAT_KEYWORDS, REWARD_KEYWORDS,
        )
        assert len(URGENCY_KEYWORDS) > 0
        assert len(CREDENTIAL_KEYWORDS) > 0
        assert len(THREAT_KEYWORDS) > 0
        assert len(REWARD_KEYWORDS) > 0

    def test_risk_levels(self):
        from src.utils.config import RISK_LEVELS
        assert "LOW" in RISK_LEVELS
        assert "MEDIUM" in RISK_LEVELS
        assert "HIGH" in RISK_LEVELS
        assert "CRITICAL" in RISK_LEVELS


# ============================================
# Inference Edge Case Tests
# ============================================
class TestInferenceEdgeCases:
    """Edge case tests for inference pipeline."""

    def test_analysis_result_to_dict(self):
        from src.inference.engine import AnalysisResult
        result = AnalysisResult(
            prediction="PHISHING",
            confidence=0.95,
            risk_level="HIGH",
            risk_score=0.85,
        )
        d = result.to_dict()
        assert d["prediction"] == "PHISHING"
        assert d["confidence"] == 0.95

    def test_analysis_result_defaults(self):
        from src.inference.engine import AnalysisResult
        result = AnalysisResult()
        assert result.prediction == ""
        assert result.confidence == 0.0
        assert result.detected_indicators == []


class TestTrainedModelInference:
    """Tests for the trained model and inference engine."""

    def setup_method(self):
        from src.inference.engine import PhishGuardInference
        self.engine = PhishGuardInference()

    def test_model_loaded(self):
        assert self.engine._is_loaded is True
        assert self.engine.model is not None

    def test_legitimate_prediction(self):
        text = "Hi Team, please find attached the meeting minutes and project timeline. Thanks, Sarah"
        result = self.engine.analyze(text)
        assert result.prediction == "LEGITIMATE"
        assert result.risk_level in ("LOW", "MEDIUM")
        assert result.confidence > 0.5

    def test_phishing_prediction(self):
        text = "URGENT: Your account has been suspended! Verify your password immediately at https://bank-login.xyz/reset or lose access within 24 hours."
        result = self.engine.analyze(text)
        assert result.prediction == "PHISHING"
        assert result.risk_level in ("HIGH", "CRITICAL")
        assert result.confidence > 0.5

    def test_malicious_prediction(self):
        text = "WARNING: Virus infection detected! Run patch.exe immediately to prevent scareware trojan infection. Attached file security_patch.exe"
        result = self.engine.analyze(text)
        assert result.prediction == "MALICIOUS"
        assert result.risk_level in ("HIGH", "CRITICAL")

    def test_empty_email_handling(self):
        result = self.engine.analyze("")
        assert result.prediction == "UNKNOWN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

