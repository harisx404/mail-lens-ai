"""
PhishGuard AI — Integration and Regression Test Suite.

Tests end-to-end inference pipelines, adversarial evasion techniques,
typosquatting detection, simulation cloaking, and false positive prevention.
"""

import pytest
from src.inference.engine import PhishGuardInference
from src.features.security_features import SecurityFeatureExtractor


@pytest.fixture(scope="module")
def engine():
    """Shared inference engine for integration tests."""
    return PhishGuardInference("models/saved")


@pytest.fixture(scope="module")
def extractor():
    """Shared security feature extractor."""
    return SecurityFeatureExtractor()


class TestSecurityFeatureRegressions:
    """Regression tests for security feature extraction bugs."""

    def test_dot_com_domain_does_not_trigger_executable_indicator(self, extractor):
        """Regression test for Bug 3.1: .com domains must NOT trigger executable mention."""
        email_text = (
            "Please check out our website at https://company.com or email us at support@example.com."
        )
        features = extractor.extract(email_text)
        assert features.has_executable_mention == 0

    def test_real_executable_does_trigger_indicator(self, extractor):
        """Verify that genuine executables (.exe, .bat, .scr) ARE detected."""
        email_text = "Please download and run setup.exe to install the urgent update."
        features = extractor.extract(email_text)
        assert features.has_executable_mention == 1

    def test_brand_typosquatting_detection(self, extractor):
        """Verify Levenshtein distance detection for visually similar lookalike domains."""
        # rnicrosoft.com (rn -> m trick)
        features = extractor.extract("Action required: log in at http://rnicrosoft.com/login")
        assert features.is_typosquat == 1
        assert features.brand_similarity_score >= 0.80

        # paypa1.com (1 -> l trick)
        features_paypal = extractor.extract("Verify payment at https://paypa1.com/checkout")
        assert features_paypal.is_typosquat == 1
        assert features_paypal.brand_similarity_score >= 0.80

    def test_exact_legitimate_domain_not_flagged_as_typosquat(self, extractor):
        """Verify that exact legitimate brand domains are NOT flagged as typosquats."""
        legit_email = "Official notification from Microsoft: visit https://microsoft.com/support"
        features = extractor.extract(legit_email)
        assert features.is_typosquat == 0

    def test_simulation_cloaking_detection(self, extractor):
        """Verify detection of meta-disclaimers used to cloak phishing attacks."""
        cloaked_text = (
            "Microsoft Security Team — SIMULATION\n"
            "This is a cybersecurity training simulation. Identify warning signs."
        )
        features = extractor.extract(cloaked_text)
        assert features.simulation_cloak_score > 0.0


class TestEndToEndInferenceIntegration:
    """End-to-end integration tests for the full inference engine."""

    def test_simulation_cloaked_phishing_triggers_override(self, engine):
        """Regression test: User's reported simulation email must be flagged as high risk."""
        simulation_email = """
        Microsoft Security Team — SIMULATION

        We detected an unusual sign-in attempt on your account from a new device.

        Date: September 30, 2026
        Location: Unknown
        Device: Windows PC

        For this security exercise, review the message and identify the warning signs before taking any action.

        [Review Account Activity — support@rnicrosoft.com]

        If you did not initiate this activity, contact your organization's IT/security team through an independently verified channel.

        Microsoft Security Team
        This is a cybersecurity training simulation.
        """
        result = engine.analyze(simulation_email)

        # Risk must be elevated to at least HIGH due to security override
        assert result.risk_level in ("HIGH", "CRITICAL"), (
            f"Expected HIGH or CRITICAL, got {result.risk_level} (score: {result.risk_score})"
        )
        assert result.risk_score >= 0.60

        # Must flag brand impersonation and cloaking indicators
        indicator_texts = " ".join(result.detected_indicators)
        assert "impersonation" in indicator_texts.lower() or "rnicrosoft" in indicator_texts.lower()
        assert "simulation" in indicator_texts.lower() or "cloaking" in indicator_texts.lower()

    def test_corporate_newsletter_is_legitimate(self, engine):
        """End-to-end test: Standard internal communication receives LOW risk."""
        legit_text = """
        Hi everyone,

        Here is the agenda for our monthly engineering all-hands meeting this Thursday at 2:00 PM.
        We will cover Q3 project milestones, team updates, and open Q&A.

        Agenda:
        1. Welcome and team highlights (15 min)
        2. Architecture roadmap review (20 min)
        3. Open Q&A (15 min)

        Looking forward to seeing you all there.

        Best regards,
        Engineering Management Team
        """
        result = engine.analyze(legit_text)
        assert result.prediction == "LEGITIMATE"
        assert result.risk_level == "LOW"
        assert result.risk_score < 0.35

    def test_urgent_credential_harvesting_is_phishing(self, engine):
        """End-to-end test: Classic credential harvesting is classified as PHISHING."""
        phish_text = """
        FINAL NOTICE: Your Office 365 mailbox quota has been exceeded!

        Incoming emails will be blocked in 4 hours unless you verify your credentials.
        Click below immediately to keep your account active:
        https://login-microsoft-auth-verify.xyz/login.php

        Failure to update will result in permanent deletion of your mailbox.
        IT Helpdesk Support
        """
        result = engine.analyze(phish_text)
        assert result.prediction == "PHISHING"
        assert result.risk_level in ("HIGH", "CRITICAL")
        assert result.risk_score >= 0.60
        assert len(result.detected_indicators) >= 2

    def test_malware_delivery_is_flagged(self, engine):
        """End-to-end test: Malware attachment email is flagged with high risk."""
        malware_text = """
        Invoice INV-2026-8819 is overdue.
        Please review the attached invoice details and execute the payment verification tool:
        Attached: Invoice_Statement_Overdue.exe

        This security patch must be applied immediately to avoid credit hold.
        Accounts Payable
        """
        result = engine.analyze(malware_text)
        assert result.risk_level in ("HIGH", "CRITICAL")
        assert result.risk_score >= 0.60
        indicators_str = " ".join(result.detected_indicators).lower()
        assert "executable" in indicators_str or "malicious" in indicators_str

    def test_batch_sequential_consistency(self, engine):
        """Verify that sequential calls do not leak internal state across inferences."""
        text_safe = "Team lunch tomorrow at noon in the cafeteria. Bring your ideas!"
        text_danger = "URGENT: Click here to verify password now or account locked: https://evil.ru/steal"

        res1 = engine.analyze(text_safe)
        res2 = engine.analyze(text_danger)
        res3 = engine.analyze(text_safe)

        # res1 and res3 should produce identical results
        assert res1.prediction == res3.prediction == "LEGITIMATE"
        assert res1.risk_level == res3.risk_level == "LOW"
        assert abs(res1.risk_score - res3.risk_score) < 1e-4

        # res2 must be flagged
        assert res2.prediction == "PHISHING"
        assert res2.risk_level in ("HIGH", "CRITICAL")
