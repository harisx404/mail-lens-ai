"""
PhishGuard AI — Inference Pipeline

Single clean pipeline for production inference:
  Input → Parse → Preprocess → Features → Model → Prediction → Risk → Explanation

This module is used by both the web application and command-line inference.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.features.feature_engineer import FeatureEngineer
from src.features.security_features import SecurityFeatureExtractor
from src.utils.config import CLASSES, LABEL_TO_CLASS, MODELS_DIR, RISK_LEVELS

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result for a single email."""

    # Classification
    prediction: str = ""              # "LEGITIMATE", "PHISHING", "MALICIOUS"
    confidence: float = 0.0           # Probability of predicted class
    probabilities: Dict[str, float] = field(default_factory=dict)

    # Risk Assessment
    risk_level: str = ""              # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_score: float = 0.0           # 0.0 - 1.0
    risk_color: str = "#28a745"       # CSS color for display

    # Security Indicators
    detected_indicators: List[str] = field(default_factory=list)
    indicator_count: int = 0

    # Explainability
    explanation: str = ""             # Human-readable explanation
    recommendation: str = ""          # Action recommendation

    # Input metadata
    email_snippet: str = ""           # First N chars of input (for history)

    def to_dict(self) -> dict:
        """Convert to dictionary for API/UI consumption."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "risk_color": self.risk_color,
            "detected_indicators": self.detected_indicators,
            "indicator_count": self.indicator_count,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
        }


class PhishGuardInference:

    """
    Production inference pipeline for PhishGuard AI.

    Usage:
        engine = PhishGuardInference()
        engine.load_model("models/saved")
        result = engine.analyze("Dear customer, verify your account...")
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model = None
        self.feature_engineer = None
        self.security_extractor = SecurityFeatureExtractor()
        self._is_loaded = False

        target_dir = Path(model_dir) if model_dir else MODELS_DIR
        if (target_dir / "best_model.joblib").exists():
            try:
                self.load_model(target_dir)
            except Exception as e:
                logger.warning(f"Could not auto-load model from {target_dir}: {e}")


    def load_model(self, model_dir: Path) -> None:
        """
        Load trained model and feature pipeline from disk.

        Expects:
            model_dir/best_model.joblib
            model_dir/tfidf_vectorizer.joblib
            model_dir/feature_names.joblib
            model_dir/feature_config.joblib
        """
        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                f"Please train the model first."
            )

        # Load model
        model_path = model_dir / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please train the model first."
            )
        self.model = joblib.load(model_path)

        # Load feature pipeline
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.load(model_dir)

        self._is_loaded = True
        logger.info("Inference pipeline loaded successfully")

    def _compute_risk_score(
        self,
        prediction: str,
        confidence: float,
        indicator_count: int,
        security_features: dict,
    ) -> float:
        """
        Compute risk score (0.0 - 1.0) from multiple signals.

        Risk Score Formula:
            base = classification_probability (for threat classes)
            modifier = security_indicator_density
            final = weighted combination

        This is an APPLICATION-LEVEL risk assessment, not an
        objectively validated security rating.
        """
        if prediction == "LEGITIMATE":
            # For legitimate emails, risk = (1 - confidence) * indicator_modifier
            base = 1.0 - confidence
            indicator_modifier = min(1.0, indicator_count / 5)
            return round(min(1.0, base * 0.6 + indicator_modifier * 0.4), 3)
        else:
            # For threat predictions, risk = confidence * indicator_modifier
            base = confidence
            indicator_modifier = min(1.0, indicator_count / 3)
            url_modifier = min(0.2, security_features.get("url_count", 0) * 0.05)
            exec_modifier = 0.1 if security_features.get("has_executable_mention", 0) else 0.0

            return round(
                min(1.0, base * 0.6 + indicator_modifier * 0.25 + url_modifier + exec_modifier),
                3,
            )

    def _get_risk_level(self, risk_score: float) -> tuple:
        """Map risk score to risk level and color."""
        for level, config in RISK_LEVELS.items():
            if config["min_score"] <= risk_score <= config["max_score"]:
                return level, config["color"]
        return "HIGH", "#fd7e14"

    def _generate_explanation(
        self,
        prediction: str,
        confidence: float,
        indicators: List[str],
    ) -> str:
        """
        Generate human-readable explanation.

        NOTE: This explanation combines the model's prediction with
        security heuristic indicators. The indicators are NOT the
        exact reasons the ML model made its decision — they are
        additional contextual signals detected by the security
        feature extractor.

        Distinction:
        - Model prediction: Based on learned patterns from TF-IDF + features
        - Security indicators: Heuristic detections from keyword/pattern matching
        """
        parts = []

        if prediction == "LEGITIMATE":
            parts.append(
                f"The email appears to be legitimate (confidence: {confidence:.1%})."
            )
            if indicators:
                parts.append(
                    "However, some security indicators were detected — "
                    "review the email carefully."
                )
        elif prediction == "PHISHING":
            parts.append(
                f"This email shows characteristics of a phishing attempt "
                f"(confidence: {confidence:.1%})."
            )
            parts.append(
                "Phishing emails use social engineering to deceive recipients "
                "into revealing sensitive information or taking harmful actions."
            )
        elif prediction == "MALICIOUS":
            parts.append(
                f"This email shows characteristics associated with malicious "
                f"content delivery (confidence: {confidence:.1%})."
            )
            parts.append(
                "Malicious emails may contain harmful attachments, links to "
                "malware, or attempt to exploit vulnerabilities."
            )

        if indicators:
            parts.append(f"\nDetected security indicators ({len(indicators)}):")
            for ind in indicators:
                parts.append(f"  • {ind}")

        return "\n".join(parts)

    def _generate_recommendation(
        self,
        prediction: str,
        risk_level: str,
        indicators: List[str],
    ) -> str:
        """Generate actionable security recommendation."""
        if prediction == "LEGITIMATE" and risk_level in ("LOW", "MEDIUM"):
            return (
                "This email appears safe. Standard email security practices apply."
            )
        elif prediction == "LEGITIMATE" and risk_level in ("HIGH", "CRITICAL"):
            return (
                "Although classified as legitimate, elevated risk indicators "
                "were detected. Verify the sender's identity before acting on "
                "any requests in this email."
            )
        elif prediction == "PHISHING":
            return (
                "DO NOT click any links or provide personal information. "
                "Do not reply to this email. Report it to your IT security team. "
                "If you have already interacted with the email, change your "
                "passwords immediately and enable two-factor authentication."
            )
        elif prediction == "MALICIOUS":
            return (
                "DO NOT open any attachments or click any links. "
                "Do not download any files referenced in this email. "
                "Report it to your IT security team immediately. "
                "If you have opened attachments, disconnect from the network "
                "and run a full antivirus scan."
            )
        return "Exercise caution with this email."

    def analyze(
        self,
        text: str,
        subject: str = "",
        sender: str = "",
    ) -> AnalysisResult:
        """
        Analyze an email and return a complete result.

        Args:
            text: Email body text
            subject: Optional email subject
            sender: Optional sender address

        Returns:
            AnalysisResult with prediction, risk, indicators, explanation
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

        # Combine subject and body for analysis
        full_text = f"{subject} {text}".strip() if subject else text

        if not full_text:
            return AnalysisResult(
                prediction="UNKNOWN",
                explanation="Please provide email content for analysis.",
                recommendation="No content was provided to analyze.",
            )

        # Extract security features from RAW text
        security_features = self.security_extractor.extract(full_text)

        # Build feature vector
        X = self.feature_engineer.transform_single(full_text)

        # Predict
        # Handle NB non-negative requirement
        from sklearn.naive_bayes import MultinomialNB
        from scipy.sparse import issparse

        X_pred = X
        if isinstance(self.model, MultinomialNB):
            if issparse(X_pred):
                X_pred = X_pred.copy()
                X_pred[X_pred < 0] = 0
            else:
                X_pred = np.clip(X_pred, 0, None)

        prediction_idx = int(self.model.predict(X_pred)[0])
        prediction = LABEL_TO_CLASS.get(prediction_idx, "UNKNOWN")

        # Probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_pred)[0]
            probabilities = {}
            for i, p in enumerate(proba):
                class_name = LABEL_TO_CLASS.get(i, f"CLASS_{i}")
                probabilities[class_name] = round(float(p), 4)
            confidence = round(float(max(proba)), 4)
        else:
            probabilities = {prediction: 1.0}
            confidence = 1.0

        # Risk assessment
        risk_score = self._compute_risk_score(
            prediction, confidence,
            security_features.total_indicators,
            security_features.to_dict(),
        )
        risk_level, risk_color = self._get_risk_level(risk_score)

        # Explanation
        explanation = self._generate_explanation(
            prediction, confidence, security_features.detected_indicators
        )
        recommendation = self._generate_recommendation(
            prediction, risk_level, security_features.detected_indicators
        )

        return AnalysisResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_color=risk_color,
            detected_indicators=security_features.detected_indicators,
            indicator_count=security_features.total_indicators,
            explanation=explanation,
            recommendation=recommendation,
            email_snippet=full_text[:100] + "..." if len(full_text) > 100 else full_text,
        )

    # Alias for analyze
    predict = analyze

