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
from scipy.sparse import issparse
from sklearn.naive_bayes import MultinomialNB

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

    # Explainability & AI Insights
    explanation: str = ""             # Human-readable explanation
    recommendation: str = ""          # Action recommendation
    top_threat_tokens: List[Dict[str, Any]] = field(default_factory=list)
    top_safe_tokens: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_trace: Dict[str, Any] = field(default_factory=dict)

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
            "top_threat_tokens": self.top_threat_tokens,
            "top_safe_tokens": self.top_safe_tokens,
            "pipeline_trace": self.pipeline_trace,
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

    def _apply_risk_override(
        self,
        risk_score: float,
        risk_level: str,
        prediction: str,
        probabilities: dict,
        indicator_count: int,
        security_features_dict: dict = None,
    ) -> tuple:
        """
        Apply security indicator override to risk assessment.

        Escalates risk level when dangerous threat signals are present:
        - Executable attachments/mentions with threat classifications
        - Brand typosquatting / lookalike domain attacks
        - High density of security indicators with threat probabilities
        """
        sec = security_features_dict or {}
        non_legit_prob = probabilities.get("PHISHING", 0.0) + probabilities.get("MALICIOUS", 0.0)

        # Override 1: Executable payload mention with threat classification
        if sec.get("has_executable_mention", 0) and (
            prediction in ("PHISHING", "MALICIOUS") or non_legit_prob > 0.40
        ):
            risk_score = max(risk_score, 0.70)
            risk_level, _ = self._get_risk_level(risk_score)

        # Override 2: Brand impersonation / typosquatting detected
        if (
            sec.get("is_typosquat", 0) or sec.get("brand_similarity_score", 0.0) >= 0.80
        ) and non_legit_prob > 0.15:
            risk_score = max(risk_score, 0.65)
            risk_level, _ = self._get_risk_level(risk_score)

        # Override 3: 3+ indicators fire with threat probability > 15%
        if indicator_count >= 3 and non_legit_prob > 0.15:
            risk_score = max(risk_score, 0.65)
            risk_level, _ = self._get_risk_level(risk_score)

        # Override 4: Threat prediction with multiple indicators and majority threat probability
        if (
            prediction in ("PHISHING", "MALICIOUS")
            and indicator_count >= 2
            and non_legit_prob >= 0.50
        ):
            risk_score = max(risk_score, 0.65)
            risk_level, _ = self._get_risk_level(risk_score)

        return risk_score, risk_level

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

    def _extract_nlp_insights(self, full_text: str, prediction: str) -> tuple:
        """
        Extract NLP feature importance and pipeline traces for AI explainability.

        Computes exact TF-IDF token contributions by projecting input n-grams
        against the calibrated Linear SVM decision boundary hyperplane.
        """
        top_threat_tokens = []
        top_safe_tokens = []
        pipeline_trace = {}

        try:
            if not self.feature_engineer or not self.feature_engineer.tfidf_vectorizer:
                return top_threat_tokens, top_safe_tokens, pipeline_trace

            preprocessor = self.feature_engineer.preprocessor
            cleaned_text = preprocessor.preprocess(full_text)
            raw_tokens = cleaned_text.split()

            vectorizer = self.feature_engineer.tfidf_vectorizer
            vec = vectorizer.transform([cleaned_text])
            feature_names = np.array(vectorizer.get_feature_names_out())

            indices = vec.indices
            data = vec.data

            pipeline_trace = {
                "raw_char_count": len(full_text),
                "raw_word_count": len(full_text.split()),
                "cleaned_text": cleaned_text,
                "token_count": len(raw_tokens),
                "vocab_match_count": int(len(indices)),
                "total_vocab_size": len(vectorizer.vocabulary_),
                "total_feature_dimensions": 10020,
            }

            # Extract base linear model coefficients
            base_model = self.model
            if hasattr(self.model, "calibrated_classifiers_"):
                base_model = self.model.calibrated_classifiers_[0].estimator

            if hasattr(base_model, "coef_") and len(indices) > 0:
                coefs = base_model.coef_  # Shape: (3, 10000)

                # Class 0: LEGITIMATE, Class 1: PHISHING, Class 2: MALICIOUS
                threat_class_idx = 2 if prediction == "MALICIOUS" else 1
                threat_coefs = coefs[threat_class_idx]
                legit_coefs = coefs[0]

                threat_items = []
                safe_items = []

                for idx, val in zip(indices, data):
                    word = str(feature_names[idx])
                    t_impact = float(threat_coefs[idx] * val)
                    l_impact = float(legit_coefs[idx] * val)

                    if t_impact > 0.001:
                        threat_items.append({
                            "token": word,
                            "tfidf": round(float(val), 3),
                            "weight": round(float(threat_coefs[idx]), 3),
                            "impact": round(t_impact, 3),
                        })
                    if l_impact > 0.001:
                        safe_items.append({
                            "token": word,
                            "tfidf": round(float(val), 3),
                            "weight": round(float(legit_coefs[idx]), 3),
                            "impact": round(l_impact, 3),
                        })

                threat_items.sort(key=lambda x: x["impact"], reverse=True)
                safe_items.sort(key=lambda x: x["impact"], reverse=True)

                top_threat_tokens = threat_items[:8]
                top_safe_tokens = safe_items[:8]

        except Exception as e:
            logger.warning(f"Could not extract NLP insights: {e}")

        return top_threat_tokens, top_safe_tokens, pipeline_trace

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

        # Apply security indicator override
        risk_score, risk_level = self._apply_risk_override(
            risk_score,
            risk_level,
            prediction,
            probabilities,
            security_features.total_indicators,
            security_features.to_dict(),
        )
        _, risk_color = self._get_risk_level(risk_score)

        # Explanation
        explanation = self._generate_explanation(
            prediction, confidence, security_features.detected_indicators
        )
        recommendation = self._generate_recommendation(
            prediction, risk_level, security_features.detected_indicators
        )

        # Extract AI / NLP explainability insights
        top_threat_tokens, top_safe_tokens, pipeline_trace = self._extract_nlp_insights(
            full_text, prediction
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
            top_threat_tokens=top_threat_tokens,
            top_safe_tokens=top_safe_tokens,
            pipeline_trace=pipeline_trace,
            email_snippet=full_text[:100] + "..." if len(full_text) > 100 else full_text,
        )

    # Alias for analyze
    predict = analyze

