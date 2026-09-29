"""
PhishGuard AI — Extended Test Suite

Comprehensive tests for:
  - FeatureEngineer pipeline (fit, transform, serialization)
  - ModelTrainer and create_model factory
  - EvaluationReport, evaluate_model, compare_experiments
  - Edge cases in inference and preprocessing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import EvaluationReport, compare_experiments, evaluate_model
from src.features.feature_engineer import FeatureEngineer
from src.models.trainer import ExperimentResult, create_model
from src.utils.config import CLASSES


# ============================================
# Feature Engineering Tests
# ============================================
class TestFeatureEngineer:
    """Tests for combined NLP + Security feature engineering."""

    def test_feature_engineer_fit_transform(self):
        df_train = pd.DataFrame({
            "text": [
                "Hello team, please see the attached report for the weekly meeting.",
                "URGENT: Click http://192.168.1.1/login.exe to verify your account immediately!",
                "Invoice attached. Enable macros and run invoice.scr to execute payment.",
                "Regular update regarding project status and team assignments.",
            ],
            "label": [0, 1, 2, 0],
        })

        df_test = pd.DataFrame({
            "text": [
                "Team meeting is scheduled for tomorrow afternoon.",
                "Action required: update your password now http://phish-site.xyz/update",
            ],
            "label": [0, 1],
        })

        fe = FeatureEngineer(
            max_features=50,
            min_df=1,
            max_df=1.0,
            use_security_features=True,
            remove_stopwords=False,
            lemmatize=False,
        )

        X_train, y_train = fe.fit_transform(df_train)
        assert fe.is_fitted
        assert X_train.shape[0] == 4
        # TF-IDF features (up to 50) + 17 security features
        assert X_train.shape[1] > 17
        assert len(y_train) == 4

        X_test, y_test = fe.transform(df_test)
        assert X_test.shape[0] == 2
        assert X_test.shape[1] == X_train.shape[1]
        assert len(y_test) == 2

        # Check feature names
        feature_names = fe.get_feature_names()
        assert len(feature_names) == X_train.shape[1]
        assert "urgency_score" in feature_names

    def test_feature_engineer_without_security_features(self):
        df = pd.DataFrame({
            "text": ["Safe email about work", "Suspicious link here"],
            "label": [0, 1],
        })
        fe = FeatureEngineer(max_features=20, min_df=1, max_df=1.0, use_security_features=False)
        X, y = fe.fit_transform(df)
        assert fe.is_fitted
        assert X.shape[0] == 2
        assert len(y) == 2

    def test_transform_unfitted_raises_error(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({"text": ["test"]})
        with pytest.raises(RuntimeError):
            fe.transform(df)

    def test_transform_single_email(self):
        df_train = pd.DataFrame({
            "text": ["Quarterly report attached", "Reset password immediately http://fake.ru"],
            "label": [0, 1],
        })
        fe = FeatureEngineer(max_features=20, min_df=1, max_df=1.0)
        fe.fit_transform(df_train)

        feat_vector = fe.transform_single("Please review the attached quarterly metrics")
        assert feat_vector.shape[0] == 1
        assert feat_vector.shape[1] == len(fe.get_feature_names())


# ============================================
# Model Factory Tests
# ============================================
class TestModelFactory:
    """Tests for model creation and initialization."""

    def test_create_logistic_regression(self):
        model = create_model("logistic_regression")
        assert isinstance(model, LogisticRegression)
        assert model.max_iter == 1000

    def test_create_naive_bayes(self):
        model = create_model("naive_bayes")
        assert isinstance(model, MultinomialNB)

    def test_create_linear_svm(self):
        model = create_model("linear_svm")
        # Wrapped in CalibratedClassifierCV
        assert hasattr(model, "estimator") or hasattr(model, "base_estimator")

    def test_create_invalid_model(self):
        with pytest.raises(ValueError):
            create_model("nonexistent_model")

    def test_model_aliases(self):
        m1 = create_model("lr")
        m2 = create_model("nb")
        m3 = create_model("svm")
        assert m1 is not None
        assert m2 is not None
        assert m3 is not None


# ============================================
# Evaluation & Reporting Tests
# ============================================
class TestEvaluationModule:
    """Tests for model evaluation metrics and reports."""

    def test_evaluation_report_dataclass(self):
        report = EvaluationReport(
            model_name="Test Model",
            experiment_name="E_TEST",
            dataset_split="test",
            n_samples=100,
            accuracy=0.95,
            macro_precision=0.94,
            macro_recall=0.93,
            macro_f1=0.935,
            weighted_f1=0.95,
        )
        d = report.to_dict()
        assert d["model_name"] == "Test Model"
        assert d["accuracy"] == 0.95
        assert d["n_samples"] == 100

    def test_evaluate_model_with_synthetic_data(self):
        from sklearn.linear_model import LogisticRegression

        # Create simple separable dataset
        X = np.array([
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1],
            [0.5, 0.5],
            [0.6, 0.5],
        ])
        y = np.array([0, 0, 1, 1, 2, 2])

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        report = evaluate_model(
            model=model,
            X=X,
            y_true=y,
            model_name="Toy LR",
            experiment_name="E_TOY",
            dataset_split="test",
            class_names=["LEGITIMATE", "PHISHING", "MALICIOUS"],
        )

        assert report.accuracy >= 0.5
        assert report.macro_f1 > 0
        assert len(report.confusion_matrix) == 3
        assert "LEGITIMATE" in report.per_class_metrics
        assert "PHISHING" in report.per_class_metrics
        assert "MALICIOUS" in report.per_class_metrics

    def test_compare_experiments(self):
        r1 = EvaluationReport(
            model_name="Model A",
            experiment_name="E1",
            dataset_split="test",
            n_samples=50,
            accuracy=0.90,
            macro_f1=0.88,
            weighted_f1=0.90,
        )
        r2 = EvaluationReport(
            model_name="Model B",
            experiment_name="E2",
            dataset_split="test",
            n_samples=50,
            accuracy=0.96,
            macro_f1=0.95,
            weighted_f1=0.96,
        )
        table = compare_experiments([r1, r2])
        assert "Model A" in table
        assert "Model B" in table
        assert "0.9600" in table
