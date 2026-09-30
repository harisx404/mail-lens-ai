"""
Mail-Lens AI — Model Trainer

Trains and evaluates multiple ML classifiers:
  - Logistic Regression (baseline)
  - Multinomial Naive Bayes (baseline)
  - Linear SVM (baseline)

Experiment tracking with full reproducibility:
  - Random seed fixed
  - Hyperparameters recorded
  - Training time recorded
  - All metrics from actual evaluation runs
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.utils.config import CLASSES, RANDOM_SEED

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Record of a single training experiment."""

    experiment_name: str
    model_name: str
    feature_set: str
    hyperparameters: Dict[str, Any]
    random_seed: int
    training_time_seconds: float
    train_samples: int
    n_features: int
    classes: List[str] = field(default_factory=lambda: CLASSES.copy())
    notes: str = ""


def create_model(
    model_name: str,
    class_weights: Optional[dict] = None,
    random_seed: int = RANDOM_SEED,
) -> Any:
    """
    Create an ML model by name.

    Supported models:
        - 'logistic_regression' or 'lr'
        - 'naive_bayes' or 'nb'
        - 'linear_svm' or 'svm'

    Class weights are applied to handle imbalanced MALICIOUS class.
    """
    name = model_name.lower().strip()

    if name in ("logistic_regression", "lr"):
        return LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
            class_weight=class_weights or "balanced",
            solver="lbfgs",
            C=1.0,
        )


    elif name in ("naive_bayes", "nb"):
        # MultinomialNB doesn't support class_weight directly.
        # We handle this by using sample_weight during fit.
        return MultinomialNB(alpha=1.0)

    elif name in ("linear_svm", "svm"):
        # LinearSVC doesn't produce probabilities by default.
        # Wrap with CalibratedClassifierCV for probability support.
        base_svm = LinearSVC(
            max_iter=2000,
            random_state=random_seed,
            class_weight=class_weights or "balanced",
            C=1.0,
            dual="auto",
        )
        return CalibratedClassifierCV(base_svm, cv=3)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: logistic_regression, naive_bayes, linear_svm"
        )


def get_model_hyperparameters(model: Any) -> Dict[str, Any]:
    """Extract hyperparameters from a model for experiment logging."""
    params = {}
    try:
        if hasattr(model, "get_params"):
            params = model.get_params()
        # Clean up non-serializable values
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None), list, tuple)):
                clean_params[k] = v
            else:
                clean_params[k] = str(v)
        return clean_params
    except Exception:
        return {"error": "Could not extract parameters"}


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute sample weights for class balancing.
    Used for models that don't support class_weight (e.g., MultinomialNB).
    """
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    experiment_name: str,
    feature_set: str = "tfidf+security",
    random_seed: int = RANDOM_SEED,
) -> Tuple[Any, ExperimentResult]:
    """
    Train a model and record the experiment.

    Args:
        model: Scikit-learn compatible model
        X_train: Training feature matrix
        y_train: Training labels
        model_name: Human-readable model name
        experiment_name: Experiment identifier (e.g., "E1")
        feature_set: Description of features used
        random_seed: Random seed

    Returns:
        Tuple of (trained_model, ExperimentResult)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {experiment_name} — {model_name}")
    logger.info(f"Features: {feature_set}")
    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Feature dimensions: {X_train.shape[1]}")
    logger.info(f"{'='*60}")

    # Record hyperparameters
    hyperparams = get_model_hyperparameters(model)

    # Train
    start_time = time.time()

    if isinstance(model, MultinomialNB):
        # MultinomialNB needs non-negative features
        # TF-IDF is non-negative, but security features might have
        # zero values (which is fine). Ensure no negatives.
        from scipy.sparse import issparse
        if issparse(X_train):
            X_fit = X_train.copy()
            X_fit[X_fit < 0] = 0
        else:
            X_fit = np.clip(X_train, 0, None)

        # Use sample weights for class balancing
        sample_weights = compute_sample_weights(y_train)
        model.fit(X_fit, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")

    result = ExperimentResult(
        experiment_name=experiment_name,
        model_name=model_name,
        feature_set=feature_set,
        hyperparameters=hyperparams,
        random_seed=random_seed,
        training_time_seconds=round(training_time, 2),
        train_samples=X_train.shape[0],
        n_features=X_train.shape[1],
    )

    return model, result


def save_model(model: Any, path: Path, name: str) -> Path:
    """Save a trained model to disk."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{name}.joblib"
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(path: Path) -> Any:
    """Load a trained model from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
