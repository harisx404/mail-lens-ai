"""
Mail-Lens AI — Feature Engineering Pipeline

Combines NLP features (TF-IDF) with cybersecurity features into a
single feature matrix for ML training and inference.

Feature Groups:
  1. NLP Features: TF-IDF vectors from preprocessed text
  2. Security Features: Extracted from raw text (before preprocessing)

IMPORTANT: The TF-IDF vectorizer is FIT ONLY on training data.
Validation and test data are TRANSFORMED only — no leakage.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.security_features import SecurityFeatureExtractor
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils.config import (
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds the combined feature matrix from email text.

    Workflow:
        1. Preprocess text (NLP pipeline)
        2. Fit TF-IDF on training text ONLY
        3. Extract cybersecurity features from raw text
        4. Combine into single feature matrix

    Usage:
        engineer = FeatureEngineer()
        X_train, y_train = engineer.fit_transform(train_df)
        X_test, y_test = engineer.transform(test_df)
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple = TFIDF_NGRAM_RANGE,
        min_df: int = TFIDF_MIN_DF,
        max_df: float = TFIDF_MAX_DF,
        use_security_features: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = True,
    ):
        self.use_security_features = use_security_features

        # NLP components
        self.preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
        )
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Apply log normalization
            strip_accents="unicode",
        )

        # Security feature extractor
        self.security_extractor = SecurityFeatureExtractor()

        # State
        self._is_fitted = False
        self._feature_names = None

    @property
    def is_fitted(self) -> bool:
        """Whether the vectorizer and feature extractor have been fitted."""
        return self._is_fitted

    @property
    def tfidf_vectorizer(self) -> TfidfVectorizer:
        """Alias for vectorizer for explicit NLP naming."""
        return self.vectorizer

    def _preprocess_texts(self, texts: pd.Series) -> list:
        """Apply NLP preprocessing to a series of texts."""
        logger.info(f"Preprocessing {len(texts)} texts...")
        return [self.preprocessor.preprocess(str(t)) for t in texts]

    def _extract_security_features(self, texts: pd.Series) -> np.ndarray:
        """Extract cybersecurity features from raw (un-preprocessed) texts."""
        logger.info(f"Extracting security features from {len(texts)} texts...")
        features = []
        for text in texts:
            sf = self.security_extractor.extract(str(text))
            features.append(sf.to_dict())

        df = pd.DataFrame(features)
        return df.values.astype(np.float64)

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the feature pipeline on training data and transform it.

        IMPORTANT: This fits the TF-IDF vectorizer. Only call on training data.

        Args:
            df: Training DataFrame with 'text' and 'label' columns

        Returns:
            Tuple of (X_train feature matrix, y_train labels)
        """
        logger.info("=" * 60)
        logger.info("FIT-TRANSFORM: Building features on TRAINING data")
        logger.info("=" * 60)

        # Preprocess text for TF-IDF
        preprocessed = self._preprocess_texts(df["text"])

        # Fit + transform TF-IDF (ONLY on training data)
        X_tfidf = self.vectorizer.fit_transform(preprocessed)
        logger.info(f"TF-IDF shape: {X_tfidf.shape}")

        # Extract security features from RAW text (not preprocessed)
        if self.use_security_features:
            X_security = self._extract_security_features(df["text"])
            logger.info(f"Security features shape: {X_security.shape}")

            # Combine sparse TF-IDF + dense security features
            X_combined = hstack([X_tfidf, X_security]).tocsr()
            logger.info(f"Combined feature matrix shape: {X_combined.shape}")

            # Store feature names
            tfidf_names = self.vectorizer.get_feature_names_out().tolist()
            security_names = list(
                self.security_extractor.extract("").to_dict().keys()
            )
            self._feature_names = tfidf_names + security_names
        else:
            X_combined = X_tfidf
            self._feature_names = self.vectorizer.get_feature_names_out().tolist()

        # Labels
        from src.utils.config import CLASS_LABELS
        y = df["label"].map(CLASS_LABELS).values

        self._is_fitted = True
        return X_combined, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using the already-fitted pipeline.

        NO FITTING occurs here — prevents data leakage.

        Args:
            df: DataFrame with 'text' and 'label' columns

        Returns:
            Tuple of (X feature matrix, y labels)
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform.")

        logger.info(f"TRANSFORM: Building features for {len(df)} samples")

        # Preprocess
        preprocessed = self._preprocess_texts(df["text"])

        # Transform TF-IDF (NO fitting)
        X_tfidf = self.vectorizer.transform(preprocessed)

        # Extract security features
        if self.use_security_features:
            X_security = self._extract_security_features(df["text"])
            X_combined = hstack([X_tfidf, X_security]).tocsr()
        else:
            X_combined = X_tfidf

        # Labels
        from src.utils.config import CLASS_LABELS
        y = df["label"].map(CLASS_LABELS).values

        return X_combined, y

    def transform_single(self, text: str) -> np.ndarray:
        """
        Transform a single text for inference.

        Args:
            text: Raw email text

        Returns:
            Feature vector (1 x n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform.")

        preprocessed = self.preprocessor.preprocess(text)
        X_tfidf = self.vectorizer.transform([preprocessed])

        if self.use_security_features:
            sf = self.security_extractor.extract(text)
            security_vals = np.array([list(sf.to_dict().values())])
            X_combined = hstack([X_tfidf, security_vals]).tocsr()
        else:
            X_combined = X_tfidf

        return X_combined

    def get_feature_names(self) -> list:
        """Return list of all feature names."""
        return self._feature_names or []

    def save(self, path: Path) -> None:
        """Save fitted pipeline components."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "tfidf_vectorizer.joblib")
        joblib.dump(self._feature_names, path / "feature_names.joblib")
        joblib.dump(
            {
                "use_security_features": self.use_security_features,
                "remove_stopwords": self.preprocessor.remove_stopwords,
                "lemmatize": self.preprocessor.lemmatize,
            },
            path / "feature_config.joblib",
        )
        logger.info(f"Feature pipeline saved to {path}")

    def load(self, path: Path) -> None:
        """Load a previously fitted pipeline."""
        self.vectorizer = joblib.load(path / "tfidf_vectorizer.joblib")
        self._feature_names = joblib.load(path / "feature_names.joblib")
        config = joblib.load(path / "feature_config.joblib")
        self.use_security_features = config["use_security_features"]
        self.preprocessor = TextPreprocessor(
            remove_stopwords=config["remove_stopwords"],
            lemmatize=config["lemmatize"],
        )
        self._is_fitted = True
        logger.info(f"Feature pipeline loaded from {path}")
