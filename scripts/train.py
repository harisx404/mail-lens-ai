"""
PhishGuard AI — Main Training Script

Executes the full training pipeline:
  Phase 2: Data Pipeline
  Phase 3-4: Feature Engineering
  Phase 5: Baseline Model Training
  Phase 6: Model Selection

Usage:
    python scripts/train.py

All results are from actual runs — nothing is fabricated.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import run_data_pipeline
from src.evaluation.evaluator import compare_experiments, evaluate_model
from src.features.feature_engineer import FeatureEngineer
from src.models.trainer import (
    create_model,
    save_model,
    train_model,
)
from src.utils.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Execute the full training pipeline."""
    logger.info("=" * 70)
    logger.info("PHISHGUARD AI — TRAINING PIPELINE")
    logger.info("=" * 70)

    # ============================================
    # PHASE 2: Data Pipeline
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: DATA PIPELINE")
    logger.info("=" * 70)

    train_df, val_df, test_df = run_data_pipeline(
        raw_dir=DATA_RAW_DIR,
        processed_dir=DATA_PROCESSED_DIR,
        test_size=TEST_SIZE,
        val_size=VALIDATION_SIZE,
        random_seed=RANDOM_SEED,
    )

    # ============================================
    # PHASE 3-4: Feature Engineering
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3-4: FEATURE ENGINEERING")
    logger.info("=" * 70)

    # Feature Engineering: Fit TF-IDF + Security Features
    fe_combined = FeatureEngineer(use_security_features=True)
    X_train_combined, y_train = fe_combined.fit_transform(train_df)
    X_val_combined, y_val = fe_combined.transform(val_df)
    X_test_combined, y_test = fe_combined.transform(test_df)

    # For TF-IDF only experiments (E1, E2, E3), slice the TF-IDF columns (first n features)
    n_tfidf = len(fe_combined.vectorizer.get_feature_names_out())
    X_train_tfidf = X_train_combined[:, :n_tfidf]
    X_val_tfidf = X_val_combined[:, :n_tfidf]
    X_test_tfidf = X_test_combined[:, :n_tfidf]

    # Create fe_tfidf_only object for saving if best model is TF-IDF only
    fe_tfidf_only = FeatureEngineer(use_security_features=False)
    fe_tfidf_only.vectorizer = fe_combined.vectorizer
    fe_tfidf_only._is_fitted = True
    fe_tfidf_only._feature_names = fe_combined.vectorizer.get_feature_names_out().tolist()


    # ============================================
    # PHASE 5: Baseline Model Training
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: BASELINE MODELS")
    logger.info("=" * 70)

    experiments = []
    all_reports = []

    # --- Experiment E1: TF-IDF + Logistic Regression ---
    model_lr, result_lr = train_model(
        model=create_model("logistic_regression"),
        X_train=X_train_tfidf,
        y_train=y_train,
        model_name="Logistic Regression",
        experiment_name="E1",
        feature_set="TF-IDF only",
    )
    report_lr = evaluate_model(
        model_lr, X_val_tfidf, y_val,
        "Logistic Regression", "E1", "validation"
    )
    experiments.append(result_lr)
    all_reports.append(report_lr)

    # --- Experiment E2: TF-IDF + Naive Bayes ---
    model_nb, result_nb = train_model(
        model=create_model("naive_bayes"),
        X_train=X_train_tfidf,
        y_train=y_train,
        model_name="Naive Bayes",
        experiment_name="E2",
        feature_set="TF-IDF only",
    )
    report_nb = evaluate_model(
        model_nb, X_val_tfidf, y_val,
        "Naive Bayes", "E2", "validation"
    )
    experiments.append(result_nb)
    all_reports.append(report_nb)

    # --- Experiment E3: TF-IDF + Linear SVM ---
    model_svm, result_svm = train_model(
        model=create_model("linear_svm"),
        X_train=X_train_tfidf,
        y_train=y_train,
        model_name="Linear SVM",
        experiment_name="E3",
        feature_set="TF-IDF only",
    )
    report_svm = evaluate_model(
        model_svm, X_val_tfidf, y_val,
        "Linear SVM", "E3", "validation"
    )
    experiments.append(result_svm)
    all_reports.append(report_svm)

    # --- Experiment E4: TF-IDF + Security + Logistic Regression ---
    model_lr_sec, result_lr_sec = train_model(
        model=create_model("logistic_regression"),
        X_train=X_train_combined,
        y_train=y_train,
        model_name="Logistic Regression",
        experiment_name="E4",
        feature_set="TF-IDF + Security Features",
    )
    report_lr_sec = evaluate_model(
        model_lr_sec, X_val_combined, y_val,
        "Logistic Regression", "E4", "validation"
    )
    experiments.append(result_lr_sec)
    all_reports.append(report_lr_sec)

    # --- Experiment E5: TF-IDF + Security + Linear SVM ---
    model_svm_sec, result_svm_sec = train_model(
        model=create_model("linear_svm"),
        X_train=X_train_combined,
        y_train=y_train,
        model_name="Linear SVM",
        experiment_name="E5",
        feature_set="TF-IDF + Security Features",
    )
    report_svm_sec = evaluate_model(
        model_svm_sec, X_val_combined, y_val,
        "Linear SVM", "E5", "validation"
    )
    experiments.append(result_svm_sec)
    all_reports.append(report_svm_sec)

    # ============================================
    # PHASE 6: Model Selection
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: MODEL SELECTION")
    logger.info("=" * 70)

    # Compare all experiments
    comparison = compare_experiments(all_reports)

    # Select best model by macro F1 (prioritizes balanced per-class performance)
    best_report = max(all_reports, key=lambda r: r.macro_f1)
    best_experiment = best_report.experiment_name

    logger.info(f"\nBest model: {best_report.model_name} ({best_experiment})")
    logger.info(f"Selection criterion: Macro F1 = {best_report.macro_f1}")

    # Map experiment to its model and feature engineer
    experiment_models = {
        "E1": (model_lr, fe_tfidf_only),
        "E2": (model_nb, fe_tfidf_only),
        "E3": (model_svm, fe_tfidf_only),
        "E4": (model_lr_sec, fe_combined),
        "E5": (model_svm_sec, fe_combined),
    }

    best_model, best_fe = experiment_models[best_experiment]

    # Evaluate best model on TEST set (final evaluation)
    if best_fe.use_security_features:
        X_test_final = X_test_combined
    else:
        X_test_final = X_test_tfidf

    test_report = evaluate_model(
        best_model, X_test_final, y_test,
        best_report.model_name, best_experiment, "test"
    )

    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Model: {test_report.model_name}")
    logger.info(f"Test Accuracy: {test_report.accuracy}")
    logger.info(f"Test Macro F1: {test_report.macro_f1}")
    logger.info(f"Test Weighted F1: {test_report.weighted_f1}")

    # ============================================
    # Save artifacts
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("SAVING ARTIFACTS")
    logger.info("=" * 70)

    # Save best model
    save_model(best_model, MODELS_DIR, "best_model")

    # Save feature pipeline
    best_fe.save(MODELS_DIR)

    # Save experiment results
    results_path = MODELS_DIR / "experiment_results.json"
    results_data = {
        "experiments": [
            {
                "name": r.experiment_name,
                "model": r.model_name,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
                "per_class": r.per_class_metrics,
                "split": r.dataset_split,
            }
            for r in all_reports
        ],
        "best_experiment": best_experiment,
        "test_results": test_report.to_dict(),
        "random_seed": RANDOM_SEED,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Experiment results saved to {results_path}")

    # Save model metadata
    metadata = {
        "model_name": best_report.model_name,
        "experiment": best_experiment,
        "feature_set": "TF-IDF + Security Features" if best_fe.use_security_features else "TF-IDF only",
        "test_accuracy": test_report.accuracy,
        "test_macro_f1": test_report.macro_f1,
        "test_weighted_f1": test_report.weighted_f1,
        "test_per_class": test_report.per_class_metrics,
        "random_seed": RANDOM_SEED,
        "classes": ["LEGITIMATE", "PHISHING", "MALICIOUS"],
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final model: {best_report.model_name} ({best_experiment})")
    logger.info(f"Test Accuracy: {test_report.accuracy}")
    logger.info(f"Test Macro F1: {test_report.macro_f1}")
    logger.info(f"Artifacts saved to: {MODELS_DIR}")

    return test_report


if __name__ == "__main__":
    main()
