# Testing

## Overview

The test suite covers the NLP preprocessing pipeline, security feature extraction, ML training utilities, end-to-end inference, and hardened edge case handling.

**Current status:** 71/71 tests passing · 76.77% statement coverage across `src/`

## Running tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a specific file
python -m pytest tests/test_integration.py -v

# Run a specific test
python -m pytest tests/test_core.py::TestTextPreprocessor::test_html_removal -v

# Run and stop on first failure
python -m pytest tests/ -x
```

## Test files

### `tests/test_core.py` (36 tests)

Unit tests for the foundational pipeline components:

| Class | Tests |
|---|---|
| `TestTextPreprocessor` | HTML removal, URL normalization, email normalization, unicode, whitespace, lowercasing, short text, extraction methods |
| `TestSecurityFeatures` | Urgency/credential/threat/reward detection, URL features, executable mention, legitimate baseline, phishing baseline |
| `TestDataPipeline` | Label mapping for both datasets, empty row removal, deduplication |
| `TestConfiguration` | Class label definitions, keyword lists populated, risk level thresholds |
| `TestInferenceEdgeCases` | `AnalysisResult` dataclass serialization and defaults |
| `TestTrainedModelInference` | Model loaded, prediction on legitimate/phishing/malicious examples, empty input handling |

### `tests/test_extended.py` (12 tests)

| Class | Tests |
|---|---|
| `TestFeatureEngineer` | Fit/transform pipeline, transform without security features, unfitted error, single-string transform |
| `TestModelFactory` | Creation of LR, NB, LinearSVM, invalid model error, model alias resolution |
| `TestEvaluationModule` | `EvaluationReport` dataclass, `evaluate_model()` with synthetic data, `compare_experiments()` |

### `tests/test_integration.py` (10 tests)

End-to-end integration tests:

| Class | Tests |
|---|---|
| `TestSecurityFeatureRegressions` | `.com` domain not triggering executable indicator (regression), real `.exe` triggering, typosquatting detection, exact brand not flagged, simulation cloaking detection |
| `TestEndToEndInferenceIntegration` | Simulation-cloaked phishing triggers risk override, corporate newsletter → legitimate, urgent credential request → phishing, malware delivery → flagged, batch sequential consistency |

### `tests/test_qa_regression.py` (13 tests)

Hardened regression and boundary tests:

| Test | What it verifies |
|---|---|
| `test_dynamic_pipeline_trace_feature_dimensions` | `pipeline_trace.total_feature_dimensions` == 10000 (not hardcoded 10020) |
| `test_xss_injection_in_sender_and_reasons` | `<script>alert('XSS')</script>` in sender is escaped in generated output |
| `test_xss_escaping_in_highlighter` | HTML injection in body text is escaped in highlight output |
| `test_boundary_50000_char_input` | 50k character input completes without error |
| `test_boundary_empty_and_whitespace` | Blank / space / tab / newline input → `UNKNOWN` |
| `test_sql_and_command_injection_treated_as_plain_text` | SQL and shell strings produce valid predictions, not crashes |
| `test_risk_threshold_boundaries` | Risk scores map correctly to LOW/MEDIUM/HIGH/CRITICAL |
| `test_calibrated_probabilities_sum_to_one` | `sum(probabilities.values()) ≈ 1.0` within 1e-6 |
| `test_unicode_and_foreign_script_robustness` | Arabic, Cyrillic, Chinese, emoji inputs don't crash |
| `test_url_tld_exception_handling_malformed_url` | Malformed URLs handled gracefully |
| `test_model_artifact_cross_consistency` | `best_model`, `tfidf_vectorizer`, and `feature_config` agree on feature count |
| `test_homoglyph_lookalike_detection` | `rnicrosoft.com` triggers typosquatting indicator |
| `test_latency_benchmark_under_50ms` | Single inference completes in <50ms on CPU |

## Coverage report

```
Name                                     Stmts   Miss  Cover
----------------------------------------------------------------------
src\utils\config.py                         41      0   100%
src\features\security_features.py          210     10    95%
src\inference\engine.py                    188     16    91%
src\preprocessing\text_preprocessor.py     85      8    91%
src\features\feature_engineer.py           98      7    93%
src\evaluation\evaluator.py                81      5    94%
src\models\trainer.py                      75     43    43%
src\data\pipeline.py                       89     51    43%
src\data\loader.py                         93     83    11%
----------------------------------------------------------------------
TOTAL                                      960    223    77%
```

Low coverage on `data/loader.py` and `data/pipeline.py` reflects that these are only exercised by the full training script (not unit-testable without downloading external data). Core inference and analysis modules all exceed 90% coverage.

## What is not covered by tests

- `app/main.py` (UI rendering logic) — Streamlit does not have a standard unit testing approach for its rendering layer
- Full data download and processing pipeline (requires internet and ~100MB download)
- The training script end-to-end

Manual testing of the Streamlit UI is documented in [docs/WORKDONE.md](WORKDONE.md).
