# Changelog

All notable changes to Mail-Lens AI.

## [1.0.0] — 2026-09-30

### Project rebrand
- Renamed project from PhishGuard AI to **Mail-Lens AI**
- Renamed `PhishGuardInference` class to `MailLensInference`
- Updated all module docstrings, page titles, UI headers, and footer branding
- Updated `pyproject.toml` project name to `mail-lens-ai`
- Removed `python-pptx` from dependencies (presentation tooling not required)

### Repository cleanup
- Removed `presentation/` directory (all 4 presentation files)
- Removed `training.log` from tracking
- Removed `PROJECT_ANALYSIS_REPORT.md` and `FINAL_QA_REPORT.md` (internal audit artifacts)
- Removed `models/saved_backup_validated/` (development backup, not needed in repo)
- Removed `.coverage` binary

### Documentation
- Rewrote `README.md` as a complete, professional GitHub-facing document
- Added `docs/PROJECT_OVERVIEW.md` — problem statement, capabilities, and scope
- Added `docs/ARCHITECTURE.md` — system design, component breakdown, and data flow
- Added `docs/MACHINE_LEARNING.md` — dataset, preprocessing decisions, model benchmark, evaluation
- Added `docs/SECURITY.md` — security posture, input handling, privacy
- Added `docs/DEVELOPMENT.md` — installation, setup, and local dev guide
- Added `docs/DEPLOYMENT.md` — Streamlit Cloud, Docker, Vercel assessment, production checklist
- Added `docs/API.md` — `MailLensInference` Python API reference
- Added `docs/TESTING.md` — test strategy, file breakdown, coverage report
- Added `docs/LIMITATIONS.md` — honest limitations of the system
- Added `docs/PROJECT_DOCUMENTATION.md` — comprehensive technical documentation
- Added `docs/RESUME_PROJECT_SUMMARY.md` — portfolio and resume descriptions
- Updated `docs/WORKDONE.md` with current project status

### Sample data
- Added `samples/` directory with 12 labeled synthetic email examples (4 legitimate, 5 phishing, 3 malicious)

### Bug fixes (from prior QA audit)
- Fixed hardcoded `total_feature_dimensions: 10020` in inference engine — now dynamically reads from vectorizer
- Fixed whitespace-only input misclassification — now returns `UNKNOWN` instead of false positive
- Narrowed `except Exception: pass` in URL TLD parsing to `except (ValueError, AttributeError)` with debug logging
- Applied `html.escape()` to all user-controlled strings rendered in HTML output
- Fixed footer test count: 71/71 (was incorrectly showing 58/58)

### Security hardening
- Enabled `enableCORS = true` and `enableXsrfProtection = true` in `.streamlit/config.toml`
- Verified zero secrets in repository

### Testing
- Added `tests/test_qa_regression.py` with 13 regression tests covering XSS, boundaries, injections, and performance
- All 71 tests passing · 76.77% statement coverage

---

## Previous development history

The following is a summary of the development phases that produced the initial codebase. Detailed commit history is in the git log.

- **Initial release** — Training pipeline, 3-class classification, basic Streamlit interface
- **v0.2** — Security feature extractor (20 features), typosquatting detection, 58-test suite
- **v0.3** — UI redesign: light theme, executive layout, token attribution, NLP research studio view
- **v0.4** — Visual threat highlighting, scam/spam detection subcategories, clear button, session state fixes
- **v0.5** — QA audit: 13 regression tests added, XSS hardening, whitespace fix, feature dimension fix
