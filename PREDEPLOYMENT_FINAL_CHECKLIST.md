# Release Readiness Checklist

This checklist is intended for teams preparing this phishing detection pipeline
for internal or public release. It focuses on engineering quality, repository
hygiene, and operational readiness.

---

## 1. Repository & Source Control

- ✅ All critical files committed to git (`src/`, `run.sh`, `requirements.txt`, `README.md`, `config/`, `notebooks/` as needed).
- ✅ `.gitignore` configured to exclude build artifacts, caches, and local-only files.
- ✅ No secrets or credentials committed (review config and notebooks).
- ✅ Branch protection and code review policies configured in the hosting platform.

---

## 2. Code Quality & Structure

- ✅ Clear module boundaries for configuration, data loading, preprocessing, training, evaluation, and persistence.
- ✅ Error handling in `pipeline.py` surfaces tracebacks and exits with appropriate status codes.
- ✅ `run.sh` performs basic environment validation and reports missing dependencies.
- ✅ Code style is consistent and readable for a mixed team of ML and software engineers.

---

## 3. Documentation

- ✅ `README.md` describes the system, how to run it, and how it fits into a broader ML/MLOps landscape.
- ✅ Architecture documents under `docs/architecture/` explain system components, data flow, and operational considerations.
- ✅ Higher-level summaries under `docs/summaries/` provide quick context for non-technical stakeholders.
- ✅ Notebooks (`notebooks/`) include narrative context appropriate for internal analysis and do not rely on proprietary datasets.

---

## 4. Configuration & Environment

- ✅ `config/config.yaml` is free of environment-specific secrets and uses placeholders (e.g., `YOUR_STORAGE_ENDPOINT`).
- ✅ Configuration precedence (config file, environment, CLI) is documented and tested.
- ✅ Example environment variables and command-line invocations are included in the docs.
- ✅ A reproducible environment definition exists (e.g., `requirements.txt`, optional Dockerfile).

---

## 5. CI/CD & Automation (Optional but Recommended)

- ✅ Automated tests (`pytest`) run in CI and cover key components (config, data loading, preprocessing).
- ✅ The pipeline can be executed non-interactively in CI using `run.sh` or `python src/pipeline.py`.
- ✅ Logs and metrics from CI runs are captured and retained for troubleshooting.
- ✅ A path is defined for integrating with experiment tracking and model registries.

---

## 6. Operational Readiness

- ✅ Logging is sufficient to diagnose failures at each stage of the pipeline.
- ✅ Metric outputs for each model are clear and can be exported for dashboards.
- ✅ Model and preprocessor artifacts can be persisted and reloaded reliably.
- ✅ A plan exists for monitoring model performance and data drift in downstream deployments.

---

## 7. Final Verification Commands

Run these commands locally to verify repository state:

```bash
git status
git ls-files | grep -E "src/|run.sh|requirements.txt|README.md|config.yaml|eda.ipynb"
```

Expected outcome: all critical files are tracked, with no uncommitted changes before creating a release tag.

---

This checklist is designed to support **production-quality releases** of the
phishing detection pipeline into internal platforms or public repositories.

