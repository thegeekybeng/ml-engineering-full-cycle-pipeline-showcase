# Model Card — Phishing Classification Pipeline

## 1. Model Overview

This repository implements a **full-cycle machine learning pipeline** for binary classification of phishing vs. legitimate websites using structured tabular features.  
The system is designed as an enterprise-ready reference implementation, illustrating how to build, evaluate, and operate a phishing detection model in a production environment.

Key characteristics:

- **Problem type**: Binary classification (phishing vs. legitimate)
- **Domain**: Web and email security / phishing detection
- **Input modality**: Structured tabular data derived from website content and metadata
- **Model portfolio**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost (configurable)
- **Primary selection metric**: Accuracy (on a balanced dataset), with secondary metrics including F1-score, ROC-AUC, and security-focused metrics such as FNR/FPR

The pipeline is written in Python using scikit-learn and XGBoost, with a config-driven architecture to support reproducibility and controlled experimentation.

---

## 2. Intended Use

### Primary Intended Use

- **Goal**: Automatically classify websites as *phishing* or *legitimate* based on structural and behavioral features.
- **Intended users**:
  - Security engineering teams building phishing detection services.
  - ML engineering teams integrating phishing risk scores into security workflows.
  - Architects evaluating reference designs for security-focused ML pipelines.
- **Intended deployment contexts**:
  - Batch scoring of URLs / website snapshots as part of offline analysis.
  - Near-real-time scoring within security services (e.g., proxy, secure email gateways, or SOC tooling) once the trained model is wrapped behind an API.

### Expected Operating Conditions

- Data distributions and feature definitions are consistent with the training data (e.g., similar feature ranges and semantics).
- The upstream data pipeline is responsible for:
  - Extracting and engineering the same set of features for inference as used during training.
  - Maintaining compatible schema and types (e.g., numeric vs. categorical).

---

## 3. Out-of-Scope Use Cases

The following use cases are **explicitly out of scope** for this pipeline and model configuration:

- **Deep content inspection**:
  - Direct analysis of raw HTML, email content, or images (e.g., screenshots of webpages).
  - NLP-based phishing detection of long-form text without appropriate feature engineering.
- **Non-phishing security problems**:
  - Malware classification, intrusion detection, spam detection (unless re-trained on appropriate features and labels).
- **High-assurance / regulated environments without additional controls**:
  - Situations requiring formal verification, certified explainability, or legally mandated fairness audits without further adaptation.
- **Use on data with significantly different distributions**:
  - For example, mobile-only traffic, IoT devices, or non-web protocols, unless the model is retrained on representative data.

Any deployment in these contexts should treat this pipeline as a **starting point only**, and require additional domain-specific modeling, evaluation, and controls.

---

## 4. Dataset Description

### Source and Structure

- **Storage format**: SQLite database (`phishing.db`).
- **Rows**: Each row corresponds to a website snapshot.
- **Columns**:
  - **Numerical features** (e.g.):
    - `LineOfCode`: approximate size of HTML/code.
    - `LargestLineLength`: length of the longest line.
    - `NoOfURLRedirect`, `NoOfSelfRedirect`: redirect counts.
    - `NoOfPopup`, `NoOfiFrame`, `NoOfImage`: counts of visual/structural elements.
    - `NoOfSelfRef`, `NoOfExternalRef`: link structure.
    - `Robots`, `IsResponsive`: simple behavioral/structural indicators.
    - `DomainAgeMonths`: domain age.
    - `LineOfCode_Missing`: engineered indicator for missing `LineOfCode`.
  - **Categorical features**:
    - `Industry`: high-level industry category for the site.
    - `HostingProvider`: infrastructure provider.
  - **Target**:
    - `label`: binary indicator (phishing vs. legitimate).

### Dataset Properties

- **Size**: ~10,500 records.
- **Class balance**: Approximately balanced between phishing and legitimate websites.
- **Data quality**:
  - One feature (`LineOfCode`) has structured missingness, which is explicitly modeled via an indicator variable plus median imputation.
  - Other features are generally complete.

### Data Limitations

- The dataset is **representative of a particular collection period and environment**.  
  New attack patterns, industries, or hosting providers may result in distribution shift.
- The dataset focuses on **structural and metadata features**, not full-content or behavioral signals; performance may differ in settings with significantly different feature generation pipelines.

---

## 5. Preprocessing Summary

The pipeline applies a consistent preprocessing strategy via a scikit-learn `ColumnTransformer`:

- **Train-test split**:
  - Stratified split to preserve class balance (default: 80% train / 20% test).
- **Numerical features**:
  - **Missing value handling**:
    - `LineOfCode`: median imputation plus `LineOfCode_Missing` indicator.
    - Other numeric features: median imputation where needed.
  - **Scaling**:
    - `RobustScaler` is used to reduce sensitivity to outliers and heavy tails, as identified in EDA.
- **Categorical features**:
  - One-hot encoding via `OneHotEncoder` with `drop='first'` to avoid multicollinearity.
  - `handle_unknown='ignore'` to ensure robustness when unseen categories appear at inference time.

All transformations are **fit on training data only**, then applied to validation and test data to avoid leakage.  
The same pipeline should be persisted and reused for inference to maintain feature alignment.

---

## 6. Model Architecture Summary

The pipeline supports a configurable portfolio of classical ML models for structured data:

- **Logistic Regression**:
  - Baseline linear model, useful for interpretability and quick iteration.
- **Random Forest**:
  - Ensemble of decision trees, robust to non-linear relationships and feature interactions.
- **Gradient Boosting**:
  - Sequential ensemble focusing on correcting previous errors; strong performance on structured tabular data.
- **XGBoost**:
  - Optimized gradient boosting implementation with built-in regularization; strong baseline for many tabular problems.

No deep learning architectures are included by default, as the target problem is well-served by tree-based and linear models over engineered features.

Model configurations (e.g., number of estimators, depth, learning rate) are controlled via `config/config.yaml` and can be tuned using optional hyperparameter search (e.g., `RandomizedSearchCV`).

---

## 7. Evaluation Metrics & Limitations

### Metrics

The pipeline computes an extensive metric suite for each model:

- **Core metrics**:
  - Accuracy, Precision, Recall, F1-score.
- **Security-focused metrics**:
  - False Positive Rate (FPR), False Negative Rate (FNR), Specificity.
- **Balanced metrics**:
  - Balanced Accuracy, Matthews Correlation Coefficient (MCC).
- **Threshold-independent metrics**:
  - ROC-AUC, PR-AUC.

For **model selection**, the default primary metric is **Accuracy** (given the dataset is approximately balanced), with F1-score, Balanced Accuracy, and MCC used as secondary checks.

### Limitations

- **Distribution shift**:
  - Performance may degrade if deployed in environments with significantly different traffic, industries, or hosting providers.
- **Feature coverage**:
  - The model is only as strong as the upstream feature engineering. If key phishing behaviors are not captured as features, they cannot be detected.
- **Latency and throughput**:
  - While classical models are generally fast, precise performance depends on deployment environment and concurrency requirements.
- **Threshold tuning**:
  - The pipeline uses default sklearn decision thresholds; production deployments may require custom thresholding to balance FNR vs. FPR according to business risk appetite.

---

## 8. Ethical Considerations

### Potential Harms

- **False negatives (undetected phishing)**:
  - May lead to credential theft, financial loss, or data breaches.
  - In production, FNR should be monitored closely, and models should be combined with other security controls (e.g., URL reputation, sandboxing).
- **False positives (benign sites flagged as phishing)**:
  - Can impact user experience and business operations (e.g., blocking legitimate partner sites).
  - SLOs for user-impacting false positives should be defined and observed.

### Mitigation Strategies

- Deploy the model as part of a **multi-layered defense**, not a single point of decision.
- Maintain **human-in-the-loop review** for high-risk decisions or uncertain predictions.
- Monitor model performance and **data drift** over time, retraining as needed.
- Ensure that training and evaluation cover diverse industries and providers to avoid systematic blind spots.

### Fairness Considerations

- While the dataset is not user-specific and does not contain direct PII, care should be taken to avoid unintended bias against specific industries or hosting providers.
- If used in regulated environments, additional fairness and bias analyses should be performed on domain-appropriate slices of the data.

---

## 9. Maintenance & Versioning

### Versioning

- Code is versioned via Git; configuration is stored in `config/config.yaml`.
- Model artifacts can be versioned by:
  - Naming conventions (e.g., `model_name_version_timestamp.pkl`).
  - Integration with a model registry such as MLflow, SageMaker Model Registry, or Vertex AI Model Registry.

### Maintenance Activities

- **Periodic retraining**:
  - Refresh the model with new data to capture evolving phishing patterns.
  - Validate retrained models with the same metric suite, plus any new business-specific KPIs.
- **Continuous evaluation**:
  - Monitor production metrics (accuracy, FNR/FPR, ROC-AUC) on fresh data.
  - Implement alerts when metrics degrade beyond acceptable thresholds.
- **Data pipeline validation**:
  - Regularly validate that upstream feature generation remains consistent with training-time assumptions (schema, ranges, distributions).

### Ownership and Operations

- The pipeline is intended to be owned jointly by:
  - **Security engineering / SOC**: defining risk thresholds and triage workflows.
  - **ML engineering / platform teams**: maintaining the pipeline, infrastructure, and MLOps integrations.

Clear ownership, SLAs, and SLOs should be defined before this model is promoted to a production environment.


