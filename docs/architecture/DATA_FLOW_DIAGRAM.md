# Data Flow Diagram: Phishing Classification Pipeline

## 1. Overview

This document describes the end-to-end data processing workflow for the phishing classification pipeline from the perspective of a senior ML engineer and solutions architect. The goal is to make the movement, transformation, and consumption of data fully transparent for engineering review, audit, and future extension.

### 1.1 Nature of the Phishing Dataset

The pipeline operates on a **structured phishing website dataset** stored in a SQLite database (`phishing.db`) with ~10,500 rows. Each row represents a single website snapshot and includes:

- **Numerical features**: Code and layout statistics such as `LineOfCode`, `LargestLineLength`, counts of redirects, popups, iframes, images, internal/external references, robots directives, responsiveness scores, and domain age.
- **Categorical features**: `Industry` and `HostingProvider`, representing high-level business segment and infrastructure provider.
- **Target label**: Binary indicator (`label`) specifying whether the site is phishing (malicious) or legitimate.

Although this implementation works on structured features rather than raw HTML, the **design is aligned with classical text-based phishing detection** where URLs, DOM content, and page text are vectorized (e.g., TF-IDF, character n-grams) into numeric feature spaces.

### 1.2 Role of Preprocessing in the ML Pipeline

In a classical ML pipeline, preprocessing bridges the gap between noisy, heterogeneous raw data and the clean, numerical representations required by ML algorithms. In this system, preprocessing:

- **Improves data quality**: Handles missing values, enforces consistent types, and resolves anomalous values.
- **Reduces noise**: Normalizes skewed distributions, mitigates outliers, and encodes categorical variables in a model-friendly way.
- **Normalizes representation**: Applies robust scaling to numerical features and one-hot encoding to categorical features; in a text-centric pipeline, this is where **tokenization, lowercasing, stopword removal, and TF-IDF/vectorization** would live.
- **Supports reproducibility**: Encapsulates all transformations in a single, serializable preprocessing pipeline.

From an enterprise standpoint, robust preprocessing is the first line of defense against **data drift, schema changes, and garbage-in–garbage-out failures**.

---

## 2. Data Stages

This section outlines the logical stages that data passes through from raw storage to model-ready inputs and final outputs.

### 2.1 Ingestion (SQLite → pandas)

- **Source**: SQLite database `data/phishing.db`, table `phishing_data`.
- **Mechanism**: `DataLoader` connects via `sqlite3`, discovers the main table, and reads it into a pandas `DataFrame`.
- **Outcome**: A structured in-memory representation of the dataset, ready for cleaning and feature operations.

### 2.2 Cleaning (Null Handling, URL/Text Cleanup Analogue)

In this structured variant of the phishing problem, most cleaning is **numerical/statistical** rather than raw text manipulation, but conceptually it maps to the same concerns:

- **Null handling**:
  - Detect columns with missing values (notably `LineOfCode`).
  - Create indicator feature `LineOfCode_Missing` to capture informative missingness.
  - Impute missing numerical values with the median; impute categorical values with the mode.
- **Outlier-aware normalization preparation**:
  - EDA has shown heavy skew and outliers in several numerical features → later handled via `RobustScaler`.
- **Text/URL cleanup analogue** (how this would look for raw URLs/HTML):
  - Normalizing casing (lowercasing URLs and text).
  - Stripping tracking parameters, session IDs, or irrelevant query components.
  - Removing obvious noise tokens and HTML boilerplate if page content is used.

### 2.3 Feature Engineering (Tabular & Text-Vectorization Friendly Design)

**Current implementation (tabular)**:

- **Indicator engineering**:
  - `LineOfCode_Missing` encodes whether source code length is unknown.
- **Categorical encoding**:
  - `Industry` and `HostingProvider` are transformed via **OneHotEncoder** with `drop='first'` to avoid multicollinearity.
- **Scaling & normalization**:
  - Numerical features are scaled using **RobustScaler**, which centers data at the median and scales by IQR, making models less sensitive to extreme values.

**Text-centric extension (conceptual)**:

If raw URLs, HTML, or email body content are ingested, this stage is where you would add:

- **Tokenization & normalization**: Lowercasing, stripping punctuation, removing stopwords.
- **Character- or word-level features**: N-grams, domain token patterns, suspicious keyword counts.
- **Vectorization**:
  - **TF-IDF** over URL tokens, HTML tags, or email text.
  - **Hashing trick** for high-cardinality features.
  - Embeddings-based representations (e.g., averaging pre-trained vectors) for more advanced variants.

### 2.4 Splitting (Train/Test)

- **Mechanism**: `train_test_split` with stratification on the `label`.
- **Ratios**: Default 80% train / 20% test (configurable via `config.yaml`).
- **Guarantees**:
  - Preserves class distribution between train and test.
  - Uses a fixed `random_state` for reproducibility.

### 2.5 Model Consumption

- **Input to models**: A dense numeric matrix where all features (engineered & encoded) have been transformed into a common vector space.
- **Consumers**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Expectations**:
  - No missing values.
  - Normalized feature scales (especially important for linear models).
  - Stable column ordering and schema between train and inference.

### 2.6 Output Generation (Metrics, Persisted Artifacts)

- **Metrics**: 11 evaluation metrics per model, including accuracy, precision, recall, F1, specificity, FPR, FNR, balanced accuracy, MCC, ROC-AUC, PR-AUC.
- **Tabular outputs**:
  - Results comparison `DataFrame` for all models.
  - Classification reports and confusion matrices.
- **Artifacts** (optional, config-driven):
  - Serialized models (e.g., `RandomForest_model.pkl`).
  - Metadata JSON (hyperparameters, metrics, timestamps).
  - Serialized preprocessing pipeline (for consistent production inference).

---

## 3. Mermaid Diagram: High-Level Data Flow

```mermaid
flowchart LR
    subgraph Source
        DB[(SQLite<br/>phishing.db)]
    end

    subgraph Ingestion
        INJ[Data Loader<br/>(SQLite → pandas)]
    end

    subgraph Cleaning
        CLEAN[Null Handling<br/>Type Checks<br/>Basic Sanitization]
    end

    subgraph FeatureEng[Feature Engineering]
        FE1[Indicator Features<br/>(e.g., LineOfCode_Missing)]
        FE2[Scaling & Encoding<br/>(RobustScaler, OneHotEncoder)]
        FE3[Text Vectorization*<br/>(TF-IDF, n-grams)]
    end

    subgraph Split
        SPLIT[Stratified<br/>Train/Test Split]
    end

    subgraph Models
        M1[LogReg]
        M2[RandomForest]
        M3[GradientBoost]
        M4[XGBoost]
    end

    subgraph Outputs
        METRICS[Evaluation Metrics]
        ARTS[Persisted Artifacts]
    end

    DB --> INJ --> CLEAN --> FE1 --> FE2 --> FE3
    FE3 --> SPLIT
    SPLIT -->|Train| M1
    SPLIT -->|Train| M2
    SPLIT -->|Train| M3
    SPLIT -->|Train| M4

    SPLIT -->|Test| M1
    SPLIT -->|Test| M2
    SPLIT -->|Test| M3
    SPLIT -->|Test| M4

    M1 --> METRICS
    M2 --> METRICS
    M3 --> METRICS
    M4 --> METRICS

    M1 --> ARTS
    M2 --> ARTS
    M3 --> ARTS
    M4 --> ARTS

    note right of FE3
      *Text vectorization is
      conceptual for future
      URL/HTML-based features
    end
```

---

## 4. Explanation of Each Stage

For each stage, we focus on inputs, outputs, rationale, and enterprise concerns.

### 4.1 Ingestion

- **What enters**:
  - SQLite database file `data/phishing.db` with table `phishing_data`.
- **What exits**:
  - pandas `DataFrame` with all raw columns, including `label`.
- **Why this stage exists**:
  - Provides a clear, typed contract between storage and processing.
  - Decouples storage technology (SQLite today, warehouse/feature store tomorrow) from downstream logic.
- **Enterprise concerns**:
  - **Auditability**: Ingestion logs include table name, row count, and schema.
  - **Reproducibility**: Exact snapshot (file path + timestamp + row count) can be recorded.
  - **Lineage**: “Dataset vX was pulled from `phishing.db` at T with schema S” is reconstructible.

### 4.2 Cleaning

- **What enters**:
  - Raw `DataFrame` from ingestion (including missing values and potential anomalies).
- **What exits**:
  - Cleaned `DataFrame` with:
    - Missing values handled.
    - New indicators (e.g., `LineOfCode_Missing`).
    - Consistent dtypes for numerical and categorical fields.
- **Why this stage exists**:
  - Ensures downstream models are not learning artifacts from missingness patterns or inconsistent encodings.
  - Provides a controlled place to apply text/URL sanitation in a text-heavy variant (e.g., stripping tracking parameters, normalizing domains).
- **Enterprise concerns**:
  - **Auditability**: Logs capture which columns had missing values and how they were imputed.
  - **Reproducibility**: Imputation strategies and constants (e.g., medians) are deterministic and can be persisted.
  - **Lineage**: You can trace which raw fields produced which cleaned columns.

### 4.3 Feature Engineering

- **What enters**:
  - Cleaned `DataFrame` with mixed numerical and categorical variables.
- **What exits**:
  - Numerical feature matrix suitable for direct consumption by scikit-learn and XGBoost models.
- **Why this stage exists**:
  - Bridges the representation gap between rich, human meaning (industry, host, counts, and—if extended—URLs/text) and the vector spaces models consume.
  - Encodes domain insights (e.g., missing `LineOfCode` is itself a signal).
- **Enterprise concerns**:
  - **Auditability**: The exact transformation graph (indicators, encoders, scalers, TF-IDF) should be serialized (as in the `ColumnTransformer` pipeline) and versioned.
  - **Reproducibility**: Fitted transformers are persisted so that production inference uses the same mappings.
  - **Lineage**: Feature store–style lineage can map “feature Xv3 came from raw column Y via pipeline Z.”

### 4.4 Splitting

- **What enters**:
  - Fully engineered feature matrix and corresponding labels.
- **What exits**:
  - `X_train`, `X_test`, `y_train`, `y_test` with preserved class distribution.
- **Why this stage exists**:
  - Separates training and evaluation responsibilities and prevents leakage.
  - Provides a stable evaluation bed to compare models and future experiments.
- **Enterprise concerns**:
  - **Auditability**: Logs and seeds allow reproducing the exact split.
  - **Reproducibility**: `random_state` is fixed and documented.
  - **Lineage**: Train vs test lineage is critical for compliance (e.g., “no production data leaked into training”).

### 4.5 Model Consumption

- **What enters**:
  - `X_train` / `X_test` as dense numeric matrices.
- **What exits**:
  - Trained models, per-model predictions, and probability scores.
- **Why this stage exists**:
  - Encapsulates learning algorithms and hyperparameter search.
  - Provides a clean boundary where feature engineering ends and modeling begins.
- **Enterprise concerns**:
  - **Auditability**: Hyperparameters, random seeds, and training durations are logged.
  - **Reproducibility**: Re-running the pipeline with the same config yields the same trained model (up to algorithmic determinism).
  - **Lineage**: Model versions can be tied back to exact data, features, and config.

### 4.6 Output Generation

- **What enters**:
  - Predictions and probability scores for all models on the test set.
- **What exits**:
  - Metrics tables, reports, confusion matrices, and (optionally) persisted models and preprocessing pipelines.
- **Why this stage exists**:
  - Translates raw model outputs into decision-grade signals for stakeholders.
  - Forms the contract between offline training and online serving systems.
- **Enterprise concerns**:
  - **Auditability**: All metrics are versioned and attributable to model/data versions.
  - **Reproducibility**: You can re-run evaluation on the same test set anytime.
  - **Lineage**: Downstream dashboards or alerts can trace their values back to specific model runs.

---

## 5. Key Considerations

### 5.1 Handling Imbalanced Data

- The current dataset is roughly balanced, but real-world phishing traffic is often **highly imbalanced** (phishing is rare relative to legitimate sites).
- Considerations:
  - Use **stratified splits** to maintain class ratios.
  - Monitor **recall** and **FNR** carefully—missing phishing sites can be costly.
  - For imbalanced extensions, incorporate techniques like **class weighting**, **oversampling (SMOTE)**, or **focal loss** in more advanced models.

### 5.2 Normalization Challenges with Phishing Texts

In a text-heavy pipeline (URLs, HTML, email content):

- **Heterogeneous structure**: URLs, DOM trees, and natural language text behave differently and require distinct normalization strategies.
- **Evasion tactics**: Attackers deliberately inject noise (random subdomains, homoglyphs, obfuscation) to bypass naive normalization.
- **Loss of signal**: Over-aggressive normalization (e.g., stripping all special characters) can remove signals like `@`, excessive `//`, or suspicious TLDs.

Design guidance:

- Treat **URL structure**, **HTML tags**, and **visible text** as separate feature families.
- Normalize **within** those families, not across them.
- Preserve suspicious character patterns and token frequencies where they meaningfully contribute to detection.

### 5.3 Feature Sparsity

- TF-IDF or n-gram vectorization produces **high-dimensional, sparse** feature matrices.
- Enterprise concerns:
  - Memory footprint for training and inference.
  - Model choice: linear models and tree-based methods handle sparsity differently.
  - Serving infrastructure: ensuring your inference stack supports sparse matrices efficiently.

Mitigation strategies:

- Use **dimensionality reduction** (e.g., truncated SVD) for extreme sparsity.
- Favor models that naturally exploit sparse structure (e.g., linear models with L1/L2 regularization, gradient boosting with sparse support).
- Apply feature selection based on mutual information or model-based importance.

### 5.4 Generalization Concerns

For phishing detection, **data drift** is the norm, not the exception:

- Hosting providers and industries change over time.
- Attackers adapt patterns once detection rules/models are known.
- New TLDs and hosting platforms emerge.

Mitigation:

- Design the dataflow so that **retraining** is a first-class operation (new data → same preprocessing → updated models).
- Monitor **performance metrics over time** (e.g., weekly ROC-AUC, FNR in production).
- Periodically **re-evaluate feature importance** to detect obsolete or brittle features.

---

## 6. Extensibility

The current dataflow is intentionally conservative and focused on tabular features, but it is architected to scale into more advanced enterprise scenarios.

### 6.1 Streaming Ingestion

- Replace batch SQLite ingestion with a streaming source:
  - Kafka topics with URL and HTML snapshots.
  - Log-based ingestion from web gateways or email systems.
- Adaptations:
  - Wrap the current preprocessing and feature engineering steps into a **stateless or stateful stream processor** (e.g., Flink, Spark Structured Streaming).
  - Maintain **sliding windows** or **sessionized features** for near-real-time risk scoring.

### 6.2 Multi-Class Classification

- Extend `label` beyond binary to include categories such as:
  - Credential harvesting
  - Malware delivery
  - Brand impersonation
  - Benign/test domains
- Dataflow impact:
  - Same ingestion and cleaning stages.
  - Feature engineering extended with additional behavior- or content-based features.
  - Models reconfigured for multi-class outputs (e.g., softmax for linear models, multi-class XGBoost).

### 6.3 Feature Stores

- Integrate with a **feature store** (Feast, SageMaker Feature Store, Vertex AI Feature Store):
  - Ingestion becomes: `entity_ids → feature store lookup` instead of SQLite queries.
  - Feature engineering logic is registered as **feature views** with versioned transformations.
- Benefits:
  - Consistent features between training and serving.
  - Centralized governance for feature definitions and access control.
  - Time-travel capabilities for backtesting (point-in-time correct features).

### 6.4 MLOps Pipelines (SageMaker, Vertex, Azure ML)

- The current Python modules (`data_loader`, `preprocessor`, `model_trainer`, `model_evaluator`) can be wrapped as **steps in a managed pipeline**:

  - **AWS SageMaker Pipelines**:
    - `ProcessingStep`: data ingestion + cleaning + feature engineering.
    - `TrainingStep`: model training with built-in algorithms or custom containers.
    - `ConditionStep`: compare metrics and branch to deployment only if thresholds are met.
    - `RegisterModel`: push final artifacts to SageMaker Model Registry.

  - **Google Vertex AI Pipelines**:
    - Use Kubeflow/Vertex components for each stage.
    - Store datasets and models in Vertex Dataset/Model resources.

  - **Azure ML Pipelines**:
    - Register dataset, build `CommandComponent` for each stage.
    - Use MLTable and Feature Store integration for production-grade lineage.

- Dataflow impact:
  - The logical stages (ingestion, cleaning, feature engineering, splitting, training, evaluation, output) remain unchanged.
  - The **orchestration layer** moves from local `pipeline.py` and `run.sh` into a managed orchestrator with experiment tracking, approvals, and automated deployment.

---

**Summary**

The dataflow described here is intentionally **modular, auditable, and extensible**. It treats data as a first-class asset, ensuring that every transformation—from SQLite ingestion to model-ready vectors—is explainable, reproducible, and suitable for enterprise-scale evolution.

This foundation comfortably supports both the current structured-feature pipeline and future, more sophisticated phishing detection systems that ingest raw URLs, HTML, and email content, while remaining compatible with modern feature stores and managed MLOps platforms.
