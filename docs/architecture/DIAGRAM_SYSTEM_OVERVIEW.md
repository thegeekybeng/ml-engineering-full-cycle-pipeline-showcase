# System Overview Diagrams — Phishing Classification Pipeline

This document contains Mermaid diagrams capturing the core architecture, data flow, and pipeline sequence for the phishing classification ML system.

---

## 1. System Architecture Diagram

```mermaid
graph TB
    subgraph Client
        NB[EDA / MLP Notebooks]
        API[Inference API]
        Batch[Batch Jobs]
    end

    subgraph Config[Configuration & Orchestration]
        CFG[Config Manager]
        PIPE[Pipeline Orchestrator]
    end

    subgraph DataLayer[Data Layer]
        DB[(SQLite<br/>data/phishing.db)]
        DL[DataLoader]
    end

    subgraph FeatureLayer[Feature & Preprocessing Layer]
        PREP[Preprocessor]
    end

    subgraph ModelLayer[Model Lifecycle Layer]
        TRAIN[ModelTrainer]
        EVAL[ModelEvaluator]
        PERSIST[ModelPersistence]
    end

    CFG --> DL
    CFG --> PREP
    CFG --> TRAIN
    CFG --> EVAL
    CFG --> PERSIST

    PIPE --> DL
    PIPE --> PREP
    PIPE --> TRAIN
    PIPE --> EVAL
    PIPE --> PERSIST

    DB --> DL --> PREP --> TRAIN --> EVAL --> PERSIST

    API --> PERSIST
    Batch --> PERSIST
    NB --> PIPE
```

---

## 2. Data Flow Diagram

```mermaid
flowchart LR
    A[Local SQLite DB<br/>data/phishing.db] --> B[DataLoader<br/>(load & basic cleaning)]
    B --> C[Feature/Target Split]
    C --> D[Preprocessor<br/>(RobustScaler + OneHotEncoder)]
    D --> E[Train/Test Split<br/>(stratified)]
    E --> F[ModelTrainer<br/>(train portfolio)]
    F --> G[ModelEvaluator<br/>(compute metrics)]
    G --> H[Best Model Selection]
    H --> I[ModelPersistence<br/>(save models & metadata)]

    subgraph Config
        CFGYAML[config/config.yaml]
    end

    CFGYAML --> B
    CFGYAML --> D
    CFGYAML --> F
    CFGYAML --> G
    CFGYAML --> I
```

---

## 3. Pipeline Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User / Engineer
    participant PIPE as Pipeline Orchestrator
    participant CFG as Config Manager
    participant DL as DataLoader
    participant PREP as Preprocessor
    participant TRAIN as ModelTrainer
    participant EVAL as ModelEvaluator
    participant PERSIST as ModelPersistence

    User->>PIPE: python src/pipeline.py --config config/config.yaml
    PIPE->>CFG: Load configuration
    CFG-->>PIPE: Config object

    PIPE->>DL: load_and_prepare(config)
    DL-->>PIPE: X, y, feature_types

    PIPE->>PREP: split_data(X, y)
    PREP-->>PIPE: X_train, X_test, y_train, y_test

    PIPE->>PREP: fit_transform(X_train, feature_types)
    PREP-->>PIPE: X_train_transformed
    PIPE->>PREP: transform(X_test)
    PREP-->>PIPE: X_test_transformed

    PIPE->>TRAIN: train_all(X_train_transformed, y_train)
    TRAIN-->>PIPE: trained_models

    PIPE->>EVAL: evaluate_all(trained_models, X_test_transformed, y_test)
    EVAL-->>PIPE: evaluation_results
    PIPE->>EVAL: create_results_dataframe()
    EVAL-->>PIPE: results_df
    PIPE->>EVAL: select_best_model(trained_models)
    EVAL-->>PIPE: best_model_name, best_model, best_results

    alt save_models enabled
        PIPE->>PERSIST: save_all_models(trained_models, evaluation_results)
        PERSIST-->>PIPE: artifact paths
    end

    PIPE-->>User: Final summary (best model, metrics, artifacts)
```

All diagrams assume a **config-driven pipeline** where behavior is controlled via `config/config.yaml` and optionally overridden by environment variables or CLI arguments.
