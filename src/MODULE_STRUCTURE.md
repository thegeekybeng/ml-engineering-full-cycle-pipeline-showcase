# Module Structure Plan

## Planned Modules

1. **config.py** - Configuration management (YAML/JSON/env vars/CLI)
2. **data_loader.py** - SQLite data loading and preparation
3. **preprocessor.py** - Data preprocessing pipeline (RobustScaler, OneHotEncoder, etc.)
4. **model_trainer.py** - Model training logic for all algorithms
5. **model_evaluator.py** - Model evaluation with comprehensive metrics
6. **visualizer.py** - Visualization utilities (optional, can be integrated)
7. **pipeline.py** - Main pipeline orchestration

## Conversion Mapping

From Notebook Steps → Python Modules:

- Step 1 (Imports) → Distributed across modules
- Step 2 (Data Loading) → `data_loader.py`
- Step 3 (Preprocessing) → `preprocessor.py`
- Step 4 (Train-Test Split) → `data_loader.py` or `preprocessor.py`
- Step 5 (Model Implementation) → `model_trainer.py`
- Step 6 (Model Evaluation) → `model_evaluator.py`
- Step 7 (Model Comparison) → `model_evaluator.py` or `pipeline.py`
- Step 8 (Final Assessment) → `pipeline.py`

