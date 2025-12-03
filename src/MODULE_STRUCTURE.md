# Module Structure Plan

## Planned Modules

1. **config.py** - Configuration management (YAML/JSON/env vars/CLI)
2. **data_loader.py** - Data loading and preparation
3. **preprocessor.py** - Data preprocessing pipeline (RobustScaler, OneHotEncoder, etc.)
4. **model_trainer.py** - Model training logic for all algorithms
5. **model_evaluator.py** - Model evaluation with comprehensive metrics
6. **visualizer.py** - Visualization utilities (optional, can be integrated)
7. **pipeline.py** - Main pipeline orchestration

## Logical Mapping

Notebook-style exploration is mapped into production modules as follows:

- Data loading and basic quality checks → `data_loader.py`
- Preprocessing and feature engineering → `preprocessor.py`
- Model implementation and training → `model_trainer.py`
- Evaluation and comparison → `model_evaluator.py` / `pipeline.py`
- Reporting and orchestration → `pipeline.py`

