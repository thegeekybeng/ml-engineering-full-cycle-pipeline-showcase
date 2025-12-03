# Task 2 Requirements Analysis & Critical Violations

## 🚨 CRITICAL REQUIREMENT VIOLATIONS

### 1. **MAJOR VIOLATION: Interactive Notebook Usage**
**Requirement**: "Do not develop your MLP in an interactive notebook."

**Current Status**: ❌ **VIOLATION**
- MLP is currently implemented in `mlp.ipynb` (interactive notebook)
- This is explicitly forbidden in Task 2 requirements

**Penalization Risk**: **HIGH** - This is a fundamental requirement violation that could result in significant score deduction or disqualification.

**Required Action**: 
- Convert entire MLP from notebook to Python scripts (`.py` files)
- Organize into `src/` folder with proper module structure
- Ensure no notebook dependencies remain

---

### 2. **MISSING: `src/` Folder with Python Modules**
**Requirement**: "A folder named `src` containing Python modules/classes in `.py` format."

**Current Status**: ❌ **MISSING**
- No `src/` folder exists
- No Python modules/classes in `.py` format

**Penalization Risk**: **HIGH** - Core deliverable missing

**Required Structure**:
```
src/
├── __init__.py
├── data_loader.py          # Data loading and SQLite operations
├── preprocessor.py          # Preprocessing pipeline
├── model_trainer.py         # Model training logic
├── model_evaluator.py       # Model evaluation metrics
├── config.py                # Configuration management
└── pipeline.py             # Main pipeline orchestration
```

---

### 3. **MISSING: Configurability**
**Requirement**: "The pipeline should be easily configurable to enable easy experimentation of different algorithms and parameters as well as ways of processing data. You can consider the usage of a config file, environment variables, or command line parameters."

**Current Status**: ❌ **NOT CONFIGURABLE**
- No config file
- No environment variable support
- No command-line parameter support
- Hard-coded values throughout notebook

**Penalization Risk**: **MEDIUM-HIGH** - Reusability requirement not met

**Required Implementation**:
- Config file (YAML/JSON) for:
  - Model selection
  - Hyperparameters
  - Preprocessing choices
  - Evaluation metrics
- Command-line arguments for:
  - Model selection
  - Config file path
  - Output directory
- Environment variables for:
  - Database URL
  - Data paths
  - Logging levels

---

### 4. **INCOMPLETE: `run.sh` Script**
**Requirement**: "An executable bash script `run.sh` at the base folder of your submission to run the aforementioned modules/classes/scripts."

**Current Status**: ⚠️ **PLACEHOLDER ONLY**
- `run.sh` exists but is just a placeholder
- Doesn't execute any Python modules
- Doesn't call `src/` modules

**Penalization Risk**: **MEDIUM** - Deliverable incomplete

**Required Implementation**:
```bash
#!/bin/bash
# Should execute: python3 src/pipeline.py --config config.yaml
# Or similar structure
```

---

### 5. **MISSING: `README.md`**
**Requirement**: "A `README.md` file that sufficiently explains the pipeline design and its usage."

**Current Status**: ❌ **MISSING**
- No README.md file exists

**Penalization Risk**: **HIGH** - Core deliverable missing

**Required Sections**:
- a. Full name (as in NRIC) and email address
- b. Overview of submitted folder and folder structure
- c. Instructions for executing the pipeline and modifying parameters
- d. Description of logical steps/flow of the pipeline
- e. Overview of key findings from EDA (Task 1) and choices made
- f. Description of how features are processed (table format)
- g. Explanation of model choices
- h. Model evaluation and metrics explanation
- i. Deployment considerations

---

## 📋 REQUIREMENTS CHECKLIST

| Requirement | Status | Risk Level | Priority |
|------------|--------|------------|----------|
| **Python scripts (.py files)** | ❌ Missing | **CRITICAL** | **P0** |
| **`src/` folder with modules** | ❌ Missing | **CRITICAL** | **P0** |
| **No interactive notebook** | ❌ Violated | **CRITICAL** | **P0** |
| **Configurability (config/env/CLI)** | ❌ Missing | **HIGH** | **P0** |
| **`run.sh` executable script** | ⚠️ Incomplete | **MEDIUM** | **P0** |
| **`README.md` documentation** | ❌ Missing | **HIGH** | **P0** |
| **SQLite data fetching** | ✅ Present | Low | P1 |
| **`requirements.txt`** | ✅ Present | Low | P1 |

---

## 🎯 REQUIRED ACTIONS (Priority Order)

### **P0 - CRITICAL (Must Complete Immediately)**

1. **Convert Notebook to Python Modules**
   - Extract all code from `mlp.ipynb`
   - Organize into logical modules in `src/` folder
   - Remove notebook-specific code (display(), IPython magic, etc.)
   - Ensure proper Python module structure

2. **Create `src/` Folder Structure**
   ```
   src/
   ├── __init__.py
   ├── data_loader.py
   ├── preprocessor.py
   ├── model_trainer.py
   ├── model_evaluator.py
   ├── config.py
   └── pipeline.py
   ```

3. **Implement Configurability**
   - Create `config.yaml` or `config.json`
   - Implement config loader in `src/config.py`
   - Add command-line argument parsing (argparse)
   - Support environment variables

4. **Update `run.sh`**
   - Execute Python modules from `src/`
   - Pass configuration
   - Handle errors properly

5. **Create `README.md`**
   - Include all required sections (a-i)
   - Document folder structure
   - Provide execution instructions
   - Explain design decisions

### **P1 - HIGH (Should Complete)**

6. **Code Reusability**
   - Modular design
   - Clear separation of concerns
   - Reusable functions/classes
   - Proper error handling

7. **Testing**
   - Ensure `run.sh` works end-to-end
   - Test with different configurations
   - Verify all models can be selected

---

## 📊 PENALIZATION RISK ASSESSMENT

### **Current Score Estimate**: **0-30/100** (Severe Penalization)

**Breakdown**:
- **Code Structure (-50 points)**: No `src/` folder, no Python modules
- **Notebook Violation (-30 points)**: Explicitly forbidden format used
- **Configurability (-15 points)**: No configuration mechanism
- **Documentation (-5 points)**: Missing README.md
- **Script (-5 points)**: Incomplete run.sh

### **Potential Score After Fixes**: **85-100/100**

**If properly implemented**:
- ✅ Proper module structure (+50 points)
- ✅ No notebook violation (+30 points)
- ✅ Configurability (+15 points)
- ✅ Complete documentation (+5 points)
- ✅ Working run.sh (+5 points)

---

## 🏗️ RECOMMENDED ARCHITECTURE

### **Module Structure**

```python
# src/config.py
class Config:
    """Load configuration from YAML/JSON/env vars"""
    def __init__(self, config_path=None):
        # Load from config file, env vars, or defaults
        pass

# src/data_loader.py
class DataLoader:
    """Handle SQLite data loading"""
    def load_data(self, db_url):
        # SQLite operations
        pass

# src/preprocessor.py
class Preprocessor:
    """Data preprocessing pipeline"""
    def fit_transform(self, X):
        # RobustScaler, OneHotEncoder, etc.
        pass

# src/model_trainer.py
class ModelTrainer:
    """Train multiple models"""
    def train(self, model_name, X_train, y_train, config):
        # Model training logic
        pass

# src/model_evaluator.py
class ModelEvaluator:
    """Evaluate models with comprehensive metrics"""
    def evaluate(self, model, X_test, y_test):
        # All metrics calculation
        pass

# src/pipeline.py
class MLPipeline:
    """Main pipeline orchestration"""
    def run(self, config):
        # End-to-end pipeline execution
        pass

# main.py (or pipeline.py)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model', choices=['all', 'rf', 'xgb', ...])
    args = parser.parse_args()
    
    config = Config(args.config)
    pipeline = MLPipeline(config)
    pipeline.run()
```

### **Config File Example** (`config.yaml`)

```yaml
data:
  db_url: "https://techassessment.blob.core.windows.net/aiap22-assessment-data/phishing.db"
  test_size: 0.2
  random_state: 42

preprocessing:
  scaler: "RobustScaler"
  handle_missing: "median_imputation"
  create_indicator: true

models:
  - name: "RandomForest"
    enabled: true
    params:
      n_estimators: 100
      max_depth: 10
  - name: "XGBoost"
    enabled: true
    params:
      n_estimators: 100
      learning_rate: 0.1

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
    - mcc
  cross_validation:
    enabled: true
    cv_folds: 5
```

---

## ⚠️ CRITICAL WARNINGS

1. **Do NOT submit the notebook as the MLP implementation** - This will result in severe penalization
2. **Do NOT skip the `src/` folder structure** - This is a core deliverable
3. **Do NOT hard-code configurations** - Must be configurable
4. **Do NOT skip the README.md** - Required documentation

---

## 📝 NEXT STEPS

1. **Immediate**: Create `src/` folder structure
2. **Immediate**: Extract code from notebook into Python modules
3. **Immediate**: Implement configuration system
4. **Immediate**: Update `run.sh` to execute modules
5. **Immediate**: Create comprehensive `README.md`
6. **Testing**: Verify end-to-end execution
7. **Review**: Ensure all requirements met

---

**Last Updated**: Generated automatically  
**Status**: ⚠️ **CRITICAL ACTION REQUIRED**

