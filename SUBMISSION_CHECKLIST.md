# AIAP Task 2 Submission Checklist
## GitHub Repository Submission Guide

**Student**: Yeo Meng Chye Andrew  
**Email**: andrew.yeo.mc@gmail.com  
**Date**: Generated automatically

---

## ✅ Pre-Submission Verification

### 1. Repository Naming Convention

**Required Format**: `aiap22-<full name (as in NRIC) separated by dashes>-<last 4 characters of NRIC>`

**Your Name**: Yeo Meng Chye Andrew  
**NRIC Last 4 Characters**: 733H  
**Repository Name**: `aiap22-yeo-meng-chye-andrew-733H`

✅ **Repository Name Confirmed**: `aiap22-yeo-meng-chye-andrew-733H`

---

### 2. Required Files and Folders

| Item | Status | Location | Notes |
|------|--------|----------|-------|
| **`.github/` folder** | ✅ Present | `.github/` | GitHub Actions workflows (from template) |
| **`src/` folder** | ✅ Present | `src/` | Python modules (7 files) |
| **`run.sh`** | ✅ Present | Root | Executable bash script |
| **`requirements.txt`** | ✅ Present | Root | Python dependencies |
| **`config.yaml`** | ✅ Present | Root | Configuration file |
| **`README.md`** | ✅ Present | Root | Documentation (all sections a-i) |

---

### 3. Repository Structure Verification

```
aiap22-yeo-meng-chye-andrew-XXXX/
│
├── .github/                    ✅ GitHub Actions (from template)
│   └── workflows/             ✅ Pipeline execution scripts
│
├── src/                        ✅ Python modules folder
│   ├── __init__.py            ✅ Package initialization
│   ├── config.py              ✅ Configuration management
│   ├── data_loader.py         ✅ SQLite data loading
│   ├── preprocessor.py        ✅ Preprocessing pipeline
│   ├── model_trainer.py       ✅ Model training
│   ├── model_evaluator.py     ✅ Model evaluation
│   └── pipeline.py            ✅ Main pipeline orchestrator
│
├── run.sh                      ✅ Executable bash script
├── requirements.txt            ✅ Python dependencies
├── config.yaml                 ✅ Configuration file
└── README.md                   ✅ Documentation (all sections a-i)
```

---

### 4. File Permissions

- ✅ `run.sh` is executable (`chmod +x run.sh`)
- ✅ All Python files are readable
- ✅ All configuration files are readable

---

### 5. Code Quality Checks

- ✅ All code in Python scripts (`.py` files)
- ✅ No notebook dependencies in pipeline execution
- ✅ Modular, reusable code structure
- ✅ Configuration-driven (no hard-coded values)
- ✅ Proper error handling

---

## 📋 GitHub Repository Setup Steps

### Step 1: Create GitHub Account
- [ ] Create GitHub account using email: `andrew.yeo.mc@gmail.com`
- [ ] Verify email address

### Step 2: Download Repository Template
- [ ] Download template from: https://techassessment.blob.core.windows.net/aiap22-assessment-data/aiap22-NAME-NRIC.zip
- [ ] Extract the template (contains `.github` folder)

### Step 3: Create Private Repository
- [ ] Create a **private** repository on GitHub
- [ ] Name it: `aiap22-yeo-meng-chye-andrew-733H`
- [ ] ⚠️ **CRITICAL**: Strictly follow naming convention or risk penalization

### Step 4: Upload Your Code
- [ ] Copy your submission files to the repository:
  - `src/` folder (all Python modules)
  - `run.sh` (executable)
  - `requirements.txt`
  - `config.yaml`
  - `README.md`
  - `.github/` folder (from template)
- [ ] Ensure all files are in the **main branch**

### Step 5: Add Collaborator
- [ ] Go to repository Settings → Collaborators
- [ ] Add collaborator:
  - **Username**: `AISG-AIAP`
  - **Email**: `aiap-internal@aisingapore.org`
- [ ] Grant appropriate access (read access should be sufficient)

### Step 6: Verify GitHub Actions
- [ ] Go to repository → Actions tab
- [ ] Verify GitHub Actions workflow is present
- [ ] Manually trigger the pipeline to test execution
- [ ] Ensure pipeline executes successfully:
  - Installs dependencies from `requirements.txt`
  - Executes `run.sh`
  - Completes without errors

### Step 7: Final Verification
- [ ] Verify repository is **private**
- [ ] Verify repository name follows convention exactly
- [ ] Verify collaborator `AISG-AIAP` is added
- [ ] Verify all files are in **main branch**
- [ ] Test `run.sh` execution locally
- [ ] Verify `README.md` contains all sections (a-i)

### Step 8: Submit Repository Link
- [ ] Complete the Google form: https://forms.gle/z2USSMNrxrTHsvwdA
- [ ] Provide your repository URL
- [ ] ⚠️ **NOTE**: You can still make changes after submitting the form (until deadline)

---

## ⚠️ Important Warnings

### Penalization Risks

1. **Repository Naming**: 
   - ❌ Wrong format → **PENALIZATION**
   - ✅ Must be: `aiap22-yeo-meng-chye-andrew-733H`

2. **Repository Visibility**:
   - ❌ Public repository → **PENALIZATION**
   - ✅ Must be **private**

3. **Branch**:
   - ❌ Code not in main branch → **PENALIZATION**
   - ✅ Must be in **main branch**

4. **Collaborator**:
   - ❌ Missing collaborator `AISG-AIAP` → **PENALIZATION**
   - ✅ Must add `AISG-AIAP` as collaborator

5. **Post-Deadline Changes**:
   - ❌ Making changes after deadline → **PENALIZATION**
   - ✅ Do NOT modify repository after deadline

---

## 📁 Files to Include in Submission

### ✅ Required Files (Include)
- `src/` folder (all Python modules)
- `run.sh`
- `requirements.txt`
- `config.yaml`
- `README.md`
- `.github/` folder (from template)

### ❌ Files to Exclude (Do NOT Include)
- `venv/` folder (virtual environment)
- `data/` folder (database will be downloaded automatically)
- `__pycache__/` folders (Python cache)
- `.DS_Store` files (macOS system files)
- Assessment reports (`.md` files except `README.md`):
  - `AIAP_EVALUATION_ASSESSMENT.md`
  - `ASSESSMENT_REPORT.md`
  - `PIPELINE_TEST_REPORT.md`
  - `PIPELINE_TEST_RESULTS.md`
  - `TASK2_FINAL_ASSESSMENT.md`
  - `TASK2_REQUIREMENTS_ANALYSIS.md`
  - `TASK2_SCORING_ASSESSMENT.md`
  - `SUBMISSION_CHECKLIST.md` (this file)
- Notebook files (optional - can include as reference but not required):
  - `eda.ipynb` (Task 1)
  - `mlp.ipynb` (development artifact, not used for execution)

---

## 🔍 Pre-Submission Testing

### Test 1: Local Execution
```bash
# Test run.sh execution
./run.sh --model RandomForest

# Expected: Pipeline executes successfully
```

### Test 2: GitHub Actions
- [ ] Push code to GitHub
- [ ] Go to Actions tab
- [ ] Manually trigger workflow
- [ ] Verify execution completes successfully

### Test 3: Dependency Installation
```bash
# Test requirements.txt
pip install -r requirements.txt

# Expected: All dependencies install successfully
```

---

## ✅ Final Checklist Before Submission

- [ ] Repository name follows convention: `aiap22-yeo-meng-chye-andrew-733H`
- [ ] Repository is **private**
- [ ] All code is in **main branch**
- [ ] Collaborator `AISG-AIAP` is added
- [ ] `src/` folder contains all 7 Python modules
- [ ] `run.sh` is executable and tested
- [ ] `requirements.txt` is correct and tested
- [ ] `config.yaml` is present
- [ ] `README.md` contains all sections (a-i)
- [ ] `.github/` folder is present (from template)
- [ ] GitHub Actions workflow executes successfully
- [ ] Google form submitted: https://forms.gle/z2USSMNrxrTHsvwdA

---

## 📝 Submission Form Information

When completing the Google form, you will need:

1. **GitHub Repository URL**: `https://github.com/<your-username>/aiap22-yeo-meng-chye-andrew-733H`
2. **Repository Name**: `aiap22-yeo-meng-chye-andrew-733H`
3. **Your Email**: `andrew.yeo.mc@gmail.com`
4. **Full Name**: `Yeo Meng Chye Andrew`

---

## 🎯 Summary

**Status**: ✅ **READY FOR SUBMISSION**

All required files and folders are present and verified. Follow the steps above to:
1. Create GitHub repository with correct naming
2. Upload your code
3. Add collaborator
4. Test GitHub Actions
5. Submit via Google form

**Remember**: 
- ⚠️ Repository name must follow convention exactly
- ⚠️ Repository must be private
- ⚠️ Code must be in main branch
- ⚠️ Do NOT modify repository after deadline

---

**Good luck with your submission!** 🚀

