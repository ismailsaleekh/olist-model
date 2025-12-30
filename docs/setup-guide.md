# Environment Setup Guide

This document details the complete environment setup process for the Olist Customer Intelligence Platform.

## Prerequisites

- macOS (tested on macOS Sequoia)
- Homebrew installed
- Git installed
- Google account with GCP access
- Kaggle account

## Step 1: Project Structure

Created the following directory structure:

```bash
mkdir -p olist-model/{data/raw,data/processed,data/splits,notebooks,src,models,tests,configs,docs}
cd olist-model
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| `configs/` | Configuration files (config.yaml) |
| `data/raw/` | Original CSV files from Kaggle |
| `data/processed/` | Cleaned and merged datasets |
| `data/splits/` | Train/val/test splits (parquet format) |
| `docs/` | Project documentation |
| `models/` | Saved model artifacts (.joblib files) |
| `notebooks/` | Jupyter notebooks for each phase |
| `src/` | Python source code modules |
| `tests/` | Unit and integration tests |

## Step 2: Git Repository

### Initialize Git

```bash
git init
```

### Configure Local Git User

```bash
git config user.email "ismailsalikhodjaev@gmail.com"
git config user.name "Ismail"
```

### Create .gitignore

```gitignore
# Data (too large for git)
data/
*.csv
*.parquet

# Models
models/*.joblib
models/*.pkl

# Environment
.venv/
venv/
__pycache__/
*.pyc
.env

# Notebooks
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# GCP
*.json
!configs/*.json
service-account-key.json
```

### Connect to GitHub

```bash
# Install GitHub CLI
brew install gh

# Authenticate
gh auth login --git-protocol ssh --hostname github.com --web

# Create and push repository
git add .
git commit -m "Initial project setup: structure, dependencies, GCP config"
gh repo create olist-model --public --source=. --push --description "E-Commerce Customer Intelligence Platform"
```

**Repository URL**: https://github.com/ismailsaleekh/olist-model

## Step 3: Python Environment

### Install Python 3.10

```bash
brew install python@3.10
```

### Create Virtual Environment

```bash
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Verify Installation

```bash
.venv/bin/python --version
# Output: Python 3.10.19

.venv/bin/pip --version
# Output: pip 25.3
```

## Step 4: Dependencies

### requirements.txt

```txt
# Core Data Science
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# ML / Boosting
xgboost>=2.0.0
lightgbm>=4.0.0

# NLP (Sentiment Analysis)
textblob>=0.17.0

# Feature Engineering
category_encoders>=2.6.0

# Data Validation
pandera>=0.17.0

# Model Interpretability
shap>=0.42.0

# ========== GCP Stack ==========
google-cloud-aiplatform>=1.38.0
google-cloud-storage>=2.10.0
google-cloud-bigquery>=3.12.0
google-cloud-logging>=3.8.0
google-auth>=2.23.0

# Vertex AI Pipelines (for Day 4)
kfp>=2.4.0
google-cloud-pipeline-components>=2.6.0

# Testing
pytest>=7.4.0

# Utilities
joblib>=1.3.0
python-dotenv>=1.0.0
tqdm>=4.66.0
pyarrow>=14.0.0
pyyaml>=6.0.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0
```

### Install Dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

### Install OpenMP (Required for XGBoost on macOS)

```bash
brew install libomp
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /opt/homebrew/lib/libomp.dylib
```

### Verify Key Packages

```python
import pandas as pd          # 2.3.3
import numpy as np           # 2.2.6
import sklearn               # 1.7.2
import xgboost               # 3.1.2
import lightgbm              # 4.6.0
import shap                  # 0.49.1
from google.cloud import aiplatform  # 1.132.0
```

## Step 5: Source Files

### Created Files in src/

| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization |
| `data_loader.py` | Data loading utilities |
| `feature_engineering.py` | Feature transformation pipeline |
| `gcp_utils.py` | GCP helper functions |
| `predict.py` | Inference utilities |
| `train.py` | Model training utilities |

### Created Files in tests/

| File | Purpose |
|------|---------|
| `__init__.py` | Test package initialization |
| `test_data_loader.py` | Tests for data loading |
| `test_features.py` | Tests for feature engineering |
| `test_model.py` | Tests for model inference |

## Step 6: Notebooks

Created 6 Jupyter notebooks:

| Notebook | Purpose |
|----------|---------|
| `00_setup_validation.ipynb` | Verify environment setup |
| `01_data_exploration.ipynb` | EDA and data quality |
| `02_feature_engineering.ipynb` | Feature pipeline creation |
| `03_unsupervised_learning.ipynb` | Clustering experiments |
| `04_supervised_learning.ipynb` | Classification & regression |
| `05_gcp_deployment.ipynb` | Deployment and monitoring |

## Troubleshooting

### XGBoost Import Error

If you see "libomp.dylib not found":

```bash
brew install libomp
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /opt/homebrew/lib/libomp.dylib
```

### Python Version Warnings

GCP libraries show warnings about Python 3.10 deprecation (2026). These can be ignored for this project.

### Virtual Environment Activation

Always activate the virtual environment before running code:

```bash
source .venv/bin/activate
```

## Next Steps

After environment setup:
1. Configure GCP (see `docs/gcp-setup.md`)
2. Download dataset from Kaggle
3. Run validation notebook
