# Day 0: Project Setup - Full Plan (GCP Stack)

## Overview

| Aspect | Details |
|--------|---------|
| Duration | 2-3 hours |
| Goal | Environment ready, GCP configured, data downloaded |
| Output | Working dev environment + GCP project ready + initial git commit |

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Python Environment** | `venv` | Simple, built-in, sufficient |
| **Python Version** | 3.10 | Best compatibility with ML libs |
| **Experiment Tracking** | Vertex AI Experiments | GCP-native, integrates with deployment |
| **Data Validation** | Pandera | Lightweight, pandas-native |
| **Artifact Storage** | Cloud Storage | GCP-native |
| **Data Versioning** | Skip | Dataset is small (~50MB) |

---

## Dataset Summary

**Source**: [Kaggle - Brazilian E-Commerce (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

**9 interconnected CSV files:**

| File | Description | Key Columns |
|------|-------------|-------------|
| `olist_orders_dataset.csv` | Core orders (~100K) | `order_id`, `customer_id`, timestamps, `order_status` |
| `olist_order_items_dataset.csv` | Items per order | `order_id`, `product_id`, `seller_id`, `price`, `freight_value` |
| `olist_products_dataset.csv` | Product catalog (~33K) | `product_id`, `product_category_name`, dimensions, weight |
| `olist_customers_dataset.csv` | Customers (~100K) | `customer_id`, `customer_unique_id`, `zip_code`, `city`, `state` |
| `olist_sellers_dataset.csv` | Sellers | `seller_id`, `zip_code`, `city`, `state` |
| `olist_order_payments_dataset.csv` | Payments | `order_id`, `payment_type`, `payment_value`, `installments` |
| `olist_order_reviews_dataset.csv` | Reviews (1-5 stars) | `order_id`, `review_score`, `review_comment_message` |
| `olist_geolocation_dataset.csv` | Zip → lat/lng | `zip_code_prefix`, `lat`, `lng`, `city`, `state` |
| `product_category_name_translation.csv` | PT → EN translation | `product_category_name`, `product_category_name_english` |

**Key relationships:**
- Orders → Order Items (1:N)
- Order Items → Products (N:1)
- Order Items → Sellers (N:1)
- Orders → Customers (N:1)
- Orders → Payments (1:N)
- Orders → Reviews (1:1)

---

## Task Breakdown

### Task 1: Create Project Structure

```bash
mkdir -p olist-model/{data/raw,data/processed,data/splits,notebooks,src,models,tests,configs}
cd olist-model
```

**Final structure:**
```
olist-model/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/           # Original 9 CSVs
│   ├── processed/     # Cleaned, merged data
│   └── splits/        # train.parquet, val.parquet, test.parquet
├── models/            # Saved .joblib artifacts
├── notebooks/
│   ├── 00_setup_validation.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_unsupervised_learning.ipynb
│   ├── 04_supervised_learning.ipynb
│   └── 05_gcp_deployment.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

### Task 2: Initialize Git Repository

```bash
git init
```

**Create `.gitignore`:**
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

---

### Task 3: Setup Python Environment

```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux

# Upgrade pip
pip install --upgrade pip
```

---

### Task 4: Create requirements.txt (GCP Stack)

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
google-cloud-aiplatform>=1.38.0    # Vertex AI (Training, Experiments, Endpoints)
google-cloud-storage>=2.10.0        # Cloud Storage
google-cloud-bigquery>=3.12.0       # BigQuery (optional, for analytics)
google-cloud-logging>=3.8.0         # Cloud Logging
google-auth>=2.23.0                 # Authentication

# Vertex AI Pipelines (for Day 4)
kfp>=2.4.0                          # Kubeflow Pipelines SDK
google-cloud-pipeline-components>=2.6.0

# Testing
pytest>=7.4.0

# Utilities
joblib>=1.3.0
python-dotenv>=1.0.0
tqdm>=4.66.0
pyarrow>=14.0.0    # For parquet support
pyyaml>=6.0.0      # For config files

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0
```

**Install:**
```bash
pip install -r requirements.txt
```

---

### Task 5: GCP Project Setup

#### Step 5.1: Create/Select GCP Project

```bash
# Install gcloud CLI if not installed
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Create new project (or use existing)
gcloud projects create olist-ml-project --name="Olist ML Project"

# Set as default
gcloud config set project olist-ml-project

# Verify
gcloud config get-value project
```

#### Step 5.2: Enable Required APIs

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    logging.googleapis.com \
    bigquery.googleapis.com
```

#### Step 5.3: Set Default Region

```bash
gcloud config set compute/region us-central1
```

#### Step 5.4: Setup Application Default Credentials (for local dev)

```bash
gcloud auth application-default login
```

#### Step 5.5: Create Cloud Storage Bucket

```bash
# Create bucket for artifacts (globally unique name required)
gsutil mb -l us-central1 gs://olist-ml-<your-unique-id>

# Verify
gsutil ls
```

#### Step 5.6: (Optional) Create Service Account for CI/CD

```bash
# Create service account
gcloud iam service-accounts create olist-ml-sa \
    --display-name="Olist ML Service Account"

# Grant roles
PROJECT_ID=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:olist-ml-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:olist-ml-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

---

### Task 6: Create Configuration File (GCP Stack)

**`configs/config.yaml`:**
```yaml
# Reproducibility
random_seed: 42

# ========== GCP Configuration ==========
gcp:
  project_id: "olist-ml-project"  # UPDATE THIS
  region: "us-central1"
  bucket: "olist-ml-<your-unique-id>"  # UPDATE THIS

# Vertex AI Experiments
vertex_ai:
  experiment_name: "olist-customer-intelligence"
  staging_bucket: "gs://olist-ml-<your-unique-id>/staging"

# Data split configuration
split:
  strategy: "time_based"  # Options: random, time_based, customer_based
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

# Paths (local)
paths:
  raw_data: "data/raw/"
  processed_data: "data/processed/"
  splits: "data/splits/"
  models: "models/"

# Paths (GCS) - for deployment
gcs_paths:
  data: "gs://olist-ml-<your-unique-id>/data/"
  models: "gs://olist-ml-<your-unique-id>/models/"
  artifacts: "gs://olist-ml-<your-unique-id>/artifacts/"

# Target variable definitions
targets:
  classification:
    name: "is_satisfied"
    threshold: 4  # review_score >= 4 is satisfied
  regression:
    name: "delivery_days"

# Model training
training:
  cv_folds: 5
  n_jobs: -1  # Use all cores
```

---

### Task 7: Download Dataset from Kaggle

**Option A: Using Kaggle API (Recommended)**

```bash
# Setup Kaggle API credentials first:
# 1. Go to kaggle.com → Account → Create New API Token
# 2. Place kaggle.json in ~/.kaggle/
# 3. chmod 600 ~/.kaggle/kaggle.json

pip install kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw
unzip data/raw/brazilian-ecommerce.zip -d data/raw/
rm data/raw/brazilian-ecommerce.zip
```

**Option B: Manual Download**
1. Visit https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Click "Download" button
3. Extract to `data/raw/`

---

### Task 8: Create Initial Source Files

**`src/__init__.py`:**
```python
"""Olist Customer Intelligence Platform."""
__version__ = "0.1.0"
```

**`src/data_loader.py`** (skeleton):
```python
"""Data loading utilities."""
import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str = "data/raw/") -> dict[str, pd.DataFrame]:
    """Load all raw CSV files into a dictionary of DataFrames."""
    pass


def merge_datasets(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all datasets into a single DataFrame."""
    pass
```

**`src/gcp_utils.py`** (GCP helper functions):
```python
"""GCP utility functions."""
import yaml
from google.cloud import aiplatform
from google.cloud import storage


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def init_vertex_ai(config: dict) -> None:
    """Initialize Vertex AI with project settings."""
    aiplatform.init(
        project=config["gcp"]["project_id"],
        location=config["gcp"]["region"],
        staging_bucket=config["vertex_ai"]["staging_bucket"],
        experiment=config["vertex_ai"]["experiment_name"],
    )


def upload_to_gcs(local_path: str, gcs_path: str, config: dict) -> str:
    """Upload a file to Google Cloud Storage."""
    client = storage.Client(project=config["gcp"]["project_id"])
    bucket = client.bucket(config["gcp"]["bucket"])
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return f"gs://{config['gcp']['bucket']}/{gcs_path}"
```

---

### Task 9: Create Validation Notebook

**`notebooks/00_setup_validation.ipynb`:**

```python
# Cell 1: Python Environment Check
import sys
print(f"Python version: {sys.version}")

import pandas as pd
import numpy as np
import sklearn
import xgboost

print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgboost.__version__}")
```

```python
# Cell 2: GCP Libraries Check
from google.cloud import aiplatform
from google.cloud import storage
import google.auth

print(f"google-cloud-aiplatform: {aiplatform.__version__}")

# Check authentication
credentials, project = google.auth.default()
print(f"\n✓ Authenticated")
print(f"  Project: {project}")
```

```python
# Cell 3: Load and Validate Config
import yaml
from pathlib import Path

with open("../configs/config.yaml") as f:
    config = yaml.safe_load(f)

print("Configuration loaded:")
print(f"  GCP Project: {config['gcp']['project_id']}")
print(f"  Region: {config['gcp']['region']}")
print(f"  Bucket: {config['gcp']['bucket']}")
print(f"  Experiment: {config['vertex_ai']['experiment_name']}")
```

```python
# Cell 4: Verify GCS Bucket Access
from google.cloud import storage

client = storage.Client(project=config["gcp"]["project_id"])
bucket_name = config["gcp"]["bucket"]

try:
    bucket = client.get_bucket(bucket_name)
    print(f"✓ Bucket '{bucket_name}' accessible")
    print(f"  Location: {bucket.location}")
    print(f"  Storage class: {bucket.storage_class}")
except Exception as e:
    print(f"✗ Bucket error: {e}")
```

```python
# Cell 5: Verify Vertex AI Connection
from google.cloud import aiplatform

aiplatform.init(
    project=config["gcp"]["project_id"],
    location=config["gcp"]["region"],
)

print(f"✓ Vertex AI initialized")
print(f"  Project: {config['gcp']['project_id']}")
print(f"  Location: {config['gcp']['region']}")

# List existing experiments (if any)
experiments = aiplatform.Experiment.list()
print(f"  Existing experiments: {len(experiments)}")
```

```python
# Cell 6: Create Experiment (if not exists)
from google.cloud import aiplatform

experiment_name = config["vertex_ai"]["experiment_name"]

try:
    experiment = aiplatform.Experiment.create(
        experiment_name=experiment_name,
        description="Olist Customer Intelligence Platform - ML Experiments"
    )
    print(f"✓ Created experiment: {experiment_name}")
except Exception as e:
    # Experiment might already exist
    experiment = aiplatform.Experiment(experiment_name=experiment_name)
    print(f"✓ Using existing experiment: {experiment_name}")
```

```python
# Cell 7: Verify Data Files
from pathlib import Path

data_path = Path("../data/raw")
expected_files = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]

print("Data files:")
all_present = True
for f in expected_files:
    path = data_path / f
    if path.exists():
        df = pd.read_csv(path, nrows=1)
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} - MISSING")
        all_present = False

if all_present:
    print("\n✓ All 9 data files present")
```

```python
# Cell 8: Quick Data Overview
print("Dataset Overview:\n")
total_memory = 0

for f in expected_files:
    df = pd.read_csv(data_path / f)
    mem = df.memory_usage(deep=True).sum() / 1024**2
    total_memory += mem
    print(f"{f}")
    print(f"  Rows: {df.shape[0]:,} | Cols: {df.shape[1]} | Memory: {mem:.2f} MB\n")

print(f"Total memory: {total_memory:.2f} MB")
```

```python
# Cell 9: Setup Summary
print("=" * 50)
print("DAY 0 SETUP VALIDATION COMPLETE")
print("=" * 50)
print(f"""
✓ Python environment: {sys.version.split()[0]}
✓ GCP Project: {config['gcp']['project_id']}
✓ GCS Bucket: {config['gcp']['bucket']}
✓ Vertex AI Experiment: {config['vertex_ai']['experiment_name']}
✓ Data files: All 9 CSVs present
✓ Total data size: {total_memory:.2f} MB

Ready for Day 1: Data Engineering & EDA
""")
```

---

### Task 10: Initial Git Commit

```bash
git add .
git commit -m "Initial project setup: structure, dependencies, GCP config"
```

---

## Day 0 Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Folder structure created | ✅ |
| 2 | Git repository initialized | ☐ |
| 3 | `.gitignore` configured | ☐ |
| 4 | Python virtual environment created | ☐ |
| 5 | Dependencies installed | ☐ |
| 6 | `gcloud` CLI installed and authenticated | ☐ |
| 7 | GCP project created/selected | ☐ |
| 8 | Required GCP APIs enabled | ☐ |
| 9 | Application Default Credentials set | ☐ |
| 10 | Cloud Storage bucket created | ☐ |
| 11 | Dataset downloaded and extracted | ☐ |
| 12 | `config.yaml` created with GCP settings | ☐ |
| 13 | Source file skeletons created | ✅ |
| 14 | Validation notebook created | ✅ |
| 15 | Vertex AI Experiment created | ☐ |
| 16 | All 9 CSV files verified present | ☐ |
| 17 | Initial commit made | ☐ |

---

## GCP Resources Created

| Resource | Name | Purpose |
|----------|------|---------|
| Project | `olist-ml-project` | Container for all resources |
| GCS Bucket | `olist-ml-<unique-id>` | Artifacts, models, data |
| Vertex AI Experiment | `olist-customer-intelligence` | Track ML experiments |

---

## Next Steps

After completing Day 0, proceed to **Day 1: Data Engineering & EDA** where you will:
1. Load all 9 CSVs
2. Create Train/Val/Test split (BEFORE any EDA)
3. Merge tables
4. Perform EDA on training set only
5. Handle missing values & outliers
6. Define data schema validation
