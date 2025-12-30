# Olist Customer Intelligence Platform

E-Commerce Customer Intelligence Platform using the Brazilian E-Commerce (Olist) dataset.

## Project Overview

This project combines **supervised** and **unsupervised** learning to:
1. **Segment customers** using clustering algorithms (unsupervised)
2. **Predict customer satisfaction** using classification (supervised)
3. **Predict delivery delays** using regression (supervised)

## Dataset

**Source**: [Kaggle - Brazilian E-Commerce (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| Attribute | Value |
|-----------|-------|
| Total Orders | 100,000+ |
| Time Period | Oct 2016 - Sep 2018 |
| Files | 9 interconnected CSV files |
| Total Size | ~50MB |

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10 |
| ML | scikit-learn, XGBoost, LightGBM |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Experiment Tracking | Vertex AI Experiments |
| Deployment | Vertex AI (Endpoints, Pipelines) |
| Storage | Google Cloud Storage |

## Project Structure

```
olist-model/
├── configs/
│   └── config.yaml          # Project configuration
├── data/
│   ├── raw/                  # Original 9 CSVs from Kaggle
│   ├── processed/            # Cleaned, merged data
│   └── splits/               # train.parquet, val.parquet, test.parquet
├── docs/                     # Documentation
│   ├── setup-guide.md        # Environment setup guide
│   └── gcp-setup.md          # GCP configuration guide
├── models/                   # Saved model artifacts (.joblib)
├── notebooks/
│   ├── 00_setup_validation.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_unsupervised_learning.ipynb
│   ├── 04_supervised_learning.ipynb
│   └── 05_gcp_deployment.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── feature_engineering.py # Feature transformations
│   ├── gcp_utils.py          # GCP helper functions
│   ├── predict.py            # Inference utilities
│   └── train.py              # Training utilities
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud SDK (`gcloud`)
- Kaggle account (for dataset download)

### 1. Clone Repository

```bash
git clone git@github.com:ismailsaleekh/olist-model.git
cd olist-model
```

### 2. Setup Environment

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure GCP

```bash
# Authenticate
gcloud auth login
gcloud config set project olist-ml-project

# Set Application Default Credentials
gcloud auth application-default login
```

### 4. Download Dataset

```bash
# Using Kaggle API
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw
unzip data/raw/brazilian-ecommerce.zip -d data/raw/
```

### 5. Run Validation Notebook

Open `notebooks/00_setup_validation.ipynb` to verify setup.

## GCP Resources

| Resource | Name | Purpose |
|----------|------|---------|
| Project | `olist-ml-project` | Container for all resources |
| GCS Bucket | `gs://olist-ml-ismail` | Artifacts, models, data |
| Vertex AI Experiment | `olist-customer-intelligence` | Track ML experiments |

## Development Timeline

| Day | Focus | Status |
|-----|-------|--------|
| Day 0 | Project Setup | In Progress |
| Day 1 | Data Engineering & EDA | Pending |
| Day 2 | Feature Engineering & Clustering | Pending |
| Day 3 | Supervised Learning | Pending |
| Day 4 | GCP Deployment & MLOps | Pending |

## License

This project uses the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) which is licensed under CC BY-NC-SA 4.0.
