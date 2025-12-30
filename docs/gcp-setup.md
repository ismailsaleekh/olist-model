# GCP Setup Guide

This document details the Google Cloud Platform configuration for the Olist Customer Intelligence Platform.

## Overview

We use GCP for:
- **Vertex AI Experiments**: Track ML experiments (metrics, params, artifacts)
- **Cloud Storage**: Store datasets, models, pipeline artifacts
- **Vertex AI Model Registry**: Version and manage trained models
- **Vertex AI Endpoints**: Serve models for real-time predictions
- **Vertex AI Pipelines**: Orchestrate ML workflows
- **Model Monitoring**: Detect data drift
- **Cloud Logging**: Centralized logs

## Step 1: Install Google Cloud SDK

### Install via Homebrew

```bash
brew install --cask google-cloud-sdk
```

### Verify Installation

```bash
gcloud --version
# Google Cloud SDK 550.0.0
# bq 2.1.26
# core 2025.12.12
```

## Step 2: Authentication

### Authenticate with Google Account

```bash
gcloud auth login
```

This opens a browser for OAuth authentication.

### Verify Authentication

```bash
gcloud auth list
# ACTIVE  ACCOUNT
# *       ismailsalikhodjaev@gmail.com
```

## Step 3: Create GCP Project

### Create New Project

```bash
gcloud projects create olist-ml-project --name="Olist ML Project"
```

### Set as Default Project

```bash
gcloud config set project olist-ml-project
```

### Verify Project

```bash
gcloud config get-value project
# olist-ml-project
```

## Step 4: Enable Billing

### List Billing Accounts

```bash
gcloud billing accounts list
```

### Link Billing to Project

```bash
gcloud billing projects link olist-ml-project --billing-account=01BD9B-2696BC-DDEB40
```

**Note**: Replace with your billing account ID.

## Step 5: Enable Required APIs

### Enable APIs

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    logging.googleapis.com \
    bigquery.googleapis.com \
    compute.googleapis.com
```

### Verify Enabled APIs

```bash
gcloud services list --enabled
```

### API Purposes

| API | Purpose |
|-----|---------|
| `aiplatform.googleapis.com` | Vertex AI (Experiments, Training, Endpoints) |
| `storage.googleapis.com` | Cloud Storage buckets |
| `logging.googleapis.com` | Cloud Logging |
| `bigquery.googleapis.com` | BigQuery analytics |
| `compute.googleapis.com` | Compute Engine (for Vertex AI) |

## Step 6: Set Default Region

```bash
gcloud config set compute/region us-central1
```

### Why us-central1?

- Good availability for Vertex AI services
- Lower latency for US-based development
- Competitive pricing

## Step 7: Application Default Credentials

### Set ADC for Local Development

```bash
gcloud auth application-default login
```

This creates credentials at:
```
~/.config/gcloud/application_default_credentials.json
```

### Verify ADC

```python
from google.cloud import storage
client = storage.Client()
print("ADC configured successfully")
```

## Step 8: Create Cloud Storage Bucket

### Create Bucket

```bash
gsutil mb -l us-central1 gs://olist-ml-ismail
```

### Verify Bucket

```bash
gsutil ls
# gs://olist-ml-ismail/
```

### Bucket Structure (Planned)

```
gs://olist-ml-ismail/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── clustering/
│   ├── classification/
│   └── regression/
├── artifacts/
│   ├── feature_pipeline.joblib
│   └── metrics/
└── staging/
    └── (Vertex AI staging area)
```

## Step 9: Configuration File

### configs/config.yaml

```yaml
# Reproducibility
random_seed: 42

# ========== GCP Configuration ==========
gcp:
  project_id: "olist-ml-project"
  region: "us-central1"
  bucket: "olist-ml-ismail"

# Vertex AI Experiments
vertex_ai:
  experiment_name: "olist-customer-intelligence"
  staging_bucket: "gs://olist-ml-ismail/staging"

# Data split configuration
split:
  strategy: "time_based"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

# Paths (local)
paths:
  raw_data: "data/raw/"
  processed_data: "data/processed/"
  splits: "data/splits/"
  models: "models/"

# Paths (GCS)
gcs_paths:
  data: "gs://olist-ml-ismail/data/"
  models: "gs://olist-ml-ismail/models/"
  artifacts: "gs://olist-ml-ismail/artifacts/"

# Target variable definitions
targets:
  classification:
    name: "is_satisfied"
    threshold: 4
  regression:
    name: "delivery_days"

# Model training
training:
  cv_folds: 5
  n_jobs: -1
```

## GCP Resources Summary

| Resource | Name | Region | Purpose |
|----------|------|--------|---------|
| Project | `olist-ml-project` | - | Container for all resources |
| GCS Bucket | `olist-ml-ismail` | us-central1 | Artifacts, models, data |
| Vertex AI Experiment | `olist-customer-intelligence` | us-central1 | Track ML experiments |

## Using GCP in Python

### Initialize Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(
    project="olist-ml-project",
    location="us-central1",
    staging_bucket="gs://olist-ml-ismail/staging",
    experiment="olist-customer-intelligence",
)
```

### Upload to Cloud Storage

```python
from google.cloud import storage

client = storage.Client(project="olist-ml-project")
bucket = client.bucket("olist-ml-ismail")
blob = bucket.blob("models/classifier.joblib")
blob.upload_from_filename("models/classifier.joblib")
```

### Create Vertex AI Experiment

```python
from google.cloud import aiplatform

experiment = aiplatform.Experiment.create(
    experiment_name="olist-customer-intelligence",
    description="Olist Customer Intelligence Platform - ML Experiments"
)
```

## Cost Considerations

### Estimated Monthly Costs

| Resource | Estimated Cost |
|----------|----------------|
| Cloud Storage (10GB) | ~$0.20 |
| Vertex AI Endpoint (n1-standard-2, 24/7) | ~$100-150 |
| Vertex AI Training (occasional) | ~$5-20/job |
| Model Monitoring | ~$10-30 |
| **Total Estimate** | ~$120-200/month |

### Cost Optimization Tips

1. Use preemptible VMs for training
2. Scale down replicas during low traffic
3. Use batch prediction instead of online for bulk inference
4. Delete unused endpoints
5. Set up budget alerts

## Troubleshooting

### Billing Not Enabled Error

```
ERROR: Billing account for project is not found
```

**Solution**: Link billing account to project:
```bash
gcloud billing projects link olist-ml-project --billing-account=YOUR_ACCOUNT_ID
```

### API Not Enabled Error

```
ERROR: API [xxx.googleapis.com] not enabled
```

**Solution**: Enable the API:
```bash
gcloud services enable xxx.googleapis.com
```

### Permission Denied

```
ERROR: Permission denied
```

**Solution**: Check IAM roles and re-authenticate:
```bash
gcloud auth login
gcloud auth application-default login
```

## Next Steps

After GCP setup:
1. Download dataset from Kaggle
2. Run validation notebook to verify GCP connectivity
3. Create Vertex AI Experiment
