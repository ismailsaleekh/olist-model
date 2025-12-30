
# PROJECT: E-Commerce Customer Intelligence Platform

## Overview

Build a complete customer analytics system that combines **supervised** and **unsupervised** learning to:
1. Segment customers using clustering algorithms (unsupervised)
2. Predict customer satisfaction/churn using classification (supervised)
3. Predict delivery delays using regression (supervised)

**Estimated Duration**: 4-5 days (including Day 0 setup)
**Difficulty**: Intermediate
**RAM Requirements**: ~4-8GB (easily fits in 24GB)

---

## Dataset: Brazilian E-Commerce (Olist)

### Source
**Kaggle URL**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

### Dataset Details
| Attribute | Value |
|-----------|-------|
| Total Orders | 100,000+ |
| Time Period | Oct 2016 - Sep 2018 |
| Files | 9 interconnected CSV files |
| Total Size | ~50MB (compressed) |
| Format | CSV |
| License | CC BY-NC-SA 4.0 |

### Available Tables

1. **olist_orders_dataset.csv** - Core orders data (100k rows)
2. **olist_order_items_dataset.csv** - Products in each order
3. **olist_products_dataset.csv** - Product catalog (33k products)
4. **olist_customers_dataset.csv** - Customer information (100k customers)
5. **olist_sellers_dataset.csv** - Seller information
6. **olist_order_payments_dataset.csv** - Payment details
7. **olist_order_reviews_dataset.csv** - Customer reviews (1-5 stars)
8. **olist_geolocation_dataset.csv** - Brazilian zip codes with lat/lng
9. **product_category_name_translation.csv** - Portuguese → English translation

### Why This Dataset is Perfect

- **Real-world commercial data** (anonymized)
- **Rich features**: demographics, behavior, reviews, geo-location
- **Multiple ML tasks possible**: classification, regression, clustering
- **Manageable size**: Fast training iterations on CPU/GPU
- **Well-documented**: Active Kaggle community with notebooks

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT SETUP (Day 0)                        │
├─────────────────────────────────────────────────────────────────┤
│  Environment │ Git │ Folder Structure │ Download Data │ Config  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER (Day 1)                           │
├─────────────────────────────────────────────────────────────────┤
│  Raw Data (9 CSVs) → Train/Val/Test Split → Merged Dataset      │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                  │
│         Train Set                    Val/Test Sets              │
│      (EDA & Statistics)            (Held out - NO peeking)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING (Day 2)                      │
├─────────────────────────────────────────────────────────────────┤
│  sklearn.Pipeline: RFM │ Temporal │ Geo │ NLP │ Encoding        │
│                              │                                   │
│              Save: feature_pipeline.joblib                       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   UNSUPERVISED   │ │    SUPERVISED    │ │    SUPERVISED    │
│   (Clustering)   │ │  (Classification)│ │   (Regression)   │
│     (Day 2)      │ │     (Day 3)      │ │     (Day 3)      │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ • K-Means        │ │ • Dummy (base)   │ │ • Dummy (base)   │
│ • DBSCAN         │ │ • Logistic Reg   │ │ • Linear Reg     │
│ • Hierarchical   │ │ • Random Forest  │ │ • Ridge/Lasso    │
│ • Gaussian Mix   │ │ • XGBoost        │ │ • Random Forest  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
              │               │               │
              │       ┌───────┴───────┐       │
              │       ▼               ▼       │
              │   Model Save    Experiment    │
              │   (.joblib)     Tracking      │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GCP DEPLOYMENT (Day 4)                       │
├─────────────────────────────────────────────────────────────────┤
│  Local Test → Container → Vertex AI Pipeline → Endpoint         │
│                              │                                   │
│              Model Registry → Monitoring → CI/CD                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Day-by-Day Implementation Plan

### Day 0: Project Setup (2-3 hours)

#### 0.1 Environment Setup

**DECISION: Python Environment Tool**
- Option A: `venv` (built-in, simple)
- Option B: `conda` (better for data science dependencies)
- Option C: `poetry` (modern dependency management)

```bash
# Create project structure
mkdir -p olist-model/{data/raw,data/processed,data/splits,notebooks,src,models,tests,configs}
cd olist-model
```

#### 0.2 Initialize Version Control

```bash
git init
```

Create `.gitignore`:
```
# Data
data/
*.csv
*.parquet

# Models
models/*.joblib
models/*.pkl

# Environment
.venv/
__pycache__/
.env

# Notebooks
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

#### 0.3 Dependencies

Create `requirements.txt`:
```
# Core
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# ML/Boosting
xgboost>=2.0.0
lightgbm>=4.0.0

# NLP
textblob>=0.17.0
# or: vaderSentiment>=3.3.0

# Feature Engineering
category_encoders>=2.6.0

# Experiment Tracking - DECISION: Pick one
# Option A: mlflow>=2.8.0
# Option B: wandb>=0.15.0
# Option C: Vertex AI Experiments (GCP native)

# Data Validation - DECISION: Pick one
# Option A: great_expectations>=0.17.0
# Option B: pandera>=0.17.0

# GCP
google-cloud-aiplatform>=1.35.0
google-cloud-storage>=2.10.0

# Testing
pytest>=7.4.0

# Utilities
joblib>=1.3.0
python-dotenv>=1.0.0
tqdm>=4.66.0
```

#### 0.4 Download Dataset

```bash
# Using Kaggle API
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw
unzip data/raw/brazilian-ecommerce.zip -d data/raw/
```

#### 0.5 Configuration File

Create `configs/config.yaml`:
```yaml
# Reproducibility
random_seed: 42

# Data splits - DECISION: Split ratios
# Option A: 70/15/15 (train/val/test)
# Option B: 80/10/10
# Option C: 60/20/20

# Split strategy - DECISION: Split method
# Option A: Random stratified split
# Option B: Time-based split (more realistic for production)

# Paths
data:
  raw: data/raw/
  processed: data/processed/
  splits: data/splits/

models:
  path: models/

# Experiment tracking
tracking:
  # Configure after choosing tool
```

#### 0.6 Project Structure Verification

```
olist-model/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/              # Original CSVs
│   ├── processed/        # Cleaned, merged data
│   └── splits/           # train.parquet, val.parquet, test.parquet
├── models/               # Saved model artifacts
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

#### 0.7 Data Versioning (Optional but Recommended)

**DECISION: Data Versioning Tool**
- Option A: DVC (Data Version Control)
- Option B: Git LFS
- Option C: Manual versioning with checksums

#### 0.8 Day 0 Checklist

- [ ] Python environment created and activated
- [ ] Dependencies installed
- [ ] Git repository initialized
- [ ] .gitignore configured
- [ ] Folder structure created
- [ ] Dataset downloaded and extracted
- [ ] Config file created
- [ ] Initial commit made

---

### Day 1: Data Engineering & EDA (6-8 hours)

#### 1.1 Data Loading and Initial Inspection

```python
# Load all 9 CSVs
# Check shapes, dtypes, memory usage
# Verify referential integrity between tables
```

**Validation Checks:**
- All expected files present
- Row counts match documentation
- No corrupted files

#### 1.2 Train/Validation/Test Split (DO THIS FIRST!)

> **CRITICAL**: Split BEFORE any EDA to prevent data leakage

**DECISION: Split Strategy**
- Option A: Random stratified split (stratify on target variable)
- Option B: Time-based split (train on older data, test on newer)
- Option C: Customer-based split (ensure same customer not in multiple sets)

**DECISION: Split Ratios**
- Option A: 70/15/15 (train/val/test)
- Option B: 80/10/10
- Option C: 60/20/20

```python
# Example structure (strategy TBD):
from sklearn.model_selection import train_test_split

# First split: separate test set
train_val, test = split_data(data, test_size=0.15, strategy=TBD)

# Second split: separate validation set
train, val = split_data(train_val, test_size=0.176, strategy=TBD)  # 0.176 of 0.85 ≈ 0.15

# Save splits
train.to_parquet('data/splits/train.parquet')
val.to_parquet('data/splits/val.parquet')
test.to_parquet('data/splits/test.parquet')
```

#### 1.3 Data Merging (On Train Set Only for EDA)

```python
# Key tables to merge
orders + order_items + products + customers + reviews + payments + sellers

# Define merge keys and join types
# Document any rows lost during merges
```

**Merge Validation:**
- Check row counts before/after each merge
- Identify and document any orphan records
- Verify no unintended duplication

#### 1.4 Exploratory Data Analysis (TRAIN SET ONLY)

> **CRITICAL**: All EDA statistics computed on training set only

- Distribution analysis for all numerical columns
- Category analysis for categorical columns
- Temporal patterns (orders over time, seasonality)
- Geographic distribution analysis
- Review score distribution
- Correlation analysis (numerical features)
- Target variable distribution analysis

#### 1.5 Missing Value Analysis

| Column | Missing % | Strategy Options |
|--------|-----------|------------------|
| TBD | TBD | Drop / Mean / Median / Mode / Forward Fill / Model-based |

**DECISION: Imputation Strategy**
- Document chosen strategy for each column
- Fit imputers on training data only
- Apply same transformation to val/test

#### 1.6 Data Quality Checks

- Handle missing values (using chosen imputation strategies)
- Remove duplicates (define what constitutes a duplicate)
- Handle outliers:
  - **DECISION: Outlier Detection Method**: IQR / Z-score / Isolation Forest
  - **DECISION: Outlier Treatment**: Remove / Cap / Transform / Keep
- Data type corrections
- Validate date ranges and logical constraints

#### 1.7 Data Schema Validation

**DECISION: Schema Validation Tool**
- Option A: Pandera (lightweight, pandas-native)
- Option B: Great Expectations (comprehensive, more setup)
- Option C: Custom validation functions

```python
# Define expected schema
# Validate all splits against schema
# Fail fast on schema violations
```

#### 1.8 Key Visualizations to Create

- Revenue by state (choropleth map)
- Orders over time (trend + seasonality decomposition)
- Review score distribution (note: this is our classification target)
- Payment method breakdown
- Delivery time distribution (note: this is our regression target)
- Target variable distributions (check for class imbalance)

#### 1.9 Save Cleaned Data & Artifacts

```python
# Save processed training data
train_processed.to_parquet('data/processed/train_processed.parquet')

# Save data processing artifacts (fitted imputers, etc.)
joblib.dump(imputers, 'models/data_imputers.joblib')

# Save EDA summary statistics (for reference)
eda_stats.to_json('data/processed/eda_statistics.json')
```

#### 1.10 Day 1 Checklist

- [ ] All raw data loaded successfully
- [ ] Train/val/test split created and saved
- [ ] EDA completed on training set only
- [ ] Missing value strategy documented
- [ ] Outlier handling strategy documented
- [ ] Data schema defined
- [ ] Processed data saved
- [ ] All code committed to git

---

### Day 2: Feature Engineering & Unsupervised Learning (6-8 hours)

#### 2.1 Feature Engineering Pipeline Setup

> **CRITICAL**: Use sklearn.Pipeline to ensure reproducibility

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# All feature engineering must be:
# 1. Fit on training data only
# 2. Applied consistently to val/test
# 3. Serializable for production
```

#### 2.2 Numerical Feature Scaling

**DECISION: Scaling Method**
- Option A: StandardScaler (zero mean, unit variance)
- Option B: MinMaxScaler (0-1 range)
- Option C: RobustScaler (median-based, outlier-resistant)

#### 2.3 Categorical Encoding

**DECISION: Encoding Strategy**

For nominal categories (no order):
- Option A: One-Hot Encoding (sparse, interpretable)
- Option B: Target Encoding (dense, risk of leakage)
- Option C: Binary Encoding (compact)

For ordinal categories (has order):
- Option A: Ordinal Encoding
- Option B: Custom mapping

For high-cardinality categories:
- Option A: Frequency Encoding
- Option B: Target Encoding with smoothing
- Option C: Embedding (if using neural networks)

#### 2.4 RFM Features (Recency, Frequency, Monetary)

```python
# For each customer (computed from training set dates):
- recency: days since last order (relative to training set max date)
- frequency: total number of orders
- monetary: total spend amount
- avg_order_value: monetary / frequency
```

#### 2.5 Temporal Features

```python
- order_hour, order_day_of_week, order_month
- days_to_delivery (actual)
- days_promised (estimated - ordered)
- delivery_delta (actual - promised)
- is_late_delivery: binary flag
- is_weekend_order: binary flag
- days_since_first_order: customer tenure
```

**DECISION: Cyclical Encoding for Time**
- Option A: Raw numerical (hour=0-23)
- Option B: Sine/Cosine transformation (captures cyclical nature)
- Option C: One-hot encoding

#### 2.6 Geographic Features

```python
- seller_customer_distance (haversine formula)
- customer_state_encoded
- seller_state_encoded
- is_same_state: binary flag
- customer_region (group states into regions)
```

#### 2.7 Product Features

```python
- product_category_encoded
- product_weight_g, product_volume_cm3
- product_photos_qty
- avg_product_price_in_category (computed from train set)
- price_vs_category_avg: relative pricing
```

#### 2.8 NLP Features from Reviews

**DECISION: Sentiment Analysis Tool**
- Option A: VADER (rule-based, fast, no training needed)
- Option B: TextBlob (simple, includes subjectivity)
- Option C: Pre-trained transformer (more accurate, slower)

```python
- review_text_length
- review_word_count
- review_sentiment_score
- review_sentiment_label (positive/neutral/negative)
- has_review_comment: binary
```

#### 2.9 Feature Selection (Pre-Clustering)

- Correlation analysis: remove highly correlated features (>0.9)
- Variance threshold: remove near-zero variance features

**DECISION: Dimensionality Reduction for Clustering**
- Option A: No reduction (use all features)
- Option B: PCA (keep N components or X% variance)
- Option C: Feature selection based on domain knowledge

#### 2.10 Save Feature Engineering Pipeline

```python
# CRITICAL: Save the fitted pipeline
from joblib import dump

# Save complete preprocessing pipeline
dump(preprocessing_pipeline, 'models/feature_pipeline.joblib')

# Save feature names for reference
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)
```

#### 2.11 Unsupervised Learning: Customer Segmentation

**Prepare Clustering Features:**
```python
# Select features for clustering
clustering_features = ['recency', 'frequency', 'monetary', 'avg_review_score']

# Apply scaling (fit on train only!)
scaler = StandardScaler()
X_cluster_train = scaler.fit_transform(train[clustering_features])
```

**Algorithm 1: K-Means Clustering**
```python
# Key steps:
1. Find optimal K using:
   - Elbow method (inertia plot)
   - Silhouette score analysis
   - Gap statistic (optional)
2. Train K-Means with chosen K
3. Analyze cluster centers
4. Visualize clusters with PCA (2D/3D)
```

**Algorithm 2: Hierarchical Clustering**
```python
# Key steps:
1. Create dendrogram (use subset if data is large)
2. Determine optimal cut height
3. Compare cluster assignments with K-Means
```

**Algorithm 3: DBSCAN**
```python
# Key steps:
1. Tune eps using k-distance graph
2. Tune min_samples based on domain knowledge
3. Identify and analyze noise points
4. Compare cluster characteristics
```

**Algorithm 4: Gaussian Mixture Models**
```python
# Key steps:
1. Select n_components using BIC/AIC
2. Analyze soft cluster probabilities
3. Compare with hard clustering results
```

#### 2.12 Clustering Model Comparison

| Algorithm | # Clusters | Silhouette | Davies-Bouldin | Interpretation |
|-----------|------------|------------|----------------|----------------|
| K-Means | TBD | TBD | TBD | TBD |
| Hierarchical | TBD | TBD | TBD | TBD |
| DBSCAN | TBD | TBD | TBD | TBD |
| GMM | TBD | TBD | TBD | TBD |

**DECISION: Final Clustering Algorithm**
- Select based on metrics + business interpretability
- Document reasoning for choice

#### 2.13 Cluster Analysis & Business Interpretation

- Profile each cluster (mean/median features)
- Assign business labels (e.g., "High-Value Loyal", "At-Risk", "New Customers")
- Cluster size distribution
- Statistical significance tests between clusters

#### 2.14 Add Cluster Labels as Feature

```python
# Add cluster assignment to feature set for supervised learning
train['customer_segment'] = cluster_model.predict(X_cluster_train)

# For val/test: transform using fitted scaler, then predict
X_cluster_val = scaler.transform(val[clustering_features])
val['customer_segment'] = cluster_model.predict(X_cluster_val)
```

#### 2.15 Save Clustering Artifacts

```python
# Save clustering model
dump(cluster_model, 'models/clustering_model.joblib')

# Save clustering scaler (separate from main pipeline)
dump(cluster_scaler, 'models/cluster_scaler.joblib')

# Save cluster profiles
cluster_profiles.to_csv('models/cluster_profiles.csv')
```

#### 2.16 Day 2 Checklist

- [ ] Feature engineering pipeline created and fitted on train only
- [ ] All transformations documented
- [ ] Feature pipeline saved (.joblib)
- [ ] Clustering experiments completed
- [ ] Clustering model selected and justified
- [ ] Cluster profiles created and interpreted
- [ ] Cluster labels added to all splits
- [ ] All artifacts saved
- [ ] Code committed to git

---

### Day 3: Supervised Learning (6-8 hours)

#### 3.1 Experiment Tracking Setup

**DECISION: Experiment Tracking Tool**
- Option A: MLflow (open source, self-hosted or managed)
- Option B: Weights & Biases (cloud-based, great visualization)
- Option C: Vertex AI Experiments (GCP native, integrates with deployment)

```python
# Initialize tracking
# Log: parameters, metrics, artifacts, model versions
```

#### 3.2 Establish Baselines FIRST

> **CRITICAL**: Always start with dummy baselines to ensure your model adds value

**Classification Baseline:**
```python
from sklearn.dummy import DummyClassifier

# Baseline strategies to try:
# - "most_frequent": always predict majority class
# - "stratified": random predictions based on class distribution
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train)
baseline_accuracy = dummy_clf.score(X_val, y_val)
```

**Regression Baseline:**
```python
from sklearn.dummy import DummyRegressor

# Baseline strategies to try:
# - "mean": always predict training mean
# - "median": always predict training median
dummy_reg = DummyRegressor(strategy='mean')
dummy_reg.fit(X_train, y_train)
baseline_rmse = np.sqrt(mean_squared_error(y_val, dummy_reg.predict(X_val)))
```

#### 3.3 Classification Task: Customer Satisfaction Prediction

**Target Variable Definition:**
```python
# review_score → Binary
# satisfied: 4-5 stars
# unsatisfied: 1-3 stars
y = (df['review_score'] >= 4).astype(int)
```

**Check Class Distribution:**
```python
# Analyze class imbalance
print(y_train.value_counts(normalize=True))

# If imbalanced (e.g., 80/20), plan mitigation strategy
```

**DECISION: Class Imbalance Handling**
- Option A: Class weights in model
- Option B: SMOTE oversampling
- Option C: Random undersampling
- Option D: Threshold tuning post-training
- Option E: Combination approach

**Features to Use:**
- Delivery-related: `delivery_delta`, `is_late_delivery`
- Product: `product_weight`, `product_category`, `price`
- Seller: `seller_state`, historical seller rating
- Customer: cluster assignment from Day 2
- Order: payment method, order value

**DECISION: Cross-Validation Strategy**
- Option A: StratifiedKFold (k=5 or k=10)
- Option B: TimeSeriesSplit (if temporal ordering matters)
- Option C: RepeatedStratifiedKFold (more robust estimates)

**Algorithms to Implement:**

1. **Logistic Regression** (Simple Baseline)
   ```python
   - Regularization: DECISION - L1 (Lasso) / L2 (Ridge) / ElasticNet
   - Interpret coefficients
   - Feature importance via coefficients
   ```

2. **Decision Tree**
   ```python
   - Visualize tree structure
   - Understand splits
   - Tune: max_depth, min_samples_split, min_samples_leaf
   ```

3. **Random Forest**
   ```python
   - Feature importance (MDI and permutation)
   - Tune: n_estimators, max_depth, min_samples_split, max_features
   ```

4. **XGBoost or LightGBM**
   ```python
   - DECISION: XGBoost vs LightGBM
   - Learning curves
   - Tune: n_estimators, max_depth, learning_rate, subsample
   ```

**DECISION: Hyperparameter Tuning Method**
- Option A: GridSearchCV (exhaustive, slower)
- Option B: RandomizedSearchCV (faster, good coverage)
- Option C: Optuna/Hyperopt (Bayesian optimization, most efficient)

**Evaluation Metrics:**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)

# Primary metric: DECISION - F1 / ROC-AUC / Precision / Recall
# (depends on business cost of false positives vs false negatives)
```

**Model Comparison Table:**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time |
|-------|----------|-----------|--------|-----|---------|------------|
| Dummy Baseline | TBD | TBD | TBD | TBD | TBD | - |
| Logistic Reg | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |

#### 3.4 Regression Task: Delivery Time Prediction

**Target Variable:**
```python
y = df['actual_delivery_days']  # or calculate from timestamps
```

**Check Target Distribution:**
```python
# Analyze distribution
# Check for outliers in target
# Consider transformation if heavily skewed
```

**DECISION: Target Transformation**
- Option A: No transformation (if approximately normal)
- Option B: Log transformation (if right-skewed)
- Option C: Box-Cox transformation (automatic)
- Option D: Quantile transformation

**Algorithms to Implement:**

1. **Linear Regression** (Baseline)
   ```python
   - Check residual plots (should be random, homoscedastic)
   - Normality of residuals (Q-Q plot)
   - Multicollinearity analysis (VIF scores)
   ```

2. **Ridge/Lasso Regression**
   ```python
   - DECISION: Ridge vs Lasso vs ElasticNet
   - Alpha tuning with cross-validation
   - Feature selection with Lasso (coefficients → 0)
   ```

3. **Decision Tree Regressor**
   ```python
   - Tune: max_depth, min_samples_split
   ```

4. **Random Forest Regressor**
   ```python
   - Tune hyperparameters
   - Feature importance analysis
   ```

5. **Gradient Boosting (XGBoost/LightGBM)**
   ```python
   - DECISION: XGBoost vs LightGBM
   - Tune hyperparameters
   ```

**Evaluation Metrics:**
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# RMSE: penalizes large errors
# MAE: more interpretable, robust to outliers
# R²: explained variance
# MAPE: percentage error (if no zeros in target)
```

**Model Comparison Table:**
| Model | RMSE | MAE | R² | MAPE | Train Time |
|-------|------|-----|-----|------|------------|
| Dummy (mean) | TBD | TBD | TBD | TBD | - |
| Linear Reg | TBD | TBD | TBD | TBD | TBD |
| Ridge | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

#### 3.5 Model Interpretability

```python
# SHAP values for feature importance
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_val)

# Summary plot
shap.summary_plot(shap_values, X_val)

# Individual prediction explanations
shap.force_plot(explainer.expected_value, shap_values[0], X_val.iloc[0])
```

#### 3.6 Final Model Selection

**DECISION: Select Best Models**
- Classification: Select based on primary metric + interpretability needs
- Regression: Select based on primary metric + business requirements

**Final Evaluation on Test Set:**
```python
# ONLY after all tuning is complete
# Run ONCE on test set
# Report final metrics
```

#### 3.7 Model Serialization

```python
# Save best classification model
dump(best_classifier, 'models/satisfaction_classifier.joblib')

# Save best regression model
dump(best_regressor, 'models/delivery_predictor.joblib')

# Save model metadata
model_metadata = {
    'classifier': {
        'algorithm': 'TBD',
        'hyperparameters': {...},
        'metrics': {...},
        'training_date': datetime.now().isoformat()
    },
    'regressor': {
        'algorithm': 'TBD',
        'hyperparameters': {...},
        'metrics': {...},
        'training_date': datetime.now().isoformat()
    }
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f)
```

#### 3.8 Day 3 Checklist

- [ ] Experiment tracking initialized
- [ ] Baseline models established (dummy classifiers/regressors)
- [ ] All classification models trained and tuned
- [ ] All regression models trained and tuned
- [ ] Cross-validation completed for all models
- [ ] Class imbalance handled (if applicable)
- [ ] Model comparison tables completed
- [ ] Best models selected with justification
- [ ] SHAP/interpretability analysis done
- [ ] Final evaluation on test set (once)
- [ ] Models serialized and saved
- [ ] Model metadata documented
- [ ] All experiments logged
- [ ] Code committed to git

---

### Day 4: GCP Deployment & MLOps (6-8 hours)

#### 4.1 Local Testing First

> **CRITICAL**: Test everything locally before cloud deployment

```python
# Test inference pipeline locally
from joblib import load

# Load all artifacts
feature_pipeline = load('models/feature_pipeline.joblib')
classifier = load('models/satisfaction_classifier.joblib')
regressor = load('models/delivery_predictor.joblib')

# Test with sample data
sample = pd.DataFrame([{...}])  # Sample input
features = feature_pipeline.transform(sample)
prediction = classifier.predict(features)

# Verify output format and values
assert prediction.shape == (1,)
assert prediction[0] in [0, 1]
```

#### 4.2 Input Validation

```python
# Define input schema validation
def validate_input(data: dict) -> bool:
    required_fields = ['customer_id', 'product_id', 'order_date', ...]

    # Check required fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Check data types
    # Check value ranges
    # Return validated data or raise exception
```

#### 4.3 Containerization

**DECISION: Container Approach**
- Option A: Use pre-built Vertex AI containers (simpler)
- Option B: Custom Docker container (more control)

**If Custom Container:**
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

# Vertex AI expects specific entrypoint
ENTRYPOINT ["python", "-m", "src.predict"]
```

#### 4.4 GCP Components Overview

| Component | Purpose | Required? |
|-----------|---------|-----------|
| **Cloud Storage** | Store datasets, model artifacts | Yes |
| **Artifact Registry** | Store Docker containers | Yes (if custom container) |
| **BigQuery** | Data warehouse for analytics | Optional |
| **Vertex AI Workbench** | Jupyter notebooks in cloud | Optional |
| **Vertex AI Training** | Custom model training jobs | Optional (can train locally) |
| **Vertex AI Model Registry** | Model versioning | Yes |
| **Vertex AI Endpoints** | Model serving (online predictions) | Yes |
| **Vertex AI Batch Prediction** | Batch inference | Optional |
| **Vertex AI Pipelines** | ML workflow orchestration | Recommended |
| **Vertex AI Model Monitoring** | Drift detection | Recommended |
| **Cloud Logging** | Centralized logging | Yes |
| **IAM** | Access control | Yes |

#### 4.5 GCP Setup

**Step 1: Setup GCP Project**
```bash
# Create project (or use existing)
gcloud projects create olist-ml-project --name="Olist ML Project"
gcloud config set project olist-ml-project

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable logging.googleapis.com

# Set up authentication
gcloud auth application-default login
```

**Step 2: IAM & Service Accounts**
```bash
# Create service account for Vertex AI
gcloud iam service-accounts create vertex-ai-sa \
    --display-name="Vertex AI Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding olist-ml-project \
    --member="serviceAccount:vertex-ai-sa@olist-ml-project.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding olist-ml-project \
    --member="serviceAccount:vertex-ai-sa@olist-ml-project.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

**Step 3: Create Cloud Storage Bucket**
```bash
gsutil mb -l us-central1 gs://olist-ml-artifacts

# Upload model artifacts
gsutil cp -r models/ gs://olist-ml-artifacts/models/
gsutil cp -r data/processed/ gs://olist-ml-artifacts/data/
```

#### 4.6 Upload Model to Vertex AI Model Registry

```python
from google.cloud import aiplatform

aiplatform.init(
    project='olist-ml-project',
    location='us-central1',
    staging_bucket='gs://olist-ml-artifacts'
)

# Upload model
model = aiplatform.Model.upload(
    display_name="customer-satisfaction-classifier",
    artifact_uri="gs://olist-ml-artifacts/models/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    labels={"version": "1.0", "task": "classification"}
)
```

#### 4.7 Deploy to Endpoint

```python
# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="olist-prediction-endpoint",
    labels={"environment": "production"}
)

# Deploy model to endpoint
deployed_model = model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3,
    traffic_split={"0": 100},
    service_account="vertex-ai-sa@olist-ml-project.iam.gserviceaccount.com"
)
```

#### 4.8 Test Deployed Endpoint

```python
# Test prediction
test_instance = {
    "feature1": value1,
    "feature2": value2,
    ...
}

prediction = endpoint.predict(instances=[test_instance])
print(prediction)
```

#### 4.9 Batch Prediction Setup (Optional)

```python
# For large-scale predictions
batch_job = model.batch_predict(
    job_display_name="batch-satisfaction-prediction",
    gcs_source="gs://olist-ml-artifacts/data/to_predict.csv",
    gcs_destination_prefix="gs://olist-ml-artifacts/predictions/",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=5
)
```

#### 4.10 Model Monitoring Setup

```python
from google.cloud.aiplatform import model_monitoring

# Define monitoring job
monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="olist-model-monitoring",
    endpoint=endpoint,
    logging_sampling_strategy={"randomSampleConfig": {"sampleRate": 0.1}},
    schedule_config={"monitorInterval": {"seconds": 3600}},  # hourly

    # Feature drift detection
    drift_detection_config={
        "drift_thresholds": {
            "delivery_delta": 0.3,
            "price": 0.2,
        }
    },

    # Skew detection (training vs serving)
    skew_detection_config={
        "skew_thresholds": {
            "delivery_delta": 0.3,
            "price": 0.2,
        },
        "attribute_skew_thresholds": {},
    },

    # Alert configuration
    alert_config={
        "email_alert_config": {
            "user_emails": ["your-email@example.com"]
        }
    }
)
```

#### 4.11 Vertex AI Pipeline (Recommended)

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component, pipeline
from google.cloud import aiplatform

@component(base_image="python:3.10")
def preprocess_data(input_path: str, output_path: str):
    # Data preprocessing logic
    pass

@component(base_image="python:3.10")
def train_model(data_path: str, model_path: str):
    # Training logic
    pass

@component(base_image="python:3.10")
def evaluate_model(model_path: str, test_data_path: str) -> float:
    # Evaluation logic
    return metrics

@component(base_image="python:3.10")
def deploy_model(model_path: str, endpoint_name: str):
    # Deployment logic
    pass

@pipeline(name="olist-customer-intelligence-pipeline")
def ml_pipeline(
    raw_data_path: str,
    model_output_path: str,
    deploy_threshold: float = 0.8
):
    # Preprocess
    preprocess_task = preprocess_data(
        input_path=raw_data_path,
        output_path=f"{model_output_path}/processed"
    )

    # Train
    train_task = train_model(
        data_path=preprocess_task.output,
        model_path=model_output_path
    )

    # Evaluate
    eval_task = evaluate_model(
        model_path=train_task.output,
        test_data_path=f"{model_output_path}/test"
    )

    # Conditional deployment
    with dsl.Condition(eval_task.output > deploy_threshold):
        deploy_model(
            model_path=train_task.output,
            endpoint_name="olist-prediction-endpoint"
        )

# Compile and run pipeline
from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path="pipeline.json"
)

# Submit pipeline job
job = aiplatform.PipelineJob(
    display_name="olist-training-pipeline",
    template_path="pipeline.json",
    parameter_values={
        "raw_data_path": "gs://olist-ml-artifacts/data/raw",
        "model_output_path": "gs://olist-ml-artifacts/models"
    }
)
job.run()
```

#### 4.12 CI/CD Setup

**DECISION: CI/CD Platform**
- Option A: GitHub Actions
- Option B: Cloud Build (GCP native)
- Option C: GitLab CI
- Option D: Jenkins

**Example GitHub Actions Workflow:**
```yaml
# .github/workflows/ml-pipeline.yaml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'tests/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - name: Trigger Vertex AI Pipeline
        run: |
          # Trigger pipeline run
          python scripts/trigger_pipeline.py
```

#### 4.13 Logging Strategy

```python
import logging
from google.cloud import logging as cloud_logging

# Setup Cloud Logging
client = cloud_logging.Client()
client.setup_logging()

logger = logging.getLogger(__name__)

# Log predictions
def log_prediction(input_data, prediction, latency):
    logger.info(
        "Prediction made",
        extra={
            "json_fields": {
                "input": input_data,
                "prediction": prediction,
                "latency_ms": latency,
                "model_version": "1.0"
            }
        }
    )
```

#### 4.14 Rollback Strategy

```python
# Traffic splitting for gradual rollout
endpoint.undeploy(deployed_model_id=new_model_id)

# Or: Split traffic between versions
endpoint.update(traffic_split={
    old_model_id: 90,
    new_model_id: 10
})

# If issues detected, rollback:
endpoint.update(traffic_split={
    old_model_id: 100,
    new_model_id: 0
})
```

#### 4.15 Cost Estimation

| Resource | Estimated Monthly Cost |
|----------|----------------------|
| Cloud Storage (10GB) | ~$0.20 |
| Vertex AI Endpoint (n1-standard-2, 24/7) | ~$100-150 |
| Vertex AI Training (occasional) | ~$5-20/job |
| Model Monitoring | ~$10-30 |
| **Total Estimate** | ~$120-200/month |

**Cost Optimization Tips:**
- Use preemptible VMs for training
- Scale down replicas during low traffic
- Use batch prediction instead of online for bulk inference
- Delete unused endpoints

#### 4.16 Day 4 Checklist

- [ ] Local inference testing passed
- [ ] Input validation implemented
- [ ] GCP project and APIs configured
- [ ] IAM and service accounts set up
- [ ] Model artifacts uploaded to Cloud Storage
- [ ] Model registered in Vertex AI Model Registry
- [ ] Endpoint created and model deployed
- [ ] Endpoint tested with sample predictions
- [ ] Model monitoring configured
- [ ] Vertex AI Pipeline created (optional)
- [ ] CI/CD pipeline configured (optional)
- [ ] Logging configured
- [ ] Rollback procedure documented
- [ ] Cost estimate reviewed
- [ ] Code committed to git

---

## Testing Strategy

### Unit Tests

```python
# tests/test_data_loader.py
def test_load_orders():
    df = load_orders('data/raw/')
    assert len(df) > 0
    assert 'order_id' in df.columns

# tests/test_features.py
def test_rfm_features():
    sample_data = pd.DataFrame({...})
    features = compute_rfm_features(sample_data)
    assert 'recency' in features.columns
    assert features['recency'].min() >= 0

# tests/test_model.py
def test_classifier_prediction_shape():
    model = load('models/satisfaction_classifier.joblib')
    X = np.random.randn(10, n_features)
    preds = model.predict(X)
    assert preds.shape == (10,)
```

### Integration Tests

```python
# tests/test_integration.py
def test_end_to_end_pipeline():
    # Load sample data
    sample = pd.read_csv('tests/fixtures/sample_input.csv')

    # Run through full pipeline
    pipeline = load('models/feature_pipeline.joblib')
    classifier = load('models/satisfaction_classifier.joblib')

    features = pipeline.transform(sample)
    predictions = classifier.predict(features)

    assert len(predictions) == len(sample)
```

---

## Expected Deliverables

1. **Jupyter Notebooks**:
   - `00_setup_validation.ipynb` - Environment verification
   - `01_data_exploration.ipynb` - EDA and data quality
   - `02_feature_engineering.ipynb` - Feature pipeline creation
   - `03_unsupervised_learning.ipynb` - Clustering experiments
   - `04_supervised_learning.ipynb` - Classification & regression
   - `05_gcp_deployment.ipynb` - Deployment and monitoring

2. **Source Code**:
   - `src/data_loader.py` - Data loading utilities
   - `src/feature_engineering.py` - Feature transformations
   - `src/train.py` - Training scripts
   - `src/predict.py` - Inference scripts

3. **Trained Models & Artifacts**:
   - `models/feature_pipeline.joblib`
   - `models/clustering_model.joblib`
   - `models/satisfaction_classifier.joblib`
   - `models/delivery_predictor.joblib`
   - `models/model_metadata.json`

4. **GCP Resources**:
   - Model in Vertex AI Model Registry
   - Live endpoint for predictions
   - Monitoring dashboard
   - (Optional) Automated pipeline

5. **Tests**:
   - Unit tests for data loading
   - Unit tests for feature engineering
   - Unit tests for model inference
   - Integration tests for full pipeline

6. **Documentation**:
   - Model cards (performance, limitations, intended use)
   - Feature documentation (description, source, transformation)
   - API documentation (endpoint usage, input/output format)
   - Runbook (deployment, monitoring, rollback procedures)

---

## Decision Tracking

Use this table to track decisions made during development:

| Decision Point | Options | Chosen | Rationale | Date |
|----------------|---------|--------|-----------|------|
| Python environment | venv / conda / poetry | TBD | | |
| Train/val/test split ratio | 70-15-15 / 80-10-10 / 60-20-20 | TBD | | |
| Split strategy | Random / Time-based / Customer-based | TBD | | |
| Data versioning | DVC / Git LFS / Manual | TBD | | |
| Schema validation | Pandera / Great Expectations / Custom | TBD | | |
| Missing value imputation | (per column) | TBD | | |
| Outlier handling | IQR / Z-score / Isolation Forest | TBD | | |
| Numerical scaling | Standard / MinMax / Robust | TBD | | |
| Categorical encoding | One-Hot / Target / Binary | TBD | | |
| Cyclical time encoding | Raw / Sine-Cosine / One-Hot | TBD | | |
| Sentiment analysis | VADER / TextBlob / Transformer | TBD | | |
| Dimensionality reduction | None / PCA / Feature selection | TBD | | |
| Final clustering algorithm | K-Means / DBSCAN / Hierarchical / GMM | TBD | | |
| Experiment tracking | MLflow / W&B / Vertex AI | TBD | | |
| CV strategy | StratifiedKFold / TimeSeriesSplit | TBD | | |
| Class imbalance handling | Weights / SMOTE / Undersampling | TBD | | |
| Hyperparameter tuning | Grid / Random / Bayesian | TBD | | |
| Classification primary metric | F1 / ROC-AUC / Precision / Recall | TBD | | |
| Regression target transform | None / Log / Box-Cox | TBD | | |
| Boosting library | XGBoost / LightGBM | TBD | | |
| Container approach | Pre-built / Custom Docker | TBD | | |
| CI/CD platform | GitHub Actions / Cloud Build | TBD | | |

---

## Key Resources

- **Dataset**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **GCP MLOps Guide**: https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai
- **Vertex AI Docs**: https://cloud.google.com/vertex-ai/docs
