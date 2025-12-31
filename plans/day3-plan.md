# Day 3: Supervised Learning - Full Plan

## Overview

| Aspect | Details |
|--------|---------|
| Duration | 6-8 hours |
| Goal | Train classification and regression models for customer satisfaction and delivery prediction |
| Output | satisfaction_classifier.joblib + delivery_predictor.joblib + model_metadata.json |

---

## Key Principles

1. **Baselines FIRST** - Always establish dummy baselines before training real models
2. **Cross-validation** - Use StratifiedKFold for robust performance estimates
3. **No data leakage** - Only fit transformers on training data
4. **Test set is sacred** - Evaluate on test set ONLY ONCE at the very end
5. **Interpretability matters** - Use SHAP for model explanations

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Experiment Tracking** | Vertex AI Experiments | Already using GCP, native integration |
| **Class Imbalance** | Class weights | Moderate imbalance (72.5/27.5), weights are simple and effective |
| **Cross-Validation** | StratifiedKFold (k=5) | Maintains class distribution, good bias-variance tradeoff |
| **Tuning Method** | RandomizedSearchCV | Faster than grid search, good coverage |
| **Primary Classification Metric** | ROC-AUC | Robust to class imbalance, threshold-independent |
| **Primary Regression Metric** | RMSE | Penalizes large errors, interpretable in days |
| **Gradient Boosting** | LightGBM | Faster than XGBoost, handles categoricals natively |
| **Target Transformation** | Log1p for delivery_days | Right-skewed distribution (mean > median) |

---

## Input Data Summary (from Day 2)

### Dataset Sizes

| Split | Rows | Purpose |
|-------|------|---------|
| Train | 79,986 | Model training |
| Validation | 17,239 | Hyperparameter tuning |
| Test | 16,867 | Final evaluation (once) |

### Target Variables

#### Classification: `is_satisfied`

| Split | Satisfied (1) | Unsatisfied (0) | Imbalance Ratio |
|-------|---------------|-----------------|-----------------|
| Train | 72.5% | 27.5% | 2.6:1 |
| Val | 77.9% | 22.1% | 3.5:1 |
| Test | 80.0% | 20.0% | 4.0:1 |

> **Note**: Distribution differs across splits (time-based split). Satisfaction improved over time.

#### Regression: `delivery_days`

| Split | Mean | Median | Std | Min | Max |
|-------|------|--------|-----|-----|-----|
| Train | 13.6 | 11.4 | 8.9 | 2.0 | 48.3 |
| Val | 10.7 | 8.9 | 7.3 | - | - |
| Test | 8.3 | 7.2 | 5.2 | - | - |

> **Note**: Right-skewed (mean > median). Delivery times improved over time. 2,624 missing values.

### Available Features (88 total)

| Category | Count | Examples |
|----------|-------|----------|
| **Numerical** | 21 | price, freight_value, product_weight_g, seller_customer_distance_km |
| **Categorical** | 6 | customer_state, seller_state, product_category_name_english, payment_type |
| **Binary** | 8 | is_weekend, is_same_state, is_late_delivery, has_review_comment |
| **RFM** | 5 | recency, frequency, monetary, avg_order_value |
| **Temporal** | 8 | order_hour_sin/cos, order_dayofweek_sin/cos |
| **Geographic** | 6 | customer_lat/lng, seller_lat/lng, distance_km |
| **Product** | 6 | product_volume_cm3, freight_ratio, price_vs_category_zscore |
| **NLP** | 6 | review_sentiment_polarity, review_word_count |

### Features to EXCLUDE from Models

| Feature | Reason |
|---------|--------|
| `review_score` | Target leakage for classification |
| `is_satisfied` | Target variable for classification |
| `delivery_days` | Target variable for regression |
| `is_late_delivery` | Target leakage for regression |
| `delivery_delay_days` | Target leakage for regression |
| `review_*` columns | Only available after delivery (leakage for delivery prediction) |
| ID columns | Not predictive |
| Timestamp columns | Use derived features instead |

---

## Task Breakdown

### Task 1: Setup & Data Preparation

**Objective**: Load data, define feature sets, create preprocessing pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_parquet('data/processed/train_featured.parquet')
val = pd.read_parquet('data/processed/val_featured.parquet')
test = pd.read_parquet('data/processed/test_featured.parquet')

# Define feature sets
CLASSIFICATION_FEATURES = [
    # Delivery-related (key predictors)
    'delivery_days', 'is_late_delivery', 'delivery_delay_days',

    # Product
    'price', 'freight_value', 'product_weight_g', 'product_volume_cm3',
    'freight_ratio', 'price_vs_category_zscore',

    # Seller/Geographic
    'seller_customer_distance_km', 'is_same_state',

    # Customer behavior
    'recency', 'frequency', 'monetary',

    # Order
    'payment_value', 'payment_installments', 'is_full_payment',

    # Temporal
    'order_hour_sin', 'order_hour_cos', 'is_weekend',
]

REGRESSION_FEATURES = [
    # Product (weight affects shipping time)
    'product_weight_g', 'product_volume_cm3', 'price',

    # Geographic (distance is key predictor)
    'seller_customer_distance_km', 'is_same_state',
    'customer_lat', 'customer_lng', 'seller_lat', 'seller_lng',

    # Seller
    'seller_state',  # categorical - some states have better logistics

    # Temporal (when order was placed)
    'order_hour', 'order_dayofweek', 'is_weekend',

    # Order details
    'freight_value',  # higher freight often = faster shipping
]

CATEGORICAL_FEATURES = [
    'customer_state', 'seller_state', 'customer_region', 'seller_region',
    'product_category_name_english', 'payment_type',
]
```

**Preprocessing Pipeline**:

```python
# Numerical preprocessing
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# Combined preprocessor
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features),
], remainder='drop')
```

---

### Task 2: Establish Baselines

**Objective**: Create dummy baselines to beat

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Classification baseline
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train_clf)
baseline_accuracy = accuracy_score(y_val_clf, dummy_clf.predict(X_val))
baseline_roc_auc = 0.5  # Random guessing

print(f"Classification Baseline (most frequent):")
print(f"  Accuracy: {baseline_accuracy:.3f}")
print(f"  ROC-AUC: {baseline_roc_auc:.3f}")

# Regression baseline
dummy_reg = DummyRegressor(strategy='mean')
dummy_reg.fit(X_train, y_train_reg)
baseline_rmse = np.sqrt(mean_squared_error(y_val_reg, dummy_reg.predict(X_val)))
baseline_mae = mean_absolute_error(y_val_reg, dummy_reg.predict(X_val))

print(f"\nRegression Baseline (mean):")
print(f"  RMSE: {baseline_rmse:.2f} days")
print(f"  MAE: {baseline_mae:.2f} days")
```

**Expected Baselines**:

| Task | Metric | Baseline | Target |
|------|--------|----------|--------|
| Classification | Accuracy | ~72.5% (majority class) | >75% |
| Classification | ROC-AUC | 0.50 (random) | >0.70 |
| Regression | RMSE | ~8.9 days (std) | <7 days |
| Regression | MAE | ~6-7 days | <5 days |

---

### Task 3: Classification Models

**Objective**: Train and compare classification models for customer satisfaction

#### 3.1 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# With class weights to handle imbalance
log_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    ))
])

log_reg.fit(X_train, y_train)
```

#### 3.2 Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42,
    ))
])
```

#### 3.3 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    ))
])
```

#### 3.4 LightGBM

```python
from lightgbm import LGBMClassifier

lgbm_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    ))
])
```

#### 3.5 Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

# LightGBM parameter grid
lgbm_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, 20],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__num_leaves': [31, 50, 100],
    'classifier__min_child_samples': [10, 20, 50],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.8, 0.9, 1.0],
}

search = RandomizedSearchCV(
    lgbm_clf,
    lgbm_param_grid,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
)

search.fit(X_train, y_train)
best_clf = search.best_estimator_
```

#### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)

def evaluate_classifier(model, X, y, name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        'name': name,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
    }
```

**Classification Model Comparison Table**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time |
|-------|----------|-----------|--------|-----|---------|------------|
| Dummy Baseline | 72.5% | - | - | - | 0.50 | - |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD | TBD |
| **LightGBM (tuned)** | TBD | TBD | TBD | TBD | TBD | TBD |

---

### Task 4: Regression Models

**Objective**: Train and compare regression models for delivery time prediction

#### 4.1 Handle Missing Values & Target Transformation

```python
# Remove rows with missing delivery_days for regression
train_reg = train.dropna(subset=['delivery_days'])
val_reg = val.dropna(subset=['delivery_days'])
test_reg = test.dropna(subset=['delivery_days'])

# Log transform target (right-skewed)
y_train_reg = np.log1p(train_reg['delivery_days'])
y_val_reg = np.log1p(val_reg['delivery_days'])
y_test_reg = np.log1p(test_reg['delivery_days'])

# Remember to inverse transform predictions!
# y_pred_original = np.expm1(y_pred_log)
```

#### 4.2 Linear Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

linear_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

ridge_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

lasso_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))
])
```

#### 4.3 Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
    ))
])
```

#### 4.4 LightGBM Regressor

```python
from lightgbm import LGBMRegressor

lgbm_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    ))
])
```

#### 4.5 Hyperparameter Tuning

```python
lgbm_reg_param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [5, 10, 15, 20],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__num_leaves': [31, 50, 100],
    'regressor__min_child_samples': [10, 20, 50],
    'regressor__subsample': [0.8, 0.9, 1.0],
}

search_reg = RandomizedSearchCV(
    lgbm_reg,
    lgbm_reg_param_grid,
    n_iter=50,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
)

search_reg.fit(X_train_reg, y_train_reg)
best_reg = search_reg.best_estimator_
```

#### Regression Metrics

```python
def evaluate_regressor(model, X, y_log, name):
    """Evaluate regressor. y_log is log-transformed target."""
    y_pred_log = model.predict(X)

    # Inverse transform for interpretable metrics
    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)

    return {
        'name': name,
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }
```

**Regression Model Comparison Table**:

| Model | RMSE (days) | MAE (days) | R² | MAPE (%) | Train Time |
|-------|-------------|------------|-----|----------|------------|
| Dummy (mean) | ~8.9 | ~6.5 | 0.0 | - | - |
| Linear Regression | TBD | TBD | TBD | TBD | TBD |
| Ridge | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD |
| **LightGBM (tuned)** | TBD | TBD | TBD | TBD | TBD |

---

### Task 5: Model Interpretability (SHAP)

**Objective**: Understand feature importance and model decisions

```python
import shap

# For tree-based models
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])

# Get feature names after preprocessing
feature_names = (
    preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
    preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
)

# Transform data
X_val_transformed = preprocessor.transform(X_val)

# Calculate SHAP values
shap_values = explainer.shap_values(X_val_transformed)

# Summary plot (global importance)
shap.summary_plot(shap_values, X_val_transformed, feature_names=feature_names)

# Bar plot (mean absolute SHAP values)
shap.summary_plot(shap_values, X_val_transformed, feature_names=feature_names, plot_type='bar')

# Single prediction explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_val_transformed[0],
    feature_names=feature_names
)
```

**Expected Key Features**:

| Task | Expected Top Features |
|------|----------------------|
| Classification (Satisfaction) | delivery_days, is_late_delivery, delivery_delay_days, price, freight_value |
| Regression (Delivery Time) | seller_customer_distance_km, product_weight_g, freight_value, seller_state |

---

### Task 6: Final Model Selection & Test Evaluation

**Objective**: Select best models and evaluate on test set (ONCE)

```python
# Select best models based on validation performance
print("Best Classification Model: LightGBM (tuned)")
print("Best Regression Model: LightGBM (tuned)")

# Final evaluation on test set
print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

# Classification
test_clf_metrics = evaluate_classifier(best_clf, X_test, y_test_clf, 'Best Classifier')
print(f"\nClassification (Test Set):")
print(f"  Accuracy: {test_clf_metrics['accuracy']:.3f}")
print(f"  ROC-AUC: {test_clf_metrics['roc_auc']:.3f}")
print(f"  F1: {test_clf_metrics['f1']:.3f}")

# Regression
test_reg_metrics = evaluate_regressor(best_reg, X_test_reg, y_test_reg, 'Best Regressor')
print(f"\nRegression (Test Set):")
print(f"  RMSE: {test_reg_metrics['rmse']:.2f} days")
print(f"  MAE: {test_reg_metrics['mae']:.2f} days")
print(f"  R²: {test_reg_metrics['r2']:.3f}")
```

---

### Task 7: Save Models & Artifacts

**Objective**: Serialize models and metadata for deployment

```python
import joblib
import json
from datetime import datetime

# Save best classifier
joblib.dump(best_clf, 'models/satisfaction_classifier.joblib')
print("Saved: models/satisfaction_classifier.joblib")

# Save best regressor
joblib.dump(best_reg, 'models/delivery_predictor.joblib')
print("Saved: models/delivery_predictor.joblib")

# Save model metadata
model_metadata = {
    'classifier': {
        'algorithm': 'LightGBM',
        'hyperparameters': best_clf.named_steps['classifier'].get_params(),
        'features': CLASSIFICATION_FEATURES,
        'metrics': {
            'val': val_clf_metrics,
            'test': test_clf_metrics,
        },
        'training_date': datetime.now().isoformat(),
        'class_distribution': {
            'train': {'satisfied': 0.725, 'unsatisfied': 0.275},
        },
    },
    'regressor': {
        'algorithm': 'LightGBM',
        'hyperparameters': best_reg.named_steps['regressor'].get_params(),
        'features': REGRESSION_FEATURES,
        'target_transform': 'log1p',
        'metrics': {
            'val': val_reg_metrics,
            'test': test_reg_metrics,
        },
        'training_date': datetime.now().isoformat(),
    },
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2, default=str)
print("Saved: models/model_metadata.json")
```

---

## Day 3 Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Load data and define feature sets | ☐ |
| 2 | Create preprocessing pipeline | ☐ |
| 3 | Establish baseline models (dummy) | ☐ |
| 4 | Train Logistic Regression | ☐ |
| 5 | Train Decision Tree | ☐ |
| 6 | Train Random Forest (classification) | ☐ |
| 7 | Train LightGBM (classification) | ☐ |
| 8 | Hyperparameter tuning (classification) | ☐ |
| 9 | Train Linear/Ridge/Lasso Regression | ☐ |
| 10 | Train Random Forest (regression) | ☐ |
| 11 | Train LightGBM (regression) | ☐ |
| 12 | Hyperparameter tuning (regression) | ☐ |
| 13 | Compare all classification models | ☐ |
| 14 | Compare all regression models | ☐ |
| 15 | SHAP analysis (classification) | ☐ |
| 16 | SHAP analysis (regression) | ☐ |
| 17 | Final evaluation on test set | ☐ |
| 18 | Save models and metadata | ☐ |
| 19 | Create notebook 04_supervised_learning.ipynb | ☐ |
| 20 | Commit code to git | ☐ |

---

## Expected Outputs

### Files Created

```
models/
├── satisfaction_classifier.joblib    # Best classification model
├── delivery_predictor.joblib         # Best regression model
├── model_metadata.json               # Model info and metrics
└── plots/
    ├── classification_roc_curves.png
    ├── classification_confusion_matrix.png
    ├── regression_residuals.png
    ├── regression_predictions_vs_actual.png
    ├── shap_classification_summary.png
    └── shap_regression_summary.png

notebooks/
└── 04_supervised_learning.ipynb      # Complete notebook
```

### Expected Performance

| Task | Metric | Baseline | Target | Stretch |
|------|--------|----------|--------|---------|
| Classification | ROC-AUC | 0.50 | >0.70 | >0.80 |
| Classification | F1 | - | >0.60 | >0.70 |
| Regression | RMSE | 8.9 days | <7 days | <5 days |
| Regression | MAE | 6.5 days | <5 days | <4 days |

---

## Decision Log

| Decision | Options Considered | Chosen | Rationale |
|----------|-------------------|--------|-----------|
| Experiment Tracking | MLflow / W&B / Vertex AI | Vertex AI Experiments | GCP native, deployment integration |
| Class Imbalance | Weights / SMOTE / Undersampling | Class weights | Simple, effective for moderate imbalance |
| Cross-Validation | KFold / StratifiedKFold / TimeSeriesSplit | StratifiedKFold (k=5) | Maintains class distribution |
| Tuning Method | Grid / Random / Bayesian | RandomizedSearchCV | Efficient, good coverage |
| Primary Clf Metric | Accuracy / F1 / ROC-AUC | ROC-AUC | Threshold-independent, handles imbalance |
| Primary Reg Metric | RMSE / MAE / R² | RMSE | Penalizes large errors |
| Boosting Library | XGBoost / LightGBM | LightGBM | Faster, native categorical support |
| Target Transform | None / Log / Box-Cox | Log1p | Simple, handles right skew |

---

## Next Steps

After completing Day 3, proceed to **Day 4: GCP Deployment & MLOps** where you will:
1. Test inference pipeline locally
2. Upload models to Vertex AI Model Registry
3. Deploy prediction endpoints
4. Create batch prediction pipeline
5. Set up monitoring and logging
6. Document API endpoints
