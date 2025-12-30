# Day 1: Data Engineering & EDA - Full Plan

## Overview

| Aspect | Details |
|--------|---------|
| Duration | 6-8 hours |
| Goal | Load data, create splits, perform EDA, clean data |
| Output | Train/val/test splits + processed data + EDA insights |

---

## Key Principles

1. **Split BEFORE EDA** - Prevent data leakage
2. **EDA on training set ONLY** - Never peek at test data
3. **Fit transformers on train** - Apply same to val/test
4. **Document all decisions** - Reproducibility matters

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Split Strategy** | Time-based | More realistic for production (train on past, predict future) |
| **Split Ratios** | 70/15/15 | Balanced between training data and evaluation |
| **Schema Validation** | Pandera | Lightweight, pandas-native, already installed |
| **Outlier Detection** | IQR method | Simple, interpretable, works well for most cases |
| **Outlier Treatment** | Cap (winsorize) | Preserves data points while limiting extreme values |

---

## Dataset Overview

### 9 CSV Files to Load

| File | Rows (approx) | Key Columns | Purpose |
|------|---------------|-------------|---------|
| `olist_orders_dataset.csv` | 99,441 | `order_id`, `customer_id`, timestamps | Core orders |
| `olist_order_items_dataset.csv` | 112,650 | `order_id`, `product_id`, `seller_id`, `price` | Items per order |
| `olist_products_dataset.csv` | 32,951 | `product_id`, `category`, dimensions | Product catalog |
| `olist_customers_dataset.csv` | 99,441 | `customer_id`, `zip_code`, `state` | Customer info |
| `olist_sellers_dataset.csv` | 3,095 | `seller_id`, `zip_code`, `state` | Seller info |
| `olist_order_payments_dataset.csv` | 103,886 | `order_id`, `payment_type`, `value` | Payments |
| `olist_order_reviews_dataset.csv` | 99,224 | `order_id`, `review_score`, `comment` | Reviews (1-5) |
| `olist_geolocation_dataset.csv` | 1,000,163 | `zip_code`, `lat`, `lng` | Geo coordinates |
| `product_category_name_translation.csv` | 71 | Category translations | PT → EN |

### Table Relationships

```
orders (1) ←→ (N) order_items
orders (1) ←→ (N) payments
orders (1) ←→ (1) reviews
orders (N) ←→ (1) customers
order_items (N) ←→ (1) products
order_items (N) ←→ (1) sellers
customers/sellers ←→ geolocation (via zip_code)
products ←→ category_translation (via category_name)
```

---

## Task Breakdown

### Task 1: Data Loading and Initial Inspection

**Objective**: Load all 9 CSVs and verify data integrity

```python
import pandas as pd
from pathlib import Path

# Load all datasets
data_path = Path("data/raw/")
datasets = {
    "orders": pd.read_csv(data_path / "olist_orders_dataset.csv"),
    "order_items": pd.read_csv(data_path / "olist_order_items_dataset.csv"),
    "products": pd.read_csv(data_path / "olist_products_dataset.csv"),
    "customers": pd.read_csv(data_path / "olist_customers_dataset.csv"),
    "sellers": pd.read_csv(data_path / "olist_sellers_dataset.csv"),
    "payments": pd.read_csv(data_path / "olist_order_payments_dataset.csv"),
    "reviews": pd.read_csv(data_path / "olist_order_reviews_dataset.csv"),
    "geolocation": pd.read_csv(data_path / "olist_geolocation_dataset.csv"),
    "category_translation": pd.read_csv(data_path / "product_category_name_translation.csv"),
}
```

**Validation Checks**:
- [ ] All 9 files loaded successfully
- [ ] Row counts match expected values
- [ ] No corrupted files (no parse errors)
- [ ] Column names match documentation

**For each dataset, check**:
```python
df.shape          # rows, columns
df.dtypes         # data types
df.info()         # memory usage, non-null counts
df.head()         # sample data
df.describe()     # basic statistics
```

---

### Task 2: Parse Date Columns

**Objective**: Convert string dates to datetime objects

```python
# Orders dataset - parse all timestamp columns
date_columns = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col])

# Reviews dataset
reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
reviews['review_answer_timestamp'] = pd.to_datetime(reviews['review_answer_timestamp'])
```

---

### Task 3: Train/Validation/Test Split (CRITICAL - DO FIRST!)

**Objective**: Create time-based split before any EDA

**Strategy**: Time-based split using `order_purchase_timestamp`
- Train: Oldest 70% of orders
- Validation: Next 15% of orders
- Test: Newest 15% of orders

```python
# Sort by purchase timestamp
orders_sorted = orders.sort_values('order_purchase_timestamp')

# Calculate split indices
n = len(orders_sorted)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Split order_ids
train_order_ids = orders_sorted.iloc[:train_end]['order_id'].values
val_order_ids = orders_sorted.iloc[train_end:val_end]['order_id'].values
test_order_ids = orders_sorted.iloc[val_end:]['order_id'].values

# Verify no overlap
assert len(set(train_order_ids) & set(val_order_ids)) == 0
assert len(set(train_order_ids) & set(test_order_ids)) == 0
assert len(set(val_order_ids) & set(test_order_ids)) == 0

print(f"Train: {len(train_order_ids):,} orders")
print(f"Val:   {len(val_order_ids):,} orders")
print(f"Test:  {len(test_order_ids):,} orders")
```

**Save split order IDs for later use**:
```python
import json

splits = {
    "train": train_order_ids.tolist(),
    "val": val_order_ids.tolist(),
    "test": test_order_ids.tolist()
}

with open("data/splits/order_id_splits.json", "w") as f:
    json.dump(splits, f)
```

---

### Task 4: Data Merging

**Objective**: Create unified dataset by merging tables

**Merge Order** (start with orders as base):

```python
# Step 1: Orders + Order Items
merged = orders.merge(order_items, on='order_id', how='left')
print(f"After order_items: {len(merged):,} rows")

# Step 2: + Products (with category translation)
products_translated = products.merge(
    category_translation,
    on='product_category_name',
    how='left'
)
merged = merged.merge(products_translated, on='product_id', how='left')
print(f"After products: {len(merged):,} rows")

# Step 3: + Customers
merged = merged.merge(customers, on='customer_id', how='left')
print(f"After customers: {len(merged):,} rows")

# Step 4: + Sellers
merged = merged.merge(
    sellers,
    on='seller_id',
    how='left',
    suffixes=('_customer', '_seller')
)
print(f"After sellers: {len(merged):,} rows")

# Step 5: + Reviews
merged = merged.merge(reviews, on='order_id', how='left')
print(f"After reviews: {len(merged):,} rows")

# Step 6: + Payments (aggregate per order first)
payments_agg = payments.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'max',
    'payment_type': lambda x: x.mode()[0] if len(x) > 0 else None
}).reset_index()
merged = merged.merge(payments_agg, on='order_id', how='left')
print(f"After payments: {len(merged):,} rows")
```

**Merge Validation**:
- [ ] Document row counts at each step
- [ ] Check for unexpected row multiplication
- [ ] Identify orphan records (orders without items, etc.)

---

### Task 5: Create Target Variables

**Objective**: Define classification and regression targets

```python
# Classification target: Customer Satisfaction
# satisfied (1) if review_score >= 4, else unsatisfied (0)
merged['is_satisfied'] = (merged['review_score'] >= 4).astype(int)

# Regression target: Delivery Time in Days
merged['delivery_days'] = (
    merged['order_delivered_customer_date'] -
    merged['order_purchase_timestamp']
).dt.total_seconds() / (24 * 3600)

# Additional useful features
merged['is_late_delivery'] = (
    merged['order_delivered_customer_date'] >
    merged['order_estimated_delivery_date']
).astype(int)

merged['delivery_delay_days'] = (
    merged['order_delivered_customer_date'] -
    merged['order_estimated_delivery_date']
).dt.total_seconds() / (24 * 3600)
```

---

### Task 6: Split Merged Data

**Objective**: Apply order_id splits to merged dataset

```python
# Split merged data using order_ids
train_df = merged[merged['order_id'].isin(train_order_ids)].copy()
val_df = merged[merged['order_id'].isin(val_order_ids)].copy()
test_df = merged[merged['order_id'].isin(test_order_ids)].copy()

print(f"Train: {len(train_df):,} rows")
print(f"Val:   {len(val_df):,} rows")
print(f"Test:  {len(test_df):,} rows")

# Save splits
train_df.to_parquet("data/splits/train.parquet", index=False)
val_df.to_parquet("data/splits/val.parquet", index=False)
test_df.to_parquet("data/splits/test.parquet", index=False)
```

---

### Task 7: Exploratory Data Analysis (TRAIN SET ONLY!)

**Objective**: Understand data patterns using training set only

#### 7.1 Basic Statistics
```python
# Numerical columns
train_df.describe()

# Categorical columns
for col in train_df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:")
    print(train_df[col].value_counts().head(10))
```

#### 7.2 Target Variable Analysis
```python
# Classification target distribution
print("Satisfaction Distribution:")
print(train_df['is_satisfied'].value_counts(normalize=True))

# Check for class imbalance
# If ratio > 80/20, consider balancing strategies

# Regression target distribution
train_df['delivery_days'].describe()
train_df['delivery_days'].hist(bins=50)
```

#### 7.3 Temporal Analysis
```python
# Orders over time
train_df.groupby(train_df['order_purchase_timestamp'].dt.to_period('M')).size().plot()

# Day of week patterns
train_df['order_purchase_timestamp'].dt.dayofweek.value_counts().sort_index().plot(kind='bar')

# Hour of day patterns
train_df['order_purchase_timestamp'].dt.hour.value_counts().sort_index().plot(kind='bar')
```

#### 7.4 Geographic Analysis
```python
# Orders by state
train_df['customer_state'].value_counts().plot(kind='bar')

# Revenue by state
train_df.groupby('customer_state')['payment_value'].sum().sort_values(ascending=False).plot(kind='bar')
```

#### 7.5 Product Analysis
```python
# Top categories
train_df['product_category_name_english'].value_counts().head(20)

# Price distribution
train_df['price'].describe()
train_df['price'].hist(bins=50)
```

#### 7.6 Correlation Analysis
```python
# Numerical correlations
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
correlation_matrix = train_df[numerical_cols].corr()

# Plot heatmap
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
```

---

### Task 8: Missing Value Analysis

**Objective**: Identify and document missing value patterns

```python
# Missing value counts
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df) * 100).round(2)
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).sort_values('missing_pct', ascending=False)

print(missing_df[missing_df['missing_count'] > 0])
```

**Expected Missing Values**:

| Column | Expected Missing | Strategy |
|--------|------------------|----------|
| `order_approved_at` | Some | Keep as-is (valid for cancelled orders) |
| `order_delivered_carrier_date` | Some | Keep as-is (not yet shipped) |
| `order_delivered_customer_date` | Some | Keep as-is (not yet delivered) |
| `review_comment_message` | Many | Fill with empty string |
| `review_comment_title` | Many | Fill with empty string |
| `product_category_name` | Few | Fill with "unknown" |
| `product_weight_g` | Few | Fill with median |
| `product_dimensions` | Few | Fill with median |

---

### Task 9: Data Quality Checks

**Objective**: Identify and handle data quality issues

#### 9.1 Duplicates
```python
# Check for duplicate orders
duplicate_orders = train_df[train_df.duplicated(subset=['order_id'], keep=False)]
print(f"Duplicate order_ids: {len(duplicate_orders)}")
# Note: Duplicates expected due to multiple items per order
```

#### 9.2 Outlier Detection (IQR Method)
```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

# Check key numerical columns
for col in ['price', 'freight_value', 'payment_value', 'delivery_days']:
    outliers, lower, upper = detect_outliers_iqr(train_df, col)
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(train_df)*100:.2f}%)")
    print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")
```

#### 9.3 Outlier Treatment (Winsorization)
```python
def winsorize_column(df, column, lower_pct=0.01, upper_pct=0.99):
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    df[column] = df[column].clip(lower, upper)
    return df

# Apply to key columns
for col in ['price', 'freight_value', 'payment_value']:
    train_df = winsorize_column(train_df, col)
```

#### 9.4 Logical Constraints
```python
# Delivery date should be after purchase date
invalid_delivery = train_df[
    train_df['order_delivered_customer_date'] < train_df['order_purchase_timestamp']
]
print(f"Invalid delivery dates: {len(invalid_delivery)}")

# Price should be positive
invalid_price = train_df[train_df['price'] <= 0]
print(f"Invalid prices: {len(invalid_price)}")
```

---

### Task 10: Schema Validation with Pandera

**Objective**: Define and validate data schema

```python
import pandera as pa
from pandera import Column, Check

# Define schema
order_schema = pa.DataFrameSchema({
    "order_id": Column(str, nullable=False, unique=False),
    "customer_id": Column(str, nullable=False),
    "order_status": Column(str, Check.isin([
        'delivered', 'shipped', 'canceled', 'unavailable',
        'invoiced', 'processing', 'created', 'approved'
    ])),
    "price": Column(float, Check.ge(0), nullable=True),
    "freight_value": Column(float, Check.ge(0), nullable=True),
    "review_score": Column(float, Check.in_range(1, 5), nullable=True),
    "is_satisfied": Column(int, Check.isin([0, 1]), nullable=True),
    "delivery_days": Column(float, Check.ge(0), nullable=True),
})

# Validate
try:
    order_schema.validate(train_df, lazy=True)
    print("Schema validation passed!")
except pa.errors.SchemaErrors as e:
    print(f"Schema validation failed:\n{e}")
```

---

### Task 11: Save Processed Data & Artifacts

**Objective**: Persist all processed data and artifacts

```python
import joblib
import json

# Save processed splits
train_df.to_parquet("data/processed/train_processed.parquet", index=False)
val_df.to_parquet("data/processed/val_processed.parquet", index=False)
test_df.to_parquet("data/processed/test_processed.parquet", index=False)

# Save EDA statistics
eda_stats = {
    "train_rows": len(train_df),
    "val_rows": len(val_df),
    "test_rows": len(test_df),
    "satisfaction_rate": float(train_df['is_satisfied'].mean()),
    "avg_delivery_days": float(train_df['delivery_days'].mean()),
    "missing_values": train_df.isnull().sum().to_dict(),
    "date_range": {
        "train_start": str(train_df['order_purchase_timestamp'].min()),
        "train_end": str(train_df['order_purchase_timestamp'].max()),
        "val_start": str(val_df['order_purchase_timestamp'].min()),
        "val_end": str(val_df['order_purchase_timestamp'].max()),
        "test_start": str(test_df['order_purchase_timestamp'].min()),
        "test_end": str(test_df['order_purchase_timestamp'].max()),
    }
}

with open("data/processed/eda_statistics.json", "w") as f:
    json.dump(eda_stats, f, indent=2)

print("All data and artifacts saved!")
```

---

## Key Visualizations to Create

| Visualization | Purpose | Tool |
|---------------|---------|------|
| Orders over time | Temporal trends | matplotlib/plotly |
| Review score distribution | Target imbalance check | seaborn |
| Delivery time distribution | Regression target analysis | matplotlib |
| Revenue by state | Geographic patterns | plotly choropleth |
| Category distribution | Product analysis | seaborn barplot |
| Correlation heatmap | Feature relationships | seaborn heatmap |
| Missing value heatmap | Data quality | seaborn/missingno |

---

## Day 1 Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Load all 9 raw CSV files | ☐ |
| 2 | Parse date columns | ☐ |
| 3 | Create time-based train/val/test split | ☐ |
| 4 | Merge all datasets | ☐ |
| 5 | Create target variables (is_satisfied, delivery_days) | ☐ |
| 6 | Split merged data and save parquet files | ☐ |
| 7 | Complete EDA on training set only | ☐ |
| 8 | Analyze and document missing values | ☐ |
| 9 | Detect and handle outliers | ☐ |
| 10 | Define and validate data schema with Pandera | ☐ |
| 11 | Save processed data and EDA artifacts | ☐ |
| 12 | Create key visualizations | ☐ |
| 13 | Document all decisions | ☐ |
| 14 | Commit code to git | ☐ |

---

## Expected Outputs

### Files Created

```
data/
├── splits/
│   ├── order_id_splits.json      # Order IDs for each split
│   ├── train.parquet             # Raw train split
│   ├── val.parquet               # Raw val split
│   └── test.parquet              # Raw test split
├── processed/
│   ├── train_processed.parquet   # Cleaned train data
│   ├── val_processed.parquet     # Cleaned val data
│   ├── test_processed.parquet    # Cleaned test data
│   └── eda_statistics.json       # EDA summary stats
```

### Notebook Created

```
notebooks/
└── 01_data_exploration.ipynb     # Complete EDA notebook
```

---

## Decision Log

| Decision | Options Considered | Chosen | Rationale |
|----------|-------------------|--------|-----------|
| Split Strategy | Random / Time-based / Customer-based | Time-based | Realistic for production |
| Split Ratio | 70-15-15 / 80-10-10 / 60-20-20 | 70-15-15 | Balanced |
| Outlier Method | IQR / Z-score / Isolation Forest | IQR | Simple, interpretable |
| Outlier Treatment | Remove / Cap / Transform | Cap (winsorize) | Preserves data |
| Schema Validation | Pandera / Great Expectations / Custom | Pandera | Lightweight |

---

## Next Steps

After completing Day 1, proceed to **Day 2: Feature Engineering & Clustering** where you will:
1. Create sklearn.Pipeline for feature transformations
2. Build RFM, temporal, geographic, and NLP features
3. Implement and compare clustering algorithms
4. Add cluster labels as features for supervised learning
