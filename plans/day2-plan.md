# Day 2: Feature Engineering & Unsupervised Learning - Full Plan

## Overview

| Aspect | Details |
|--------|---------|
| Duration | 6-8 hours |
| Goal | Build feature pipeline + customer segmentation via clustering |
| Output | feature_pipeline.joblib + clustering_model.joblib + cluster-enriched data |

---

## Key Principles

1. **Use sklearn.Pipeline** - Ensures reproducibility and prevents data leakage
2. **Fit on training data ONLY** - Apply same transformations to val/test
3. **Save all artifacts** - Pipelines, scalers, encoders must be serializable
4. **Business interpretability** - Clusters should have meaningful business labels

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Numerical Scaling** | StandardScaler | Zero mean, unit variance - works well with most algorithms |
| **Categorical Encoding** | One-Hot (low cardinality) + Target Encoding (high cardinality) | Balance between sparsity and information |
| **Cyclical Time Encoding** | Sine/Cosine | Captures cyclical nature of hours/days/months |
| **Sentiment Analysis** | TextBlob | Simple, fast, no training needed |
| **Dimensionality Reduction** | PCA (optional) | Only if needed for clustering visualization |
| **Primary Clustering** | K-Means | Fast, interpretable, good baseline |
| **Alternative Clustering** | DBSCAN, GMM | For comparison and density-based segmentation |

---

## Input Data Summary (from Day 1)

### Processed Training Data

| Attribute | Value |
|-----------|-------|
| File | `data/processed/train_processed.parquet` |
| Rows | 79,986 |
| Columns | 44 |
| Target 1 | `is_satisfied` (72.5% satisfied) |
| Target 2 | `delivery_days` (mean 13.7 days) |

### Key Columns for Feature Engineering

| Category | Columns |
|----------|---------|
| **Identifiers** | `order_id`, `customer_id`, `customer_unique_id`, `product_id`, `seller_id` |
| **Timestamps** | `order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date` |
| **Monetary** | `price`, `freight_value`, `payment_value`, `payment_installments` |
| **Product** | `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`, `product_photos_qty` |
| **Categorical** | `customer_state`, `seller_state`, `product_category_name_english`, `payment_type`, `order_status` |
| **Text** | `review_comment_message` |
| **Targets** | `is_satisfied`, `delivery_days`, `is_late_delivery`, `delivery_delay_days` |

---

## Task Breakdown

### Task 1: Load Processed Data

**Objective**: Load the processed training data from Day 1

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load processed data
train_df = pd.read_parquet("data/processed/train_processed.parquet")
val_df = pd.read_parquet("data/processed/val_processed.parquet")
test_df = pd.read_parquet("data/processed/test_processed.parquet")

print(f"Train: {len(train_df):,} rows")
print(f"Val: {len(val_df):,} rows")
print(f"Test: {len(test_df):,} rows")
```

---

### Task 2: RFM Feature Engineering

**Objective**: Create Recency, Frequency, Monetary features per customer

RFM is a classic customer segmentation technique:
- **Recency**: How recently did the customer purchase?
- **Frequency**: How often do they purchase?
- **Monetary**: How much do they spend?

```python
def create_rfm_features(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Create RFM features aggregated at customer level.

    Args:
        df: DataFrame with order data
        reference_date: Date to calculate recency from (use training max date)

    Returns:
        DataFrame with RFM features per customer_unique_id
    """
    if reference_date is None:
        reference_date = df['order_purchase_timestamp'].max()

    # Aggregate by customer
    rfm = df.groupby('customer_unique_id').agg({
        # Recency: days since last order
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        # Frequency: number of orders
        'order_id': 'nunique',
        # Monetary: total spend
        'payment_value': 'sum',
    }).reset_index()

    rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']

    # Additional derived features
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
    rfm['monetary_per_day'] = rfm['monetary'] / (rfm['recency'] + 1)  # +1 to avoid division by zero

    return rfm
```

**RFM Features to Create**:

| Feature | Description | Formula |
|---------|-------------|---------|
| `recency` | Days since last order | `reference_date - max(order_date)` |
| `frequency` | Number of unique orders | `count(distinct order_id)` |
| `monetary` | Total spend | `sum(payment_value)` |
| `avg_order_value` | Average order value | `monetary / frequency` |
| `monetary_per_day` | Spend velocity | `monetary / recency` |

---

### Task 3: Temporal Feature Engineering

**Objective**: Extract time-based features from timestamps

```python
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from order timestamps."""
    df = df.copy()

    # Basic time extraction
    df['order_hour'] = df['order_purchase_timestamp'].dt.hour
    df['order_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['order_month'] = df['order_purchase_timestamp'].dt.month
    df['order_day'] = df['order_purchase_timestamp'].dt.day
    df['order_quarter'] = df['order_purchase_timestamp'].dt.quarter

    # Binary flags
    df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['order_day'] <= 7).astype(int)
    df['is_month_end'] = (df['order_day'] >= 24).astype(int)

    # Time of day categories
    df['time_of_day'] = pd.cut(
        df['order_hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )

    return df
```

**Cyclical Encoding** (for time features):

```python
def encode_cyclical(df: pd.DataFrame, column: str, max_val: int) -> pd.DataFrame:
    """
    Encode cyclical features using sine/cosine transformation.

    Example: hour 23 and hour 0 are close, but numerically far.
    Sine/cosine encoding captures this cyclical nature.
    """
    df = df.copy()
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_val)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_val)
    return df

# Apply cyclical encoding
df = encode_cyclical(df, 'order_hour', 24)
df = encode_cyclical(df, 'order_dayofweek', 7)
df = encode_cyclical(df, 'order_month', 12)
```

**Temporal Features to Create**:

| Feature | Type | Description |
|---------|------|-------------|
| `order_hour` | Numerical | Hour of purchase (0-23) |
| `order_dayofweek` | Numerical | Day of week (0=Mon, 6=Sun) |
| `order_month` | Numerical | Month (1-12) |
| `is_weekend` | Binary | 1 if Saturday/Sunday |
| `is_month_start` | Binary | 1 if day <= 7 |
| `is_month_end` | Binary | 1 if day >= 24 |
| `order_hour_sin/cos` | Cyclical | Sine/cosine of hour |
| `order_dayofweek_sin/cos` | Cyclical | Sine/cosine of day |
| `order_month_sin/cos` | Cyclical | Sine/cosine of month |

---

### Task 4: Geographic Feature Engineering

**Objective**: Create features based on customer/seller locations

```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

def create_geographic_features(df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create geographic features.

    Args:
        df: Main DataFrame
        geo_df: Geolocation DataFrame with zip_code, lat, lng
    """
    df = df.copy()

    # Get average lat/lng per zip code prefix (deduplicate geolocation data)
    geo_agg = geo_df.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()

    # Merge customer location
    df = df.merge(
        geo_agg,
        left_on='customer_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left'
    )
    df.rename(columns={
        'geolocation_lat': 'customer_lat',
        'geolocation_lng': 'customer_lng'
    }, inplace=True)

    # Merge seller location
    df = df.merge(
        geo_agg,
        left_on='seller_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left',
        suffixes=('', '_seller')
    )
    df.rename(columns={
        'geolocation_lat': 'seller_lat',
        'geolocation_lng': 'seller_lng'
    }, inplace=True)

    # Calculate distance
    df['seller_customer_distance_km'] = df.apply(
        lambda row: haversine_distance(
            row['customer_lat'], row['customer_lng'],
            row['seller_lat'], row['seller_lng']
        ) if pd.notna(row['customer_lat']) and pd.notna(row['seller_lat']) else np.nan,
        axis=1
    )

    # Same state flag
    df['is_same_state'] = (df['customer_state'] == df['seller_state']).astype(int)

    return df
```

**Geographic Features to Create**:

| Feature | Description |
|---------|-------------|
| `customer_lat`, `customer_lng` | Customer coordinates |
| `seller_lat`, `seller_lng` | Seller coordinates |
| `seller_customer_distance_km` | Distance between seller and customer |
| `is_same_state` | 1 if customer and seller in same state |
| `customer_region` | Region grouping (SE, NE, S, N, CO) |

**Brazilian Regions Mapping**:

```python
BRAZIL_REGIONS = {
    'SE': ['SP', 'RJ', 'MG', 'ES'],           # Southeast
    'S': ['PR', 'SC', 'RS'],                   # South
    'NE': ['BA', 'PE', 'CE', 'MA', 'PB', 'RN', 'AL', 'SE', 'PI'],  # Northeast
    'N': ['AM', 'PA', 'AC', 'RO', 'RR', 'AP', 'TO'],  # North
    'CO': ['GO', 'MT', 'MS', 'DF'],            # Central-West
}

def map_state_to_region(state):
    for region, states in BRAZIL_REGIONS.items():
        if state in states:
            return region
    return 'unknown'
```

---

### Task 5: Product Feature Engineering

**Objective**: Create features from product attributes

```python
def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create product-related features."""
    df = df.copy()

    # Product volume
    df['product_volume_cm3'] = (
        df['product_length_cm'] *
        df['product_height_cm'] *
        df['product_width_cm']
    )

    # Product density (weight per volume)
    df['product_density'] = df['product_weight_g'] / (df['product_volume_cm3'] + 1)

    # Price per weight
    df['price_per_kg'] = df['price'] / (df['product_weight_g'] / 1000 + 0.1)

    # Freight ratio
    df['freight_ratio'] = df['freight_value'] / (df['price'] + 1)

    # Category price statistics (fit on training data)
    category_stats = df.groupby('product_category_name_english')['price'].agg(['mean', 'std']).reset_index()
    category_stats.columns = ['product_category_name_english', 'category_price_mean', 'category_price_std']

    df = df.merge(category_stats, on='product_category_name_english', how='left')

    # Price relative to category
    df['price_vs_category'] = (df['price'] - df['category_price_mean']) / (df['category_price_std'] + 1)

    return df
```

**Product Features to Create**:

| Feature | Description |
|---------|-------------|
| `product_volume_cm3` | L × H × W |
| `product_density` | Weight / Volume |
| `price_per_kg` | Price normalized by weight |
| `freight_ratio` | Freight / Price |
| `category_price_mean` | Average price in category |
| `price_vs_category` | Z-score of price within category |

---

### Task 6: NLP Feature Engineering (Review Sentiment)

**Objective**: Extract sentiment from review comments

```python
from textblob import TextBlob

def create_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create NLP features from review comments."""
    df = df.copy()

    # Has review comment
    df['has_review_comment'] = df['review_comment_message'].notna().astype(int)

    # Text length features
    df['review_text_length'] = df['review_comment_message'].fillna('').str.len()
    df['review_word_count'] = df['review_comment_message'].fillna('').str.split().str.len()

    # Sentiment analysis (only for non-null comments)
    def get_sentiment(text):
        if pd.isna(text) or text.strip() == '':
            return 0.0, 0.0  # neutral sentiment for empty
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0.0, 0.0

    # Apply sentiment analysis
    sentiments = df['review_comment_message'].apply(get_sentiment)
    df['review_sentiment_polarity'] = sentiments.apply(lambda x: x[0])
    df['review_sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])

    # Sentiment category
    df['review_sentiment_label'] = pd.cut(
        df['review_sentiment_polarity'],
        bins=[-1.01, -0.1, 0.1, 1.01],
        labels=['negative', 'neutral', 'positive']
    )

    return df
```

**NLP Features to Create**:

| Feature | Description |
|---------|-------------|
| `has_review_comment` | Binary: 1 if comment exists |
| `review_text_length` | Character count |
| `review_word_count` | Word count |
| `review_sentiment_polarity` | -1 (negative) to +1 (positive) |
| `review_sentiment_subjectivity` | 0 (objective) to 1 (subjective) |
| `review_sentiment_label` | Categorical: negative/neutral/positive |

---

### Task 7: Payment Feature Engineering

**Objective**: Create features from payment information

```python
def create_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create payment-related features."""
    df = df.copy()

    # Installment ratio (how much of total per installment)
    df['payment_per_installment'] = df['payment_value'] / (df['payment_installments'] + 1)

    # Is full payment (no installments)
    df['is_full_payment'] = (df['payment_installments'] <= 1).astype(int)

    # High installment flag
    df['is_high_installment'] = (df['payment_installments'] >= 6).astype(int)

    return df
```

---

### Task 8: Build sklearn Pipeline

**Objective**: Create a reproducible feature transformation pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# Define column groups
NUMERICAL_FEATURES = [
    'price', 'freight_value', 'payment_value', 'payment_installments',
    'product_weight_g', 'product_volume_cm3', 'product_photos_qty',
    'seller_customer_distance_km', 'freight_ratio', 'price_vs_category',
    'review_text_length', 'review_word_count',
    'review_sentiment_polarity', 'review_sentiment_subjectivity',
    'order_hour_sin', 'order_hour_cos',
    'order_dayofweek_sin', 'order_dayofweek_cos',
]

CATEGORICAL_FEATURES = [
    'customer_state', 'seller_state',
    'product_category_name_english', 'payment_type',
    'customer_region', 'time_of_day', 'review_sentiment_label',
]

BINARY_FEATURES = [
    'is_weekend', 'is_month_start', 'is_month_end',
    'is_same_state', 'is_full_payment', 'is_high_installment',
    'has_review_comment', 'is_late_delivery',
]

# Build preprocessing pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, NUMERICAL_FEATURES),
    ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ('bin', 'passthrough', BINARY_FEATURES),
])
```

---

### Task 9: Feature Selection

**Objective**: Remove redundant or low-value features

```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

def remove_low_variance_features(X, threshold=0.01):
    """Remove features with near-zero variance."""
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    return X_selected, selector

def remove_highly_correlated_features(df, threshold=0.95):
    """Remove one of each pair of highly correlated features."""
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_triangle.columns
               if any(upper_triangle[col] > threshold)]
    return df.drop(columns=to_drop), to_drop
```

---

### Task 10: Prepare Clustering Features

**Objective**: Select and scale features for customer segmentation

```python
# Features for clustering (RFM + behavioral)
CLUSTERING_FEATURES = [
    'recency',
    'frequency',
    'monetary',
    'avg_order_value',
    'avg_review_score',  # Customer's average review score
    'avg_delivery_days', # Customer's average delivery experience
    'late_delivery_rate', # % of late deliveries for customer
]

def prepare_clustering_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data at customer level for clustering.
    """
    # Reference date for recency
    reference_date = df['order_purchase_timestamp'].max()

    customer_features = df.groupby('customer_unique_id').agg({
        # RFM
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum',

        # Behavioral
        'review_score': 'mean',
        'delivery_days': 'mean',
        'is_late_delivery': 'mean',

        # Product preferences
        'price': 'mean',
        'product_category_name_english': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown',
    }).reset_index()

    customer_features.columns = [
        'customer_unique_id', 'recency', 'frequency', 'monetary',
        'avg_review_score', 'avg_delivery_days', 'late_delivery_rate',
        'avg_price', 'favorite_category'
    ]

    # Derived features
    customer_features['avg_order_value'] = (
        customer_features['monetary'] / customer_features['frequency']
    )

    return customer_features
```

---

### Task 11: K-Means Clustering

**Objective**: Segment customers using K-Means algorithm

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

def find_optimal_k(X, k_range=range(2, 11)):
    """
    Find optimal number of clusters using elbow method and silhouette score.
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))

        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.3f}")

    return inertias, silhouettes

# Scale features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_features[CLUSTERING_FEATURES])

# Find optimal K
inertias, silhouettes = find_optimal_k(X_scaled)

# Train final model with chosen K
OPTIMAL_K = 4  # Determined from elbow/silhouette analysis
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
customer_features['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
```

**Visualization**:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(range(2, 11), inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

# Silhouette plot
axes[1].plot(range(2, 11), silhouettes, 'ro-')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')

plt.tight_layout()
plt.show()
```

---

### Task 12: DBSCAN Clustering

**Objective**: Density-based clustering for comparison

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def find_optimal_eps(X, k=5):
    """
    Find optimal eps using k-distance graph.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)

    # Sort distances to kth nearest neighbor
    distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title('K-Distance Graph (look for elbow)')
    plt.show()

    return distances

# Find optimal eps
distances = find_optimal_eps(X_scaled, k=5)

# Train DBSCAN
EPS = 0.5  # Determined from k-distance graph
MIN_SAMPLES = 10  # Minimum points to form a cluster

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
customer_features['cluster_dbscan'] = dbscan.fit_predict(X_scaled)

# Check results
print(f"Clusters found: {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}")
print(f"Noise points: {(dbscan.labels_ == -1).sum()}")
```

---

### Task 13: Hierarchical Clustering

**Objective**: Agglomerative clustering with dendrogram visualization

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create dendrogram (use subset for visualization)
sample_size = min(1000, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]

# Linkage matrix
linkage_matrix = linkage(X_sample, method='ward')

# Plot dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Train agglomerative clustering
N_CLUSTERS = 4
agg_clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
customer_features['cluster_hierarchical'] = agg_clustering.fit_predict(X_scaled)
```

---

### Task 14: Gaussian Mixture Models

**Objective**: Soft clustering with probability assignments

```python
from sklearn.mixture import GaussianMixture

def find_optimal_gmm_components(X, n_range=range(2, 11)):
    """
    Find optimal number of components using BIC/AIC.
    """
    bics = []
    aics = []

    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)

        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

        print(f"N={n}: BIC={bics[-1]:.2f}, AIC={aics[-1]:.2f}")

    return bics, aics

# Find optimal components
bics, aics = find_optimal_gmm_components(X_scaled)

# Train GMM
N_COMPONENTS = 4  # Determined from BIC/AIC
gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42)
customer_features['cluster_gmm'] = gmm.fit_predict(X_scaled)
customer_features['cluster_gmm_proba'] = gmm.predict_proba(X_scaled).max(axis=1)
```

---

### Task 15: Clustering Model Comparison

**Objective**: Compare all clustering algorithms and select the best

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X, labels, name):
    """Evaluate clustering quality with multiple metrics."""

    # Filter out noise points for DBSCAN
    mask = labels != -1
    if mask.sum() < len(labels):
        X_eval = X[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X
        labels_eval = labels

    n_clusters = len(set(labels_eval))

    if n_clusters < 2:
        return {'name': name, 'n_clusters': n_clusters, 'error': 'Less than 2 clusters'}

    return {
        'name': name,
        'n_clusters': n_clusters,
        'silhouette': silhouette_score(X_eval, labels_eval),
        'davies_bouldin': davies_bouldin_score(X_eval, labels_eval),
        'calinski_harabasz': calinski_harabasz_score(X_eval, labels_eval),
    }

# Compare all models
results = []
results.append(evaluate_clustering(X_scaled, customer_features['cluster_kmeans'], 'K-Means'))
results.append(evaluate_clustering(X_scaled, customer_features['cluster_dbscan'], 'DBSCAN'))
results.append(evaluate_clustering(X_scaled, customer_features['cluster_hierarchical'], 'Hierarchical'))
results.append(evaluate_clustering(X_scaled, customer_features['cluster_gmm'], 'GMM'))

comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))
```

**Metrics Interpretation**:

| Metric | Range | Best |
|--------|-------|------|
| Silhouette Score | -1 to 1 | Higher is better (>0.5 good) |
| Davies-Bouldin Index | 0 to ∞ | Lower is better |
| Calinski-Harabasz Index | 0 to ∞ | Higher is better |

---

### Task 16: Cluster Profiling & Business Interpretation

**Objective**: Understand and label each cluster

```python
def profile_clusters(df, cluster_col, features):
    """
    Create cluster profiles showing mean values per cluster.
    """
    profiles = df.groupby(cluster_col)[features].agg(['mean', 'median', 'std'])
    return profiles

# Profile clusters
PROFILE_FEATURES = ['recency', 'frequency', 'monetary', 'avg_order_value',
                    'avg_review_score', 'avg_delivery_days', 'late_delivery_rate']

profiles = profile_clusters(customer_features, 'cluster_kmeans', PROFILE_FEATURES)
print(profiles)
```

**Example Business Labels**:

| Cluster | RFM Profile | Suggested Label |
|---------|-------------|-----------------|
| 0 | Low R, High F, High M | "Loyal Champions" |
| 1 | High R, Low F, Low M | "At-Risk / Churned" |
| 2 | Medium R, Medium F, Medium M | "Potential Loyalists" |
| 3 | Low R, Low F, High M | "Big Spenders (New)" |

```python
# Assign business labels
CLUSTER_LABELS = {
    0: 'loyal_champions',
    1: 'at_risk',
    2: 'potential_loyalists',
    3: 'big_spenders',
}

customer_features['customer_segment'] = customer_features['cluster_kmeans'].map(CLUSTER_LABELS)
```

---

### Task 17: Add Cluster Labels to Main Dataset

**Objective**: Merge cluster assignments back to order-level data

```python
# Merge cluster labels back to main dataset
train_df = train_df.merge(
    customer_features[['customer_unique_id', 'cluster_kmeans', 'customer_segment']],
    on='customer_unique_id',
    how='left'
)

val_df = val_df.merge(
    customer_features[['customer_unique_id', 'cluster_kmeans', 'customer_segment']],
    on='customer_unique_id',
    how='left'
)

test_df = test_df.merge(
    customer_features[['customer_unique_id', 'cluster_kmeans', 'customer_segment']],
    on='customer_unique_id',
    how='left'
)

# Fill new customers (not in training) with 'unknown'
for df in [train_df, val_df, test_df]:
    df['cluster_kmeans'].fillna(-1, inplace=True)
    df['customer_segment'].fillna('unknown', inplace=True)
```

---

### Task 18: Save All Artifacts

**Objective**: Persist all models and processed data

```python
import joblib
import json

# Save clustering artifacts
joblib.dump(scaler, 'models/cluster_scaler.joblib')
joblib.dump(kmeans, 'models/clustering_model.joblib')

# Save cluster profiles
profiles.to_csv('models/cluster_profiles.csv')

# Save feature pipeline
joblib.dump(preprocessor, 'models/feature_pipeline.joblib')

# Save feature names
feature_names = {
    'numerical': NUMERICAL_FEATURES,
    'categorical': CATEGORICAL_FEATURES,
    'binary': BINARY_FEATURES,
    'clustering': CLUSTERING_FEATURES,
}
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)

# Save processed data with new features
train_df.to_parquet('data/processed/train_featured.parquet', index=False)
val_df.to_parquet('data/processed/val_featured.parquet', index=False)
test_df.to_parquet('data/processed/test_featured.parquet', index=False)

# Save customer-level clustering data
customer_features.to_parquet('data/processed/customer_segments.parquet', index=False)

print("All artifacts saved!")
```

---

### Task 19: Run Clustering on Vertex AI (Optional Cloud Execution)

**Objective**: Run the clustering pipeline on Google Cloud's Vertex AI instead of locally

For large datasets or production workloads, run clustering on Vertex AI Custom Training:

**Prerequisites**:
- GCP project configured (`configs/config.yaml`)
- Service account with Vertex AI and GCS permissions
- Customer segments data generated locally (`data/processed/customer_segments.parquet`)

**Files Created for Vertex AI**:
```
src/
├── gcp_utils.py                   # GCS upload/download + Vertex AI job submission
└── vertex_training.py             # Standalone training script for Vertex AI
```

**Usage**:

```python
from src.gcp_utils import load_config, run_clustering_on_vertex_ai, download_clustering_artifacts

# Load configuration
config = load_config()

# Run clustering on Vertex AI
# This will: upload data to GCS, submit training job, wait for completion
job = run_clustering_on_vertex_ai(config, optimal_k=4, sync=True)

# Download trained artifacts from GCS
artifacts = download_clustering_artifacts(config, local_dir="models")
```

**Vertex AI Configuration** (from `configs/config.yaml`):

| Setting | Value |
|---------|-------|
| Machine Type | `n1-standard-4` (4 vCPUs, 15GB RAM) |
| Container | `us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-3:latest` |
| Output | `gs://olist-ml-ismail/training-output` |

**What Happens on Vertex AI**:
1. Data loaded from GCS (`gs://bucket/data/customer_segments.parquet`)
2. K-Means optimal K search (silhouette method)
3. All 4 clustering algorithms run (K-Means, DBSCAN, Hierarchical, GMM)
4. Models and artifacts saved to GCS (`gs://bucket/models/`)

**Artifacts Saved to GCS**:
- `cluster_scaler.joblib` - StandardScaler for clustering features
- `kmeans_model.joblib` - Final K-Means model
- `cluster_profiles.json` - Cluster statistics and profiles
- `clustering_results.json` - Algorithm comparison metrics
- `customer_segments_clustered.parquet` - Data with cluster labels

---

## Day 2 Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Load processed data from Day 1 | ✅ |
| 2 | Create RFM features (Recency, Frequency, Monetary) | ✅ |
| 3 | Create temporal features (hour, day, month, cyclical) | ✅ |
| 4 | Create geographic features (distance, region) | ✅ |
| 5 | Create product features (volume, density, price ratios) | ✅ |
| 6 | Create NLP features (sentiment analysis) | ✅ |
| 7 | Create payment features | ✅ |
| 8 | Build sklearn preprocessing pipeline | ✅ |
| 9 | Feature selection (variance, correlation) | ✅ |
| 10 | Prepare customer-level clustering data | ✅ |
| 11 | Implement K-Means clustering | ✅ |
| 12 | Implement DBSCAN clustering | ✅ |
| 13 | Implement Hierarchical clustering | ✅ |
| 14 | Implement Gaussian Mixture Models | ✅ |
| 15 | Compare clustering algorithms | ✅ |
| 16 | Profile clusters and assign business labels | ✅ |
| 17 | Add cluster labels to main datasets | ✅ |
| 18 | Save all artifacts (pipelines, models, data) | ✅ |
| 19 | Run clustering on Vertex AI (optional cloud execution) | ✅ |
| 20 | Create notebook 02_feature_engineering.ipynb | ✅ |
| 21 | Commit code to git | ✅ |

---

## Expected Outputs

### Files Created

```
models/
├── feature_pipeline.joblib        # sklearn ColumnTransformer
├── cluster_scaler.joblib          # StandardScaler for clustering
├── clustering_model.joblib        # Final K-Means model
├── cluster_profiles.csv           # Cluster statistics
└── feature_names.json             # Feature lists

data/processed/
├── train_featured.parquet         # Train with all new features
├── val_featured.parquet           # Val with all new features
├── test_featured.parquet          # Test with all new features
└── customer_segments.parquet      # Customer-level clustering data

notebooks/
└── 02_feature_engineering.ipynb   # Complete notebook
```

### New Features Summary

| Category | Count | Examples |
|----------|-------|----------|
| RFM | 5 | recency, frequency, monetary, avg_order_value |
| Temporal | 12 | order_hour, is_weekend, hour_sin, hour_cos |
| Geographic | 6 | distance_km, is_same_state, customer_region |
| Product | 6 | volume, density, price_vs_category, freight_ratio |
| NLP | 5 | sentiment_polarity, word_count, has_comment |
| Payment | 3 | is_full_payment, payment_per_installment |
| Cluster | 2 | cluster_kmeans, customer_segment |
| **Total** | **~40** | |

---

## Decision Log

| Decision | Options Considered | Chosen | Rationale |
|----------|-------------------|--------|-----------|
| Numerical Scaling | Standard / MinMax / Robust | StandardScaler | Works well with clustering and most ML algorithms |
| Categorical Encoding | One-Hot / Target / Binary | One-Hot (low card) | Interpretable, no leakage risk |
| Time Encoding | Raw / One-Hot / Sine-Cosine | Sine-Cosine | Captures cyclical patterns |
| Sentiment Tool | VADER / TextBlob / Transformer | TextBlob | Simple, fast, adequate for Portuguese |
| Primary Clustering | K-Means / DBSCAN / Hierarchical | K-Means | Fast, interpretable, good baseline |
| Optimal K Method | Elbow / Silhouette / Gap | Silhouette + Elbow | Both for confirmation |

---

## Next Steps

After completing Day 2, proceed to **Day 3: Supervised Learning** where you will:
1. Establish baseline models (DummyClassifier, DummyRegressor)
2. Train classification models for customer satisfaction
3. Train regression models for delivery time prediction
4. Hyperparameter tuning with cross-validation
5. Model comparison and selection
6. SHAP interpretability analysis
7. Final evaluation on test set
