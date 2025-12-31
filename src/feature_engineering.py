"""
Feature Engineering utilities for Olist E-Commerce dataset.

This module handles:
1. RFM Features (customer-level aggregations)
2. Temporal Features (time extraction + cyclical encoding)
3. Geographic Features (distance calculation + region mapping)
4. Product Features (volume, density, price ratios)
5. NLP Features (text statistics + sentiment analysis)
6. Payment Features (installment analysis)
7. sklearn Pipeline for preprocessing

Day 2, Part 1 Implementation
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from math import radians, sin, cos, sqrt, atan2

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

import joblib

# Try to import TextBlob for sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Sentiment features will be neutral.")


# =============================================================================
# CONSTANTS
# =============================================================================

# Brazilian region mapping
BRAZIL_REGIONS = {
    'SP': 'SE', 'RJ': 'SE', 'MG': 'SE', 'ES': 'SE',  # Southeast
    'PR': 'S', 'SC': 'S', 'RS': 'S',                   # South
    'BA': 'NE', 'PE': 'NE', 'CE': 'NE', 'MA': 'NE',   # Northeast
    'PB': 'NE', 'RN': 'NE', 'AL': 'NE', 'SE': 'NE', 'PI': 'NE',
    'AM': 'N', 'PA': 'N', 'AC': 'N', 'RO': 'N',        # North
    'RR': 'N', 'AP': 'N', 'TO': 'N',
    'GO': 'CO', 'MT': 'CO', 'MS': 'CO', 'DF': 'CO',   # Central-West
}

# Feature column definitions
NUMERICAL_FEATURES = [
    'price', 'freight_value', 'payment_value', 'payment_installments',
    'product_weight_g', 'product_volume_cm3', 'product_photos_qty',
    'seller_customer_distance_km', 'freight_ratio', 'price_vs_category_zscore',
    'review_text_length', 'review_word_count', 'review_caps_ratio',
    'review_sentiment_polarity', 'review_sentiment_subjectivity',
    'order_hour_sin', 'order_hour_cos',
    'order_dayofweek_sin', 'order_dayofweek_cos',
    'payment_per_installment',
]

CATEGORICAL_FEATURES = [
    'customer_state', 'seller_state', 'customer_region', 'seller_region',
    'product_category_name_english', 'payment_type',
]

BINARY_FEATURES = [
    'is_weekend', 'is_month_start', 'is_month_end',
    'is_same_state', 'is_full_payment', 'is_high_installment',
    'has_review_comment', 'is_late_delivery',
]

# RFM features for clustering
RFM_FEATURES = [
    'recency', 'frequency', 'monetary', 'avg_order_value', 'monetary_per_day',
]

# Customer behavioral features for clustering
CUSTOMER_BEHAVIORAL_FEATURES = [
    'avg_review_score', 'avg_delivery_days', 'late_delivery_rate',
    'avg_price', 'total_items',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def state_to_region(state: str) -> str:
    """Map Brazilian state code to region."""
    if pd.isna(state):
        return 'unknown'
    return BRAZIL_REGIONS.get(str(state).upper(), 'unknown')


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Coordinates of point 1
        lat2, lon2: Coordinates of point 2

    Returns:
        Distance in kilometers
    """
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan

    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def haversine_vectorized(lat1: pd.Series, lon1: pd.Series,
                         lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Vectorized haversine distance calculation."""
    R = 6371  # Earth's radius in km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def get_sentiment(text: str) -> Tuple[float, float]:
    """
    Get sentiment polarity and subjectivity from text.

    Args:
        text: Input text

    Returns:
        Tuple of (polarity, subjectivity)
    """
    if not TEXTBLOB_AVAILABLE:
        return 0.0, 0.0

    if pd.isna(text) or str(text).strip() == '':
        return 0.0, 0.0

    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return 0.0, 0.0


def caps_ratio(text: str) -> float:
    """Calculate ratio of uppercase letters in text."""
    if pd.isna(text) or len(str(text)) == 0:
        return 0.0

    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0

    return sum(c.isupper() for c in letters) / len(letters)


# =============================================================================
# TASK 2: RFM FEATURES (Customer-Level)
# =============================================================================

def create_rfm_features(df: pd.DataFrame,
                        reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Create RFM (Recency, Frequency, Monetary) features aggregated at customer level.

    Args:
        df: DataFrame with order data
        reference_date: Date to calculate recency from (use training max date)

    Returns:
        DataFrame with RFM features per customer_unique_id
    """
    if reference_date is None:
        reference_date = df['order_purchase_timestamp'].max()

    print(f"📊 Creating RFM features (reference date: {reference_date.date()})")

    # Aggregate by customer
    rfm = df.groupby('customer_unique_id').agg({
        # Recency: days since last order
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        # Frequency: number of unique orders
        'order_id': 'nunique',
        # Monetary: total spend
        'payment_value': 'sum',
    }).reset_index()

    rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']

    # Derived features
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
    rfm['monetary_per_day'] = rfm['monetary'] / (rfm['recency'] + 1)  # +1 to avoid div by zero

    print(f"  ✓ Created RFM features for {len(rfm):,} customers")
    print(f"    Recency: mean={rfm['recency'].mean():.1f} days")
    print(f"    Frequency: mean={rfm['frequency'].mean():.2f} orders")
    print(f"    Monetary: mean=${rfm['monetary'].mean():.2f}")

    return rfm


def create_customer_behavioral_features(df: pd.DataFrame,
                                        reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Create additional behavioral features aggregated at customer level.
    Used for clustering along with RFM.

    Args:
        df: DataFrame with order data
        reference_date: Reference date for recency calculation

    Returns:
        DataFrame with customer-level behavioral features
    """
    if reference_date is None:
        reference_date = df['order_purchase_timestamp'].max()

    print("📊 Creating customer behavioral features")

    customer_features = df.groupby('customer_unique_id').agg({
        # RFM base
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum',

        # Behavioral
        'review_score': 'mean',
        'delivery_days': 'mean',
        'is_late_delivery': 'mean',

        # Product preferences
        'price': 'mean',
        'order_item_id': 'count',  # total items purchased

        # Favorite category
        'product_category_name_english': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
    }).reset_index()

    customer_features.columns = [
        'customer_unique_id', 'recency', 'frequency', 'monetary',
        'avg_review_score', 'avg_delivery_days', 'late_delivery_rate',
        'avg_price', 'total_items', 'favorite_category'
    ]

    # Derived RFM features
    customer_features['avg_order_value'] = (
        customer_features['monetary'] / customer_features['frequency']
    )
    customer_features['monetary_per_day'] = (
        customer_features['monetary'] / (customer_features['recency'] + 1)
    )

    # Fill NaN review scores with median
    median_review = customer_features['avg_review_score'].median()
    customer_features['avg_review_score'].fillna(median_review, inplace=True)

    # Fill NaN delivery days with median
    median_delivery = customer_features['avg_delivery_days'].median()
    customer_features['avg_delivery_days'].fillna(median_delivery, inplace=True)

    print(f"  ✓ Created behavioral features for {len(customer_features):,} customers")

    return customer_features


# =============================================================================
# TASK 3: TEMPORAL FEATURES (Order-Level)
# =============================================================================

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from order timestamps.

    Features created:
    - Raw time components: hour, dayofweek, month, day, quarter
    - Binary flags: is_weekend, is_month_start, is_month_end
    - Cyclical encoding: sin/cos for hour and dayofweek

    Args:
        df: DataFrame with order_purchase_timestamp column

    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    ts = df['order_purchase_timestamp']

    print("📊 Creating temporal features")

    # Raw time extraction
    df['order_hour'] = ts.dt.hour
    df['order_dayofweek'] = ts.dt.dayofweek
    df['order_month'] = ts.dt.month
    df['order_day'] = ts.dt.day
    df['order_quarter'] = ts.dt.quarter

    # Binary flags
    df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['order_day'] <= 7).astype(int)
    df['is_month_end'] = (df['order_day'] >= 24).astype(int)

    # Cyclical encoding for hour (0-23)
    df['order_hour_sin'] = np.sin(2 * np.pi * df['order_hour'] / 24)
    df['order_hour_cos'] = np.cos(2 * np.pi * df['order_hour'] / 24)

    # Cyclical encoding for day of week (0-6)
    df['order_dayofweek_sin'] = np.sin(2 * np.pi * df['order_dayofweek'] / 7)
    df['order_dayofweek_cos'] = np.cos(2 * np.pi * df['order_dayofweek'] / 7)

    print(f"  ✓ Created 12 temporal features")
    print(f"    Weekend orders: {df['is_weekend'].mean():.1%}")
    print(f"    Peak hour: {df['order_hour'].mode().iloc[0]}")

    return df


# =============================================================================
# TASK 4: GEOGRAPHIC FEATURES (Order-Level)
# =============================================================================

def load_geolocation_data(geo_path: str = "data/raw/olist_geolocation_dataset.csv") -> pd.DataFrame:
    """Load and aggregate geolocation data."""
    geo = pd.read_csv(geo_path)

    # Aggregate to get mean lat/lng per zip code prefix (removes duplicates)
    geo_agg = geo.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()

    print(f"📍 Loaded geolocation data: {len(geo_agg):,} unique zip codes")

    return geo_agg


def create_geographic_features(df: pd.DataFrame,
                               geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create geographic features based on customer/seller locations.

    Features created:
    - seller_customer_distance_km: Haversine distance
    - is_same_state: Binary flag
    - customer_region, seller_region: Brazilian region codes

    Args:
        df: DataFrame with customer/seller zip codes and states
        geo_df: Aggregated geolocation DataFrame

    Returns:
        DataFrame with geographic features added
    """
    df = df.copy()

    print("📊 Creating geographic features")

    # Merge customer coordinates
    df = df.merge(
        geo_df,
        left_on='customer_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left'
    )
    df.rename(columns={
        'geolocation_lat': 'customer_lat',
        'geolocation_lng': 'customer_lng'
    }, inplace=True)
    df.drop(columns=['geolocation_zip_code_prefix'], inplace=True, errors='ignore')

    # Merge seller coordinates
    df = df.merge(
        geo_df,
        left_on='seller_zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left',
        suffixes=('', '_seller')
    )
    df.rename(columns={
        'geolocation_lat': 'seller_lat',
        'geolocation_lng': 'seller_lng'
    }, inplace=True)
    df.drop(columns=['geolocation_zip_code_prefix'], inplace=True, errors='ignore')

    # Calculate haversine distance (vectorized)
    df['seller_customer_distance_km'] = haversine_vectorized(
        df['customer_lat'], df['customer_lng'],
        df['seller_lat'], df['seller_lng']
    )

    # Fill missing distances with median
    median_distance = df['seller_customer_distance_km'].median()
    df['seller_customer_distance_km'].fillna(median_distance, inplace=True)

    # Same state flag
    df['is_same_state'] = (
        df['customer_state'].astype(str) == df['seller_state'].astype(str)
    ).astype(int)

    # Region mapping
    df['customer_region'] = df['customer_state'].apply(state_to_region)
    df['seller_region'] = df['seller_state'].apply(state_to_region)

    # Stats
    print(f"  ✓ Created 6 geographic features")
    print(f"    Avg distance: {df['seller_customer_distance_km'].mean():.1f} km")
    print(f"    Same state: {df['is_same_state'].mean():.1%}")

    return df


# =============================================================================
# TASK 5: PRODUCT FEATURES (Order-Level)
# =============================================================================

def create_product_features(df: pd.DataFrame,
                            category_stats: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create product-related features.

    Features created:
    - product_volume_cm3: L × H × W
    - product_density: weight / volume
    - price_per_kg: price / (weight in kg)
    - freight_ratio: freight / price
    - price_vs_category_mean: price - category mean
    - price_vs_category_zscore: standardized price within category

    Args:
        df: DataFrame with product columns
        category_stats: Pre-computed category statistics (for val/test)

    Returns:
        Tuple of (DataFrame with features, category_stats DataFrame)
    """
    df = df.copy()

    print("📊 Creating product features")

    # Product volume
    df['product_volume_cm3'] = (
        df['product_length_cm'] *
        df['product_height_cm'] *
        df['product_width_cm']
    )

    # Product density (weight per volume)
    df['product_density'] = df['product_weight_g'] / (df['product_volume_cm3'] + 1)

    # Price per kg
    df['price_per_kg'] = df['price'] / (df['product_weight_g'] / 1000 + 0.1)

    # Freight ratio
    df['freight_ratio'] = df['freight_value'] / (df['price'] + 1)

    # Category price statistics
    if category_stats is None:
        # Compute from this data (training set)
        category_stats = df.groupby('product_category_name_english')['price'].agg(
            ['mean', 'std']
        ).reset_index()
        category_stats.columns = ['product_category_name_english',
                                  'category_price_mean', 'category_price_std']
        # Fill NaN std with 1 (for categories with single product)
        category_stats['category_price_std'].fillna(1.0, inplace=True)

    # Merge category stats
    df = df.merge(category_stats, on='product_category_name_english', how='left')

    # Fill missing category stats with global stats
    global_mean = df['price'].mean()
    global_std = df['price'].std()
    df['category_price_mean'].fillna(global_mean, inplace=True)
    df['category_price_std'].fillna(global_std, inplace=True)

    # Relative pricing
    df['price_vs_category_mean'] = df['price'] - df['category_price_mean']
    df['price_vs_category_zscore'] = (
        (df['price'] - df['category_price_mean']) /
        (df['category_price_std'] + 1)
    )

    print(f"  ✓ Created 6 product features")
    print(f"    Avg volume: {df['product_volume_cm3'].mean():,.0f} cm³")
    print(f"    Avg freight ratio: {df['freight_ratio'].mean():.2%}")

    return df, category_stats


# =============================================================================
# TASK 6: NLP FEATURES (Order-Level)
# =============================================================================

def create_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create NLP features from review comments.

    Features created:
    - has_review_comment: Binary flag
    - review_text_length: Character count
    - review_word_count: Word count
    - review_exclamation_count: Count of "!"
    - review_question_count: Count of "?"
    - review_caps_ratio: Ratio of uppercase letters
    - review_sentiment_polarity: TextBlob polarity (-1 to 1)
    - review_sentiment_subjectivity: TextBlob subjectivity (0 to 1)

    Args:
        df: DataFrame with review_comment_message column

    Returns:
        DataFrame with NLP features added
    """
    df = df.copy()

    print("📊 Creating NLP features")

    # Fill NaN with empty string for text operations
    text = df['review_comment_message'].fillna('')

    # Binary flag
    df['has_review_comment'] = (text.str.strip() != '').astype(int)

    # Text statistics (language-agnostic)
    df['review_text_length'] = text.str.len()
    df['review_word_count'] = text.str.split().str.len().fillna(0).astype(int)
    df['review_exclamation_count'] = text.str.count('!')
    df['review_question_count'] = text.str.count(r'\?')

    # Caps ratio (shouting indicator)
    df['review_caps_ratio'] = text.apply(caps_ratio)

    # Sentiment analysis (TextBlob - works as weak signal for Portuguese)
    print("    Computing sentiment (this may take a moment)...")
    sentiments = text.apply(get_sentiment)
    df['review_sentiment_polarity'] = sentiments.apply(lambda x: x[0])
    df['review_sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])

    print(f"  ✓ Created 8 NLP features")
    print(f"    Reviews with comments: {df['has_review_comment'].mean():.1%}")
    print(f"    Avg word count: {df['review_word_count'].mean():.1f}")

    return df


# =============================================================================
# TASK 7: PAYMENT FEATURES (Order-Level)
# =============================================================================

def create_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create payment-related features.

    Features created:
    - payment_per_installment: Payment value / installments
    - is_full_payment: 1 if installments <= 1
    - is_high_installment: 1 if installments >= 6

    Args:
        df: DataFrame with payment columns

    Returns:
        DataFrame with payment features added
    """
    df = df.copy()

    print("📊 Creating payment features")

    # Payment per installment
    df['payment_per_installment'] = (
        df['payment_value'] / (df['payment_installments'].fillna(1) + 1)
    )

    # Binary flags
    df['is_full_payment'] = (df['payment_installments'].fillna(1) <= 1).astype(int)
    df['is_high_installment'] = (df['payment_installments'].fillna(1) >= 6).astype(int)

    print(f"  ✓ Created 3 payment features")
    print(f"    Full payment: {df['is_full_payment'].mean():.1%}")
    print(f"    High installment: {df['is_high_installment'].mean():.1%}")

    return df


# =============================================================================
# TASK 8: SKLEARN PIPELINE
# =============================================================================

def create_preprocessing_pipeline(
    numerical_features: List[str] = None,
    categorical_features: List[str] = None,
    binary_features: List[str] = None
) -> ColumnTransformer:
    """
    Create sklearn ColumnTransformer for feature preprocessing.

    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        binary_features: List of binary column names (passthrough)

    Returns:
        Fitted ColumnTransformer
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if binary_features is None:
        binary_features = BINARY_FEATURES

    print("📊 Creating preprocessing pipeline")

    # Numerical pipeline: impute + scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # Categorical pipeline: impute + one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # Combined preprocessor
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features),
        ('bin', 'passthrough', binary_features),
    ], remainder='drop')

    print(f"  ✓ Pipeline created")
    print(f"    Numerical features: {len(numerical_features)}")
    print(f"    Categorical features: {len(categorical_features)}")
    print(f"    Binary features: {len(binary_features)}")

    return preprocessor


def get_feature_names_from_pipeline(preprocessor: ColumnTransformer,
                                     numerical_features: List[str],
                                     categorical_features: List[str],
                                     binary_features: List[str]) -> List[str]:
    """Extract feature names after pipeline transformation."""
    feature_names = []

    # Numerical features (same names after scaling)
    feature_names.extend(numerical_features)

    # Categorical features (one-hot encoded names)
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names.tolist())
    except Exception:
        # Fallback: just use original names
        feature_names.extend(categorical_features)

    # Binary features (passthrough)
    feature_names.extend(binary_features)

    return feature_names


# =============================================================================
# TASK 9: FEATURE SELECTION
# =============================================================================

def remove_low_variance_features(X: np.ndarray,
                                  threshold: float = 0.01) -> Tuple[np.ndarray, VarianceThreshold]:
    """
    Remove features with near-zero variance.

    Args:
        X: Feature matrix
        threshold: Variance threshold (default 0.01)

    Returns:
        Tuple of (filtered X, fitted selector)
    """
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)

    n_removed = X.shape[1] - X_selected.shape[1]
    print(f"  Removed {n_removed} low-variance features (threshold={threshold})")

    return X_selected, selector


def remove_highly_correlated_features(df: pd.DataFrame,
                                       threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove one of each pair of highly correlated features.

    Args:
        df: DataFrame with numerical features
        threshold: Correlation threshold (default 0.95)

    Returns:
        Tuple of (filtered DataFrame, list of dropped columns)
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Get upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find columns to drop
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"  Removed {len(to_drop)} highly correlated features (threshold={threshold})")
    if to_drop:
        print(f"    Dropped: {to_drop}")

    return df.drop(columns=to_drop), to_drop


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def engineer_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> Dict:
    """
    Run the complete feature engineering pipeline.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        geo_df: Geolocation DataFrame
        reference_date: Reference date for RFM (default: train max date)

    Returns:
        Dictionary with all DataFrames and artifacts
    """
    if reference_date is None:
        reference_date = train_df['order_purchase_timestamp'].max()

    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print(f"Reference date: {reference_date}")
    print(f"Train: {len(train_df):,} rows")
    print(f"Val: {len(val_df):,} rows")
    print(f"Test: {len(test_df):,} rows")
    print("=" * 60)

    # =========================================================================
    # STEP 1: ORDER-LEVEL FEATURES
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 1: ORDER-LEVEL FEATURES")
    print("=" * 60)

    # Process each split
    category_stats = None
    processed = {}

    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\n--- Processing {name} set ---")

        # Temporal features
        df = create_temporal_features(df)

        # Geographic features
        df = create_geographic_features(df, geo_df)

        # Product features (use train category stats for val/test)
        df, cat_stats = create_product_features(df, category_stats)
        if name == 'train':
            category_stats = cat_stats

        # NLP features
        df = create_nlp_features(df)

        # Payment features
        df = create_payment_features(df)

        processed[name] = df

    # =========================================================================
    # STEP 2: CUSTOMER-LEVEL FEATURES (for clustering)
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 2: CUSTOMER-LEVEL FEATURES")
    print("=" * 60)

    # Create customer features from training data only
    customer_features = create_customer_behavioral_features(
        processed['train'], reference_date
    )

    # =========================================================================
    # STEP 3: MERGE CUSTOMER FEATURES BACK TO ORDER DATA
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 3: MERGE CUSTOMER FEATURES")
    print("=" * 60)

    # Select columns to merge (RFM features)
    merge_cols = ['customer_unique_id'] + RFM_FEATURES

    for name in ['train', 'val', 'test']:
        df = processed[name]

        # Merge RFM features
        df = df.merge(
            customer_features[merge_cols],
            on='customer_unique_id',
            how='left'
        )

        # Fill NaN for new customers (not in training set)
        for col in RFM_FEATURES:
            if col in df.columns:
                median_val = customer_features[col].median()
                df[col].fillna(median_val, inplace=True)

        processed[name] = df
        print(f"  ✓ Merged RFM features to {name}: {len(df):,} rows")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    train_featured = processed['train']
    print(f"\nFinal feature counts:")
    print(f"  Train: {train_featured.shape[0]:,} rows × {train_featured.shape[1]} columns")
    print(f"  Val: {processed['val'].shape[0]:,} rows × {processed['val'].shape[1]} columns")
    print(f"  Test: {processed['test'].shape[0]:,} rows × {processed['test'].shape[1]} columns")

    # List new features
    original_cols = set(train_df.columns)
    new_cols = set(train_featured.columns) - original_cols
    print(f"\nNew features created: {len(new_cols)}")

    return {
        'train': processed['train'],
        'val': processed['val'],
        'test': processed['test'],
        'customer_features': customer_features,
        'category_stats': category_stats,
        'reference_date': reference_date,
    }


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_feature_artifacts(
    result: Dict,
    output_dir: str = "data/processed",
    models_dir: str = "models",
) -> None:
    """
    Save all feature engineering artifacts.

    Args:
        result: Dictionary from engineer_features()
        output_dir: Directory for processed data
        models_dir: Directory for model artifacts
    """
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    # Save featured DataFrames
    result['train'].to_parquet(output_dir / "train_featured.parquet", index=False)
    result['val'].to_parquet(output_dir / "val_featured.parquet", index=False)
    result['test'].to_parquet(output_dir / "test_featured.parquet", index=False)
    print(f"✓ Saved featured parquet files to {output_dir}")

    # Save customer-level features
    result['customer_features'].to_parquet(
        output_dir / "customer_segments.parquet", index=False
    )
    print(f"✓ Saved customer segments to {output_dir}/customer_segments.parquet")

    # Save category stats
    result['category_stats'].to_csv(models_dir / "category_stats.csv", index=False)
    print(f"✓ Saved category stats to {models_dir}/category_stats.csv")

    # Save feature names
    feature_names = {
        'numerical': NUMERICAL_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'binary': BINARY_FEATURES,
        'rfm': RFM_FEATURES,
        'customer_behavioral': CUSTOMER_BEHAVIORAL_FEATURES,
        'reference_date': str(result['reference_date']),
    }

    with open(models_dir / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Saved feature names to {models_dir}/feature_names.json")

    print("\n✅ All artifacts saved successfully!")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_featured_data(data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved featured DataFrames."""
    data_dir = Path(data_dir)

    train = pd.read_parquet(data_dir / "train_featured.parquet")
    val = pd.read_parquet(data_dir / "val_featured.parquet")
    test = pd.read_parquet(data_dir / "test_featured.parquet")

    print(f"Loaded featured data: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")

    return train, val, test


def load_customer_segments(data_dir: str = "data/processed") -> pd.DataFrame:
    """Load customer-level segment data."""
    return pd.read_parquet(Path(data_dir) / "customer_segments.parquet")


def load_feature_config(models_dir: str = "models") -> Dict:
    """Load feature configuration."""
    with open(Path(models_dir) / "feature_names.json") as f:
        return json.load(f)
