"""
Data loading utilities for Olist E-Commerce dataset.

This module handles:
1. Loading all 9 raw CSV files
2. Parsing date columns
3. Merging datasets into a unified DataFrame
4. Creating time-based train/val/test splits
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# =============================================================================
# CONSTANTS
# =============================================================================

# All CSV files in the dataset
CSV_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Date columns that need parsing
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data(data_path: str = "data/raw/") -> dict[str, pd.DataFrame]:
    """
    Load all 9 raw CSV files into a dictionary of DataFrames.

    Args:
        data_path: Path to the directory containing raw CSV files

    Returns:
        Dictionary mapping dataset names to DataFrames

    Example:
        >>> datasets = load_raw_data("data/raw/")
        >>> datasets["orders"].shape
        (99441, 8)
    """
    data_path = Path(data_path)
    datasets = {}

    for name, filename in CSV_FILES.items():
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing file: {filepath}")

        df = pd.read_csv(filepath)
        datasets[name] = df
        print(f"✓ Loaded {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    return datasets


def parse_date_columns(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Convert string date columns to datetime objects.

    Args:
        datasets: Dictionary of DataFrames from load_raw_data()

    Returns:
        Same dictionary with date columns converted to datetime
    """
    for dataset_name, columns in DATE_COLUMNS.items():
        if dataset_name not in datasets:
            continue

        df = datasets[dataset_name]
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        print(f"✓ Parsed {len(columns)} date columns in {dataset_name}")

    return datasets


# =============================================================================
# DATA SPLITTING
# =============================================================================

def create_time_based_split(
    orders_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create time-based train/val/test split using order_purchase_timestamp.

    WHY TIME-BASED SPLIT?
    - More realistic for production: train on past data, predict future
    - Prevents data leakage from future orders
    - Simulates real deployment scenario

    Args:
        orders_df: Orders DataFrame with order_purchase_timestamp column
        train_ratio: Fraction of data for training (default 0.70)
        val_ratio: Fraction of data for validation (default 0.15)
        test_ratio: Fraction of data for testing (default 0.15)

    Returns:
        Tuple of (train_order_ids, val_order_ids, test_order_ids)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Sort by timestamp (oldest first)
    orders_sorted = orders_df.sort_values("order_purchase_timestamp")

    # Calculate split indices
    n = len(orders_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Extract order_ids for each split
    train_ids = orders_sorted.iloc[:train_end]["order_id"].values
    val_ids = orders_sorted.iloc[train_end:val_end]["order_id"].values
    test_ids = orders_sorted.iloc[val_end:]["order_id"].values

    # Get date ranges for each split
    train_dates = orders_sorted.iloc[:train_end]["order_purchase_timestamp"]
    val_dates = orders_sorted.iloc[train_end:val_end]["order_purchase_timestamp"]
    test_dates = orders_sorted.iloc[val_end:]["order_purchase_timestamp"]

    print(f"\n📊 Time-Based Split Results:")
    print(f"  Train: {len(train_ids):,} orders ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"  Val:   {len(val_ids):,} orders ({val_dates.min().date()} to {val_dates.max().date()})")
    print(f"  Test:  {len(test_ids):,} orders ({test_dates.min().date()} to {test_dates.max().date()})")

    return train_ids, val_ids, test_ids


def save_split_ids(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    output_path: str = "data/splits/order_id_splits.json",
) -> None:
    """
    Save split order IDs to JSON file for reproducibility.

    Args:
        train_ids, val_ids, test_ids: Arrays of order IDs
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist(),
        "metadata": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "split_strategy": "time_based",
        }
    }

    with open(output_path, "w") as f:
        json.dump(splits, f)

    print(f"✓ Saved split IDs to {output_path}")


def load_split_ids(input_path: str = "data/splits/order_id_splits.json") -> dict:
    """Load previously saved split order IDs."""
    with open(input_path) as f:
        return json.load(f)


# =============================================================================
# DATA MERGING
# =============================================================================

def merge_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all datasets into a single unified DataFrame.

    MERGE ORDER:
    1. orders (base table - one row per order)
    2. + order_items (expands to one row per item)
    3. + products (with English category names)
    4. + customers (customer info)
    5. + sellers (seller info)
    6. + reviews (review scores and comments)
    7. + payments (aggregated per order)

    Args:
        datasets: Dictionary of DataFrames from load_raw_data()

    Returns:
        Merged DataFrame with all information
    """
    print("\n🔗 Merging datasets...")

    # Start with orders
    merged = datasets["orders"].copy()
    print(f"  Base (orders): {len(merged):,} rows")

    # 1. Merge order_items (1:N - will expand rows)
    merged = merged.merge(
        datasets["order_items"],
        on="order_id",
        how="left"
    )
    print(f"  + order_items: {len(merged):,} rows")

    # 2. Merge products with category translation
    products = datasets["products"].merge(
        datasets["category_translation"],
        on="product_category_name",
        how="left"
    )
    merged = merged.merge(
        products,
        on="product_id",
        how="left"
    )
    print(f"  + products: {len(merged):,} rows")

    # 3. Merge customers
    merged = merged.merge(
        datasets["customers"],
        on="customer_id",
        how="left"
    )
    print(f"  + customers: {len(merged):,} rows")

    # 4. Merge sellers (add suffix to avoid column name conflicts)
    merged = merged.merge(
        datasets["sellers"],
        on="seller_id",
        how="left",
        suffixes=("_customer", "_seller")
    )
    print(f"  + sellers: {len(merged):,} rows")

    # 5. Merge reviews
    merged = merged.merge(
        datasets["reviews"],
        on="order_id",
        how="left"
    )
    print(f"  + reviews: {len(merged):,} rows")

    # 6. Aggregate and merge payments
    # (one order can have multiple payment methods)
    payments_agg = datasets["payments"].groupby("order_id").agg({
        "payment_value": "sum",
        "payment_installments": "max",
        "payment_type": lambda x: x.mode().iloc[0] if len(x) > 0 else None,
        "payment_sequential": "count"  # number of payment methods used
    }).reset_index()
    payments_agg.rename(columns={"payment_sequential": "payment_methods_count"}, inplace=True)

    merged = merged.merge(
        payments_agg,
        on="order_id",
        how="left"
    )
    print(f"  + payments: {len(merged):,} rows")

    print(f"\n✓ Final merged dataset: {len(merged):,} rows × {len(merged.columns)} cols")

    return merged


# =============================================================================
# TARGET VARIABLE CREATION
# =============================================================================

def create_target_variables(df: pd.DataFrame, satisfaction_threshold: int = 4) -> pd.DataFrame:
    """
    Create target variables for supervised learning tasks.

    TARGETS:
    1. is_satisfied (classification): 1 if review_score >= threshold, else 0
    2. delivery_days (regression): Days from purchase to delivery

    ADDITIONAL FEATURES:
    - is_late_delivery: 1 if delivered after estimated date
    - delivery_delay_days: Actual - estimated delivery (negative = early)

    Args:
        df: Merged DataFrame
        satisfaction_threshold: Score threshold for satisfaction (default 4)

    Returns:
        DataFrame with new target columns
    """
    df = df.copy()

    # Classification target: Customer Satisfaction
    df["is_satisfied"] = (df["review_score"] >= satisfaction_threshold).astype("Int64")

    # Regression target: Delivery Time in Days
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)

    # Additional useful features
    df["is_late_delivery"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype("Int64")

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)

    # Stats
    satisfied_rate = df["is_satisfied"].mean()
    avg_delivery = df["delivery_days"].mean()
    late_rate = df["is_late_delivery"].mean()

    print(f"\n📌 Target Variables Created:")
    print(f"  is_satisfied: {satisfied_rate:.1%} satisfied (score >= {satisfaction_threshold})")
    print(f"  delivery_days: {avg_delivery:.1f} days average")
    print(f"  is_late_delivery: {late_rate:.1%} late deliveries")

    return df


# =============================================================================
# SPLIT APPLICATION
# =============================================================================

def apply_split(
    df: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply order_id splits to merged DataFrame.

    Args:
        df: Merged DataFrame with order_id column
        train_ids, val_ids, test_ids: Arrays of order IDs

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = df[df["order_id"].isin(train_ids)].copy()
    val_df = df[df["order_id"].isin(val_ids)].copy()
    test_df = df[df["order_id"].isin(test_ids)].copy()

    print(f"\n📦 Applied splits to merged data:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    return train_df, val_df, test_df


def save_splits_parquet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/splits/",
) -> None:
    """
    Save train/val/test DataFrames as parquet files.

    WHY PARQUET?
    - Much faster to read/write than CSV
    - Preserves data types (dates stay as dates)
    - Smaller file size with compression
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"\n✓ Saved parquet files to {output_dir}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_splits_parquet(splits_dir: str = "data/splits/") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train/val/test parquet files."""
    splits_dir = Path(splits_dir)

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    print(f"Loaded: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    return train_df, val_df, test_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get a summary of the DataFrame for quick inspection."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "missing_total": df.isnull().sum().sum(),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }
