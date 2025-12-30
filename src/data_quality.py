"""
Data Quality utilities for Olist dataset.

This module handles:
1. Missing value imputation
2. Outlier treatment (winsorization)
3. Data type corrections
4. Schema validation with Pandera

IMPORTANT: All transformers must be fit on TRAINING data only!
"""
import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, Check
from typing import Optional
import json
from pathlib import Path


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

class MissingValueHandler:
    """
    Handle missing values with different strategies.

    Strategies:
    - 'drop': Drop rows with missing values
    - 'mean': Fill with mean (numerical)
    - 'median': Fill with median (numerical)
    - 'mode': Fill with mode (categorical)
    - 'constant': Fill with a constant value
    - 'keep': Keep as-is (for nullable columns)
    """

    def __init__(self):
        self.fill_values = {}  # Stores fitted values per column
        self.strategies = {}   # Stores strategy per column

    def fit(self, df: pd.DataFrame, column_strategies: dict) -> "MissingValueHandler":
        """
        Fit the handler on training data.

        Args:
            df: Training DataFrame
            column_strategies: Dict of {column: (strategy, optional_value)}
                e.g., {"price": ("median", None), "category": ("constant", "unknown")}
        """
        self.strategies = column_strategies

        for column, (strategy, value) in column_strategies.items():
            if column not in df.columns:
                continue

            if strategy == "mean":
                self.fill_values[column] = df[column].mean()
            elif strategy == "median":
                self.fill_values[column] = df[column].median()
            elif strategy == "mode":
                mode_result = df[column].mode()
                self.fill_values[column] = mode_result.iloc[0] if len(mode_result) > 0 else None
            elif strategy == "constant":
                self.fill_values[column] = value
            elif strategy in ("drop", "keep"):
                self.fill_values[column] = None  # No fill needed
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted missing value handling to DataFrame."""
        df = df.copy()

        for column, (strategy, _) in self.strategies.items():
            if column not in df.columns:
                continue

            if strategy == "drop":
                df = df.dropna(subset=[column])
            elif strategy != "keep" and self.fill_values.get(column) is not None:
                fill_value = self.fill_values[column]

                # Handle categorical columns - need to add category first
                if hasattr(df[column], 'cat'):
                    if fill_value not in df[column].cat.categories:
                        df[column] = df[column].cat.add_categories([fill_value])

                df[column] = df[column].fillna(fill_value)

        return df

    def fit_transform(self, df: pd.DataFrame, column_strategies: dict) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, column_strategies).transform(df)

    def get_fill_values(self) -> dict:
        """Return fitted fill values (for saving)."""
        return {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in self.fill_values.items() if v is not None}


# Recommended strategies for Olist dataset
OLIST_IMPUTATION_STRATEGIES = {
    # Drop columns with too many missing values
    "review_comment_title": ("keep", None),     # 99% missing, not useful
    "review_comment_message": ("keep", None),   # 57% missing, keep for NLP if needed

    # Keep null for undelivered orders (valid business meaning)
    "order_delivered_customer_date": ("keep", None),
    "order_delivered_carrier_date": ("keep", None),
    "order_approved_at": ("keep", None),
    "delivery_days": ("keep", None),
    "delivery_delay_days": ("keep", None),
    "is_late_delivery": ("keep", None),
    "is_satisfied": ("keep", None),  # Will be null if no review

    # Fill product attributes with median
    "product_weight_g": ("median", None),
    "product_length_cm": ("median", None),
    "product_height_cm": ("median", None),
    "product_width_cm": ("median", None),
    "product_photos_qty": ("median", None),
    "product_name_lenght": ("median", None),
    "product_description_lenght": ("median", None),

    # Fill categorical with 'unknown'
    "product_category_name": ("constant", "unknown"),
    "product_category_name_english": ("constant", "unknown"),
}


# =============================================================================
# OUTLIER HANDLING
# =============================================================================

class OutlierHandler:
    """
    Handle outliers using winsorization (capping at percentiles).

    This approach:
    - Preserves all data points (no dropping)
    - Limits extreme values to reasonable bounds
    - Is fitted on training data to ensure consistency
    """

    def __init__(self, lower_pct: float = 0.01, upper_pct: float = 0.99):
        """
        Args:
            lower_pct: Lower percentile for capping (default 1%)
            upper_pct: Upper percentile for capping (default 99%)
        """
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.bounds = {}  # {column: (lower_bound, upper_bound)}

    def fit(self, df: pd.DataFrame, columns: list) -> "OutlierHandler":
        """
        Calculate bounds from training data.

        Args:
            df: Training DataFrame
            columns: List of numerical columns to handle
        """
        for column in columns:
            if column not in df.columns:
                continue

            # Check if column is numeric (handles both numpy and pandas nullable types)
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue

            lower = df[column].quantile(self.lower_pct)
            upper = df[column].quantile(self.upper_pct)
            self.bounds[column] = (lower, upper)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply capping to DataFrame using fitted bounds."""
        df = df.copy()

        for column, (lower, upper) in self.bounds.items():
            if column in df.columns:
                original_min = df[column].min()
                original_max = df[column].max()
                df[column] = df[column].clip(lower, upper)

        return df

    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, columns).transform(df)

    def get_bounds(self) -> dict:
        """Return fitted bounds (for saving)."""
        return {k: (float(v[0]), float(v[1])) for k, v in self.bounds.items()}


# Columns to winsorize for Olist dataset
OLIST_OUTLIER_COLUMNS = [
    "price",
    "freight_value",
    "payment_value",
    "payment_installments",
    "delivery_days",
    "delivery_delay_days",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]


# =============================================================================
# DATA TYPE CORRECTIONS
# =============================================================================

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix and optimize data types.

    - Convert object columns with few unique values to category
    - Ensure proper nullable int types
    - Convert date columns if needed
    """
    df = df.copy()

    # Columns that should be categorical
    categorical_columns = [
        "order_status",
        "customer_state",
        "seller_state",
        "product_category_name",
        "product_category_name_english",
        "payment_type",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Ensure integer columns are nullable int
    int_columns = [
        "order_item_id",
        "payment_installments",
        "payment_methods_count",
        "product_photos_qty",
        "is_satisfied",
        "is_late_delivery",
    ]

    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    return df


# =============================================================================
# SCHEMA VALIDATION WITH PANDERA
# =============================================================================

def create_olist_schema() -> pa.DataFrameSchema:
    """
    Create Pandera schema for Olist dataset.

    This validates:
    - Column presence and types
    - Value ranges and constraints
    - Business logic rules
    """
    schema = pa.DataFrameSchema(
        columns={
            # Core identifiers (required)
            "order_id": Column(str, nullable=False),
            "customer_id": Column(str, nullable=False),

            # Order status
            "order_status": Column(
                str,
                Check.isin([
                    "delivered", "shipped", "canceled", "unavailable",
                    "invoiced", "processing", "created", "approved"
                ]),
                nullable=False
            ),

            # Prices (non-negative)
            "price": Column(float, Check.ge(0), nullable=True),
            "freight_value": Column(float, Check.ge(0), nullable=True),
            "payment_value": Column(float, Check.ge(0), nullable=True),

            # Review score (1-5)
            "review_score": Column(float, Check.in_range(1, 5), nullable=True),

            # Target variables
            "is_satisfied": Column("Int64", Check.isin([0, 1]), nullable=True),
            "delivery_days": Column(float, nullable=True),  # Can be null for undelivered
            "is_late_delivery": Column("Int64", Check.isin([0, 1]), nullable=True),

            # Product dimensions (non-negative)
            "product_weight_g": Column(float, Check.ge(0), nullable=True),
            "product_length_cm": Column(float, Check.ge(0), nullable=True),
            "product_height_cm": Column(float, Check.ge(0), nullable=True),
            "product_width_cm": Column(float, Check.ge(0), nullable=True),
        },
        # Only check columns that are defined (allow extra columns)
        strict=False,
        # Return all errors at once
        coerce=False,
    )

    return schema


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> bool:
    """
    Validate a DataFrame against the Olist schema.

    Args:
        df: DataFrame to validate
        name: Name for error messages

    Returns:
        True if valid, raises error if invalid
    """
    schema = create_olist_schema()

    try:
        schema.validate(df, lazy=True)
        print(f"✓ {name} passed schema validation")
        return True
    except pa.errors.SchemaErrors as e:
        print(f"✗ {name} failed schema validation:")
        print(f"  {e.failure_cases}")
        raise


# =============================================================================
# FULL PROCESSING PIPELINE
# =============================================================================

def process_training_data(
    train_df: pd.DataFrame,
    impute: bool = True,
    winsorize: bool = True,
    validate: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Full processing pipeline for training data.

    Returns:
        - Processed DataFrame
        - Dictionary of fitted parameters (for applying to val/test)
    """
    print("\n🔧 Processing training data...")
    artifacts = {}

    # 1. Fix data types
    train_df = fix_data_types(train_df)
    print("  ✓ Fixed data types")

    # 2. Handle missing values
    if impute:
        handler = MissingValueHandler()
        train_df = handler.fit_transform(train_df, OLIST_IMPUTATION_STRATEGIES)
        artifacts["imputation"] = handler.get_fill_values()
        print(f"  ✓ Imputed missing values ({len(artifacts['imputation'])} columns)")

    # 3. Handle outliers
    if winsorize:
        outlier_handler = OutlierHandler()
        train_df = outlier_handler.fit_transform(train_df, OLIST_OUTLIER_COLUMNS)
        artifacts["outlier_bounds"] = outlier_handler.get_bounds()
        print(f"  ✓ Winsorized outliers ({len(artifacts['outlier_bounds'])} columns)")

    # 4. Validate schema
    if validate:
        validate_dataframe(train_df, "Training data")

    print(f"\n✓ Training data processed: {len(train_df):,} rows")

    return train_df, artifacts


def process_inference_data(
    df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    """
    Process validation/test data using fitted artifacts from training.

    IMPORTANT: Use the exact same transformations as training!
    """
    print("\n🔧 Processing inference data...")

    # 1. Fix data types
    df = fix_data_types(df)

    # 2. Apply imputation with training values
    if "imputation" in artifacts:
        for column, value in artifacts["imputation"].items():
            if column in df.columns:
                # Handle categorical columns - need to add category first
                if hasattr(df[column], 'cat'):
                    if value not in df[column].cat.categories:
                        df[column] = df[column].cat.add_categories([value])
                df[column] = df[column].fillna(value)

    # 3. Apply outlier bounds from training
    if "outlier_bounds" in artifacts:
        for column, (lower, upper) in artifacts["outlier_bounds"].items():
            if column in df.columns:
                df[column] = df[column].clip(lower, upper)

    print(f"✓ Inference data processed: {len(df):,} rows")

    return df


def save_artifacts(artifacts: dict, output_path: str = "models/data_processing_artifacts.json") -> None:
    """Save processing artifacts for later use."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"✓ Artifacts saved to {output_path}")


def load_artifacts(input_path: str = "models/data_processing_artifacts.json") -> dict:
    """Load processing artifacts."""
    with open(input_path) as f:
        return json.load(f)
