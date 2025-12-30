"""
Exploratory Data Analysis utilities for Olist dataset.

IMPORTANT: All analysis is performed on TRAINING SET ONLY to prevent data leakage.

This module provides functions for:
1. Basic statistics and distributions
2. Target variable analysis
3. Missing value analysis
4. Outlier detection
5. Correlation analysis
"""
import pandas as pd
import numpy as np
from typing import Optional
import json
from pathlib import Path


# =============================================================================
# BASIC STATISTICS
# =============================================================================

def get_basic_stats(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """
    Get basic statistics for a DataFrame.

    Args:
        df: DataFrame to analyze
        name: Name for display purposes

    Returns:
        Dictionary of statistics
    """
    stats = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "duplicates": df.duplicated().sum(),
        "total_missing": df.isnull().sum().sum(),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
    }

    print(f"\n📊 {name} Statistics:")
    print(f"  Rows: {stats['rows']:,}")
    print(f"  Columns: {stats['columns']}")
    print(f"  Memory: {stats['memory_mb']} MB")
    print(f"  Missing values: {stats['total_missing']:,} ({stats['missing_pct']}%)")

    return stats


def analyze_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze data types and unique values per column."""
    analysis = []

    for col in df.columns:
        analysis.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": df[col].notna().sum(),
            "null_count": df[col].isna().sum(),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique": df[col].nunique(),
            "sample": str(df[col].dropna().iloc[0]) if df[col].notna().any() else None
        })

    return pd.DataFrame(analysis)


# =============================================================================
# TARGET VARIABLE ANALYSIS
# =============================================================================

def analyze_classification_target(df: pd.DataFrame, target_col: str = "is_satisfied") -> dict:
    """
    Analyze classification target distribution.

    Checks for class imbalance and provides recommendations.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Drop nulls for analysis
    target = df[target_col].dropna()

    # Calculate distribution
    value_counts = target.value_counts()
    value_pct = target.value_counts(normalize=True)

    # Imbalance ratio
    majority_class = value_counts.idxmax()
    minority_class = value_counts.idxmin()
    imbalance_ratio = value_counts[majority_class] / value_counts[minority_class]

    analysis = {
        "target_column": target_col,
        "total_samples": len(target),
        "null_count": df[target_col].isna().sum(),
        "class_distribution": value_counts.to_dict(),
        "class_percentages": {k: round(v * 100, 2) for k, v in value_pct.to_dict().items()},
        "majority_class": int(majority_class),
        "minority_class": int(minority_class),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "is_imbalanced": imbalance_ratio > 3,  # Common threshold
    }

    print(f"\n🎯 Classification Target: {target_col}")
    print(f"  Total samples: {analysis['total_samples']:,}")
    print(f"  Class 0 (Unsatisfied): {value_counts.get(0, 0):,} ({value_pct.get(0, 0)*100:.1f}%)")
    print(f"  Class 1 (Satisfied): {value_counts.get(1, 0):,} ({value_pct.get(1, 0)*100:.1f}%)")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

    if analysis["is_imbalanced"]:
        print("  ⚠️  Dataset is imbalanced! Consider: class weights, SMOTE, or undersampling")
    else:
        print("  ✓ Dataset is reasonably balanced")

    return analysis


def analyze_regression_target(df: pd.DataFrame, target_col: str = "delivery_days") -> dict:
    """
    Analyze regression target distribution.

    Checks for skewness and outliers.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Drop nulls and infinities
    target = df[target_col].replace([np.inf, -np.inf], np.nan).dropna()

    # Basic statistics
    stats = target.describe()

    # Skewness and kurtosis
    skewness = target.skew()
    kurtosis = target.kurtosis()

    # IQR for outlier detection
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((target < lower_bound) | (target > upper_bound)).sum()

    analysis = {
        "target_column": target_col,
        "total_samples": len(target),
        "null_count": df[target_col].isna().sum(),
        "mean": round(stats["mean"], 2),
        "median": round(stats["50%"], 2),
        "std": round(stats["std"], 2),
        "min": round(stats["min"], 2),
        "max": round(stats["max"], 2),
        "skewness": round(skewness, 2),
        "kurtosis": round(kurtosis, 2),
        "outliers_count": int(outliers),
        "outliers_pct": round(outliers / len(target) * 100, 2),
        "iqr_lower": round(lower_bound, 2),
        "iqr_upper": round(upper_bound, 2),
        "needs_transform": abs(skewness) > 1,  # Common threshold
    }

    print(f"\n📈 Regression Target: {target_col}")
    print(f"  Total samples: {analysis['total_samples']:,}")
    print(f"  Mean: {analysis['mean']:.2f} | Median: {analysis['median']:.2f}")
    print(f"  Std: {analysis['std']:.2f} | Range: [{analysis['min']:.2f}, {analysis['max']:.2f}]")
    print(f"  Skewness: {analysis['skewness']:.2f} | Kurtosis: {analysis['kurtosis']:.2f}")
    print(f"  Outliers (IQR): {analysis['outliers_count']:,} ({analysis['outliers_pct']:.1f}%)")

    if analysis["needs_transform"]:
        print("  ⚠️  Distribution is skewed. Consider: log transform or Box-Cox")
    else:
        print("  ✓ Distribution is approximately normal")

    return analysis


# =============================================================================
# MISSING VALUE ANALYSIS
# =============================================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive missing value analysis.

    Returns DataFrame with columns sorted by missing percentage.
    """
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).values,
        "dtype": df.dtypes.astype(str).values,
    })

    # Add suggested imputation strategy
    def suggest_strategy(row):
        if row["missing_pct"] == 0:
            return "none"
        elif row["missing_pct"] > 50:
            return "drop_column"
        elif row["dtype"] in ["object", "category"]:
            return "mode_or_unknown"
        elif row["dtype"] in ["int64", "float64", "Int64", "Float64"]:
            return "median"
        elif "datetime" in row["dtype"]:
            return "keep_null"
        else:
            return "investigate"

    missing["suggested_strategy"] = missing.apply(suggest_strategy, axis=1)

    # Sort by missing percentage
    missing = missing.sort_values("missing_pct", ascending=False)

    # Print summary
    cols_with_missing = missing[missing["missing_count"] > 0]
    print(f"\n🔍 Missing Value Analysis:")
    print(f"  Columns with missing values: {len(cols_with_missing)}/{len(df.columns)}")

    if len(cols_with_missing) > 0:
        print("\n  Top 10 columns with missing values:")
        for _, row in cols_with_missing.head(10).iterrows():
            print(f"    {row['column']}: {row['missing_count']:,} ({row['missing_pct']}%) → {row['suggested_strategy']}")

    return missing


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers using IQR method.

    Args:
        df: DataFrame to analyze
        columns: Numerical columns to check (default: all numeric)
        multiplier: IQR multiplier (default 1.5 for standard, 3 for extreme)

    Returns:
        DataFrame with outlier statistics per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()
        if len(data) == 0:
            continue

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        outliers_low = (data < lower).sum()
        outliers_high = (data > upper).sum()
        outliers_total = outliers_low + outliers_high

        results.append({
            "column": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "outliers_low": int(outliers_low),
            "outliers_high": int(outliers_high),
            "outliers_total": int(outliers_total),
            "outliers_pct": round(outliers_total / len(data) * 100, 2),
        })

    outliers_df = pd.DataFrame(results).sort_values("outliers_pct", ascending=False)

    # Print summary
    print(f"\n📊 Outlier Detection (IQR × {multiplier}):")
    for _, row in outliers_df[outliers_df["outliers_total"] > 0].head(10).iterrows():
        print(f"  {row['column']}: {row['outliers_total']:,} outliers ({row['outliers_pct']}%)")
        print(f"    Range: [{row['lower_bound']}, {row['upper_bound']}]")

    return outliers_df


def winsorize_column(
    df: pd.DataFrame,
    column: str,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pd.DataFrame:
    """
    Winsorize (cap) outliers at specified percentiles.

    Args:
        df: DataFrame
        column: Column to winsorize
        lower_pct: Lower percentile (default 1%)
        upper_pct: Upper percentile (default 99%)

    Returns:
        DataFrame with winsorized column
    """
    df = df.copy()

    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)

    original_min = df[column].min()
    original_max = df[column].max()

    df[column] = df[column].clip(lower, upper)

    print(f"  Winsorized {column}: [{original_min:.2f}, {original_max:.2f}] → [{lower:.2f}, {upper:.2f}]")

    return df


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    Analyze correlations between numerical features.

    Args:
        df: DataFrame
        target_col: If provided, focus on correlations with target
        threshold: Correlation threshold for "high" correlation

    Returns:
        Correlation matrix or series
    """
    # Get numerical columns only
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) == 0:
        print("No numerical columns found!")
        return pd.DataFrame()

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    if target_col and target_col in corr_matrix.columns:
        # Focus on correlations with target
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)

        print(f"\n📊 Correlations with '{target_col}':")
        print("  Top 10 features:")
        for feat, corr in target_corr.head(10).items():
            direction = "+" if corr > 0 else ""
            print(f"    {feat}: {direction}{corr:.3f}")

        return target_corr

    else:
        # Find highly correlated pairs
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": round(corr, 3),
                    })

        if high_corr_pairs:
            print(f"\n📊 Highly Correlated Feature Pairs (|r| >= {threshold}):")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True):
                print(f"  {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")

        return corr_matrix


# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

def analyze_temporal_patterns(df: pd.DataFrame, date_col: str = "order_purchase_timestamp") -> dict:
    """Analyze temporal patterns in the data."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found")

    dates = pd.to_datetime(df[date_col])

    analysis = {
        "date_range": {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "span_days": (dates.max() - dates.min()).days,
        },
        "orders_by_dayofweek": dates.dt.dayofweek.value_counts().sort_index().to_dict(),
        "orders_by_month": dates.dt.month.value_counts().sort_index().to_dict(),
        "orders_by_hour": dates.dt.hour.value_counts().sort_index().to_dict(),
    }

    print(f"\n📅 Temporal Analysis:")
    print(f"  Date range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
    print(f"  Span: {analysis['date_range']['span_days']} days")

    # Day of week analysis
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("\n  Orders by day of week:")
    for dow, count in analysis["orders_by_dayofweek"].items():
        print(f"    {dow_names[dow]}: {count:,}")

    return analysis


# =============================================================================
# CATEGORICAL ANALYSIS
# =============================================================================

def analyze_categorical(df: pd.DataFrame, column: str, top_n: int = 10) -> dict:
    """Analyze a categorical column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    value_counts = df[column].value_counts()
    value_pct = df[column].value_counts(normalize=True)

    analysis = {
        "column": column,
        "unique_values": df[column].nunique(),
        "null_count": df[column].isna().sum(),
        "top_values": value_counts.head(top_n).to_dict(),
        "top_percentages": {k: round(v * 100, 2) for k, v in value_pct.head(top_n).to_dict().items()},
    }

    print(f"\n📊 Categorical: {column}")
    print(f"  Unique values: {analysis['unique_values']}")
    print(f"  Top {top_n}:")
    for val, count in list(analysis["top_values"].items())[:top_n]:
        pct = analysis["top_percentages"][val]
        print(f"    {val}: {count:,} ({pct}%)")

    return analysis


# =============================================================================
# SAVE EDA RESULTS
# =============================================================================

def save_eda_results(results: dict, output_path: str = "data/processed/eda_statistics.json") -> None:
    """Save EDA results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    results = convert_types(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ EDA results saved to {output_path}")
