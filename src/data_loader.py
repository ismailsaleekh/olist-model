"""Data loading utilities."""
import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str = "data/raw/") -> dict[str, pd.DataFrame]:
    """Load all raw CSV files into a dictionary of DataFrames."""
    pass


def merge_datasets(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all datasets into a single DataFrame."""
    pass
