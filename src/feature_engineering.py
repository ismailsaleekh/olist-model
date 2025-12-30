"""Feature engineering utilities."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_feature_pipeline() -> Pipeline:
    """Create the feature engineering pipeline."""
    pass


def compute_rfm_features(df):
    """Compute RFM (Recency, Frequency, Monetary) features."""
    pass


def compute_temporal_features(df):
    """Compute temporal features from timestamps."""
    pass


def compute_geographic_features(df):
    """Compute geographic features."""
    pass
