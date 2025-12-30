"""Prediction utilities."""
from joblib import load
import pandas as pd


def load_model(model_path: str):
    """Load a trained model from disk."""
    return load(model_path)


def predict_satisfaction(input_data: pd.DataFrame) -> list:
    """Predict customer satisfaction."""
    pass


def predict_delivery_time(input_data: pd.DataFrame) -> list:
    """Predict delivery time."""
    pass
