"""GCP utility functions."""
import yaml
from google.cloud import aiplatform
from google.cloud import storage


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def init_vertex_ai(config: dict) -> None:
    """Initialize Vertex AI with project settings."""
    aiplatform.init(
        project=config["gcp"]["project_id"],
        location=config["gcp"]["region"],
        staging_bucket=config["vertex_ai"]["staging_bucket"],
        experiment=config["vertex_ai"]["experiment_name"],
    )


def upload_to_gcs(local_path: str, gcs_path: str, config: dict) -> str:
    """Upload a file to Google Cloud Storage."""
    client = storage.Client(project=config["gcp"]["project_id"])
    bucket = client.bucket(config["gcp"]["bucket"])
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return f"gs://{config['gcp']['bucket']}/{gcs_path}"


def download_from_gcs(gcs_path: str, local_path: str, config: dict) -> str:
    """Download a file from Google Cloud Storage."""
    client = storage.Client(project=config["gcp"]["project_id"])
    bucket = client.bucket(config["gcp"]["bucket"])
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    return local_path
