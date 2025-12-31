"""GCP utility functions for Vertex AI and Cloud Storage."""
import yaml
import time
from pathlib import Path
from typing import Optional, List, Dict
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


# =============================================================================
# CLOUD STORAGE FUNCTIONS
# =============================================================================

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


def upload_directory_to_gcs(
    local_dir: str,
    gcs_prefix: str,
    config: dict,
    file_pattern: str = "*",
) -> List[str]:
    """
    Upload all files matching pattern from local directory to GCS.

    Args:
        local_dir: Local directory path
        gcs_prefix: GCS prefix (folder) to upload to
        config: Configuration dict
        file_pattern: Glob pattern for files (default: "*")

    Returns:
        List of GCS URIs for uploaded files
    """
    local_path = Path(local_dir)
    uploaded = []

    for file_path in local_path.glob(file_pattern):
        if file_path.is_file():
            gcs_path = f"{gcs_prefix}/{file_path.name}"
            uri = upload_to_gcs(str(file_path), gcs_path, config)
            uploaded.append(uri)
            print(f"  Uploaded: {file_path.name} -> {uri}")

    return uploaded


def download_directory_from_gcs(
    gcs_prefix: str,
    local_dir: str,
    config: dict,
) -> List[str]:
    """
    Download all files from GCS prefix to local directory.

    Args:
        gcs_prefix: GCS prefix (folder) to download from
        local_dir: Local directory to download to
        config: Configuration dict

    Returns:
        List of local file paths downloaded
    """
    client = storage.Client(project=config["gcp"]["project_id"])
    bucket = client.bucket(config["gcp"]["bucket"])
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    downloaded = []
    blobs = bucket.list_blobs(prefix=gcs_prefix)

    for blob in blobs:
        if not blob.name.endswith('/'):  # Skip directories
            filename = Path(blob.name).name
            local_file = local_path / filename
            blob.download_to_filename(str(local_file))
            downloaded.append(str(local_file))
            print(f"  Downloaded: {blob.name} -> {local_file}")

    return downloaded


# =============================================================================
# VERTEX AI CUSTOM TRAINING JOBS
# =============================================================================

def submit_custom_training_job(
    config: dict,
    display_name: str,
    script_path: str,
    args: List[str] = None,
    requirements: List[str] = None,
    environment_variables: Dict[str, str] = None,
    sync: bool = True,
) -> aiplatform.CustomJob:
    """
    Submit a custom training job to Vertex AI.

    Args:
        config: Configuration dict with GCP and Vertex AI settings
        display_name: Display name for the job
        script_path: Local path to the Python training script
        args: Command line arguments to pass to the script
        requirements: Additional pip packages to install
        environment_variables: Environment variables for the job
        sync: If True, wait for job completion (default: True)

    Returns:
        CustomJob object
    """
    # Initialize Vertex AI
    aiplatform.init(
        project=config["gcp"]["project_id"],
        location=config["gcp"]["region"],
        staging_bucket=config["vertex_ai_training"]["staging_bucket"],
    )

    training_config = config["vertex_ai_training"]

    # Default requirements for clustering
    if requirements is None:
        requirements = [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "pyarrow>=14.0.0",
            "gcsfs>=2023.1.0",
        ]

    # Create the custom job
    job = aiplatform.CustomJob.from_local_script(
        display_name=display_name,
        script_path=script_path,
        container_uri=training_config["container_uri"],
        requirements=requirements,
        args=args or [],
        environment_variables=environment_variables or {},
        machine_type=training_config["machine_type"],
        base_output_dir=training_config["output_dir"],
    )

    print(f"Submitting training job: {display_name}")
    print(f"  Machine type: {training_config['machine_type']}")
    print(f"  Container: {training_config['container_uri']}")
    print(f"  Output dir: {training_config['output_dir']}")

    # Run the job
    job.run(
        sync=sync,
        service_account=training_config.get("service_account"),
    )

    if sync:
        print(f"Job completed: {job.display_name}")
        print(f"  State: {job.state}")

    return job


def get_job_status(job_name: str, config: dict) -> str:
    """
    Get the status of a Vertex AI custom job.

    Args:
        job_name: Full resource name of the job
        config: Configuration dict

    Returns:
        Job state string
    """
    aiplatform.init(
        project=config["gcp"]["project_id"],
        location=config["gcp"]["region"],
    )

    job = aiplatform.CustomJob.get(job_name)
    return job.state.name


def list_training_jobs(
    config: dict,
    filter_str: str = None,
    limit: int = 10,
) -> List[aiplatform.CustomJob]:
    """
    List recent custom training jobs.

    Args:
        config: Configuration dict
        filter_str: Optional filter string
        limit: Maximum number of jobs to return

    Returns:
        List of CustomJob objects
    """
    aiplatform.init(
        project=config["gcp"]["project_id"],
        location=config["gcp"]["region"],
    )

    jobs = aiplatform.CustomJob.list(
        filter=filter_str,
        order_by="create_time desc",
    )

    return list(jobs)[:limit]


# =============================================================================
# CONVENIENCE FUNCTIONS FOR CLUSTERING PIPELINE
# =============================================================================

def upload_clustering_data(config: dict) -> str:
    """
    Upload customer segments data to GCS for Vertex AI training.

    Args:
        config: Configuration dict

    Returns:
        GCS URI of uploaded file
    """
    local_path = "data/processed/customer_segments.parquet"
    gcs_path = "data/customer_segments.parquet"

    print("Uploading clustering data to GCS...")
    uri = upload_to_gcs(local_path, gcs_path, config)
    print(f"  Uploaded: {uri}")

    return uri


def download_clustering_artifacts(config: dict, local_dir: str = "models") -> List[str]:
    """
    Download clustering artifacts from GCS after training job.

    Args:
        config: Configuration dict
        local_dir: Local directory to save artifacts

    Returns:
        List of downloaded file paths
    """
    print("Downloading clustering artifacts from GCS...")

    # Download from training output directory
    gcs_prefix = "training-output/model"
    downloaded = download_directory_from_gcs(gcs_prefix, local_dir, config)

    # Also download from models directory if exists
    gcs_prefix_models = "models"
    try:
        downloaded.extend(download_directory_from_gcs(gcs_prefix_models, local_dir, config))
    except Exception:
        pass  # Directory might not exist

    print(f"Downloaded {len(downloaded)} artifacts")
    return downloaded


def run_clustering_on_vertex_ai(
    config: dict,
    optimal_k: int = 4,
    sync: bool = True,
) -> aiplatform.CustomJob:
    """
    Run the full clustering pipeline on Vertex AI.

    This function:
    1. Uploads customer_segments.parquet to GCS
    2. Submits a custom training job
    3. Waits for completion (if sync=True)

    Args:
        config: Configuration dict
        optimal_k: Number of clusters for K-Means
        sync: Wait for job completion

    Returns:
        CustomJob object
    """
    # Step 1: Upload data
    data_uri = upload_clustering_data(config)

    # Step 2: Generate job name
    timestamp = int(time.time())
    job_name = f"{config['vertex_ai_training']['job_prefix']}-{timestamp}"

    # Step 3: Submit training job
    job = submit_custom_training_job(
        config=config,
        display_name=job_name,
        script_path="src/vertex_training.py",
        args=[
            f"--input-data={data_uri}",
            f"--output-dir=gs://{config['gcp']['bucket']}/models",
            f"--optimal-k={optimal_k}",
            f"--bucket={config['gcp']['bucket']}",
        ],
    )

    return job
