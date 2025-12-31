#!/usr/bin/env python3
"""
Vertex AI Training Script for Customer Clustering.

This script runs the clustering pipeline on Vertex AI Custom Training.
It reads data from GCS, trains clustering models, and saves artifacts back to GCS.

Usage (local testing):
    python src/vertex_training.py \
        --input-data=data/processed/customer_segments.parquet \
        --output-dir=models \
        --optimal-k=4

Usage (Vertex AI - called via gcp_utils.run_clustering_on_vertex_ai):
    python vertex_training.py \
        --input-data=gs://bucket/data/customer_segments.parquet \
        --output-dir=gs://bucket/models \
        --optimal-k=4 \
        --bucket=bucket-name
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage

# Suppress sklearn warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

CLUSTERING_FEATURES = [
    'recency',
    'frequency',
    'monetary',
    'avg_review_score',
    'avg_delivery_days',
    'late_delivery_rate',
]

LOG_TRANSFORM_FEATURES = ['frequency', 'monetary']


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_data(input_path: str) -> pd.DataFrame:
    """Load customer data from local path or GCS."""
    print(f"Loading data from: {input_path}")

    if input_path.startswith("gs://"):
        # Read from GCS using pandas with gcsfs
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_parquet(input_path)

    print(f"  Loaded {len(df):,} customers")
    print(f"  Columns: {list(df.columns)}")
    return df


def prepare_clustering_data(df: pd.DataFrame) -> tuple:
    """
    Prepare customer data for clustering.

    Returns:
        Tuple of (X_scaled, scaler, feature_names)
    """
    print("Preparing clustering data...")

    # Select and copy features
    X = df[CLUSTERING_FEATURES].copy()
    feature_names = []

    # Log transform skewed features
    X_processed = pd.DataFrame()
    for col in CLUSTERING_FEATURES:
        if col in LOG_TRANSFORM_FEATURES:
            X_processed[f'{col}_log'] = np.log1p(X[col])
            feature_names.append(f'{col}_log')
        else:
            X_processed[col] = X[col]
            feature_names.append(col)

    # Handle missing values with median
    X_processed = X_processed.fillna(X_processed.median())

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    print(f"  Selected {len(CLUSTERING_FEATURES)} features")
    print(f"  Log transformed: {LOG_TRANSFORM_FEATURES}")
    print(f"  Final features: {feature_names}")
    print(f"  Shape: {X_scaled.shape}")

    return X_scaled, scaler, feature_names


# =============================================================================
# CLUSTERING ALGORITHMS
# =============================================================================

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
    """Find optimal K for K-Means using multiple metrics."""
    print("Finding optimal K for K-Means...")

    results = {'k': [], 'inertia': [], 'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': []}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)

        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))

        print(f"  K={k}: Silhouette={results['silhouette'][-1]:.3f}, "
              f"DB={results['davies_bouldin'][-1]:.3f}")

    return pd.DataFrame(results)


def perform_kmeans(X: np.ndarray, n_clusters: int) -> tuple:
    """Perform K-Means clustering."""
    print(f"Training K-Means with K={n_clusters}...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X)

    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"    Cluster {cluster}: {count:,} ({pct:.1f}%)")

    return kmeans, labels


def perform_dbscan(X: np.ndarray, eps: float = None, min_samples: int = None) -> tuple:
    """Perform DBSCAN clustering."""
    # Find optimal eps using k-distance
    if eps is None:
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        k_distances = np.sort(distances[:, 4])
        eps = np.percentile(k_distances, 95)

    if min_samples is None:
        min_samples = max(10, len(X) // 1000)

    print(f"Training DBSCAN (eps={eps:.3f}, min_samples={min_samples})...")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")

    return dbscan, labels


def perform_hierarchical(X: np.ndarray, n_clusters: int) -> tuple:
    """Perform Hierarchical clustering."""
    print(f"Training Hierarchical clustering (n={n_clusters})...")

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(X)

    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"    Cluster {cluster}: {count:,} ({pct:.1f}%)")

    return agg, labels


def perform_gmm(X: np.ndarray, n_components: int) -> tuple:
    """Perform Gaussian Mixture Model clustering."""
    print(f"Training GMM with n_components={n_components}...")

    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5, max_iter=200)
    gmm.fit(X)

    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    max_proba = probabilities.max(axis=1)

    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        avg_proba = max_proba[labels == cluster].mean()
        print(f"    Cluster {cluster}: {count:,} ({pct:.1f}%), confidence: {avg_proba:.2f}")

    return gmm, labels, max_proba


# =============================================================================
# EVALUATION & PROFILING
# =============================================================================

def evaluate_clustering(X: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """Evaluate clustering quality."""
    mask = labels != -1
    if mask.sum() < len(labels):
        X_eval = X[mask]
        labels_eval = labels[mask]
        n_noise = (~mask).sum()
    else:
        X_eval = X
        labels_eval = labels
        n_noise = 0

    n_clusters = len(set(labels_eval))

    if n_clusters < 2:
        return {'name': name, 'n_clusters': n_clusters, 'n_noise': n_noise, 'error': 'Less than 2 clusters'}

    return {
        'name': name,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette_score(X_eval, labels_eval),
        'davies_bouldin': davies_bouldin_score(X_eval, labels_eval),
        'calinski_harabasz': calinski_harabasz_score(X_eval, labels_eval),
    }


def compare_methods(X: np.ndarray, labels_dict: dict) -> pd.DataFrame:
    """Compare multiple clustering methods."""
    print("\nComparing clustering methods...")

    results = []
    for name, labels in labels_dict.items():
        metrics = evaluate_clustering(X, labels, name)
        results.append(metrics)

    comparison = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("CLUSTERING COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))

    return comparison


def profile_clusters(df: pd.DataFrame, cluster_col: str, features: list) -> tuple:
    """Create cluster profiles."""
    print(f"\nProfiling clusters by {cluster_col}...")

    profiles = df.groupby(cluster_col)[features].agg(['mean', 'median', 'std']).round(2)

    cluster_sizes = df[cluster_col].value_counts().sort_index()
    summary = df.groupby(cluster_col)[features].mean().round(2)
    summary['count'] = cluster_sizes
    summary['pct'] = (cluster_sizes / len(df) * 100).round(1)

    print("\nCluster Summary:")
    print(summary.to_string())

    return profiles, summary


def assign_business_labels(summary: pd.DataFrame) -> dict:
    """Assign business labels based on cluster profiles."""
    print("\nAssigning business labels...")

    labels = {}
    n_clusters = len(summary)

    # Normalize for comparison (use original column names)
    summary_norm = summary.copy()
    for col in ['recency', 'frequency', 'monetary', 'avg_review_score']:
        if col in summary_norm.columns:
            min_val = summary_norm[col].min()
            max_val = summary_norm[col].max()
            if max_val > min_val:
                summary_norm[col] = (summary_norm[col] - min_val) / (max_val - min_val)

    for cluster_id in summary.index:
        row = summary_norm.loc[cluster_id]
        recency = row.get('recency', 0.5)
        frequency = row.get('frequency', 0.5)
        monetary = row.get('monetary', 0.5)
        review = row.get('avg_review_score', 0.5)

        if recency < 0.4 and (monetary > 0.5 or frequency > 0.5):
            if review > 0.5:
                labels[cluster_id] = 'champions'
            else:
                labels[cluster_id] = 'needs_attention'
        elif recency > 0.6:
            labels[cluster_id] = 'at_risk'
        elif monetary > 0.6:
            labels[cluster_id] = 'big_spenders'
        else:
            labels[cluster_id] = 'potential_loyalists'

    # Ensure unique labels
    used_labels = set()
    for cluster_id in labels:
        base_label = labels[cluster_id]
        if base_label in used_labels:
            labels[cluster_id] = f"{base_label}_{cluster_id}"
        used_labels.add(labels[cluster_id])

    print("\nAssigned labels:")
    for cluster_id, label in sorted(labels.items()):
        print(f"  Cluster {cluster_id}: {label}")

    return labels


# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

def save_artifacts(
    output_dir: str,
    scaler: StandardScaler,
    model: KMeans,
    profiles: pd.DataFrame,
    summary: pd.DataFrame,
    cluster_labels: dict,
    comparison: pd.DataFrame,
    feature_names: list,
    k_metrics: pd.DataFrame,
) -> None:
    """Save all clustering artifacts."""
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    # Handle GCS paths
    if output_dir.startswith("gs://"):
        # For GCS, save to local temp first, then upload
        import tempfile
        local_dir = tempfile.mkdtemp()
        gcs_dir = output_dir
    else:
        local_dir = output_dir
        gcs_dir = None

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Save scaler
    scaler_path = f"{local_dir}/cluster_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved: cluster_scaler.joblib")

    # Save model
    model_path = f"{local_dir}/clustering_model.joblib"
    joblib.dump(model, model_path)
    print(f"  Saved: clustering_model.joblib")

    # Save profiles
    profiles_path = f"{local_dir}/cluster_profiles.csv"
    summary.to_csv(profiles_path)
    print(f"  Saved: cluster_profiles.csv")

    # Save comparison metrics
    metrics_path = f"{local_dir}/clustering_metrics.csv"
    comparison.to_csv(metrics_path, index=False)
    print(f"  Saved: clustering_metrics.csv")

    # Save K-Means metrics
    k_metrics_path = f"{local_dir}/kmeans_k_metrics.csv"
    k_metrics.to_csv(k_metrics_path, index=False)
    print(f"  Saved: kmeans_k_metrics.csv")

    # Save config
    config = {
        'cluster_labels': {int(k): v for k, v in cluster_labels.items()},
        'feature_names': feature_names,
        'n_clusters': len(cluster_labels),
        'algorithm': 'KMeans',
        'clustering_features': CLUSTERING_FEATURES,
        'log_transform_features': LOG_TRANSFORM_FEATURES,
    }
    config_path = f"{local_dir}/cluster_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: cluster_config.json")

    # Upload to GCS if needed
    if gcs_dir:
        from google.cloud import storage

        # Parse bucket and prefix from gs:// path
        gcs_path = gcs_dir.replace("gs://", "")
        bucket_name = gcs_path.split("/")[0]
        prefix = "/".join(gcs_path.split("/")[1:])

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for filename in os.listdir(local_dir):
            local_file = f"{local_dir}/{filename}"
            blob_name = f"{prefix}/{filename}" if prefix else filename
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file)
            print(f"  Uploaded to GCS: gs://{bucket_name}/{blob_name}")

    print("\nAll artifacts saved!")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_clustering_pipeline(
    input_data: str,
    output_dir: str,
    optimal_k: int = 4,
    k_range: tuple = (2, 9),
) -> dict:
    """
    Run the complete clustering pipeline.

    Args:
        input_data: Path to customer_segments.parquet (local or GCS)
        output_dir: Directory to save artifacts (local or GCS)
        optimal_k: Pre-specified K for K-Means
        k_range: Range of K values to evaluate

    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("VERTEX AI CLUSTERING PIPELINE")
    print("=" * 60)

    # Load data
    customer_df = load_data(input_data)

    # Prepare data
    X_scaled, scaler, feature_names = prepare_clustering_data(customer_df)

    # K-Means analysis
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)
    k_metrics = find_optimal_k(X_scaled, range(*k_range))
    kmeans, labels_kmeans = perform_kmeans(X_scaled, optimal_k)

    # DBSCAN
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTERING")
    print("=" * 60)
    dbscan, labels_dbscan = perform_dbscan(X_scaled)

    # Hierarchical
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 60)
    hier, labels_hier = perform_hierarchical(X_scaled, optimal_k)

    # GMM
    print("\n" + "=" * 60)
    print("GAUSSIAN MIXTURE MODELS")
    print("=" * 60)
    gmm, labels_gmm, gmm_proba = perform_gmm(X_scaled, optimal_k)

    # Compare methods
    labels_dict = {
        'K-Means': labels_kmeans,
        'DBSCAN': labels_dbscan,
        'Hierarchical': labels_hier,
        'GMM': labels_gmm,
    }
    comparison = compare_methods(X_scaled, labels_dict)

    # Profile clusters (using K-Means)
    print("\n" + "=" * 60)
    print("CLUSTER PROFILING")
    print("=" * 60)
    customer_df = customer_df.copy()
    customer_df['cluster_id'] = labels_kmeans

    # Use original feature names for profiling (not log-transformed names)
    profiles, summary = profile_clusters(customer_df, 'cluster_id', CLUSTERING_FEATURES)
    cluster_labels = assign_business_labels(summary)
    customer_df['customer_segment'] = customer_df['cluster_id'].map(cluster_labels)

    # Save artifacts
    save_artifacts(
        output_dir=output_dir,
        scaler=scaler,
        model=kmeans,
        profiles=profiles,
        summary=summary,
        cluster_labels=cluster_labels,
        comparison=comparison,
        feature_names=feature_names,
        k_metrics=k_metrics,
    )

    # Save updated customer data
    customer_output = output_dir.rstrip('/') + '/customer_segments_clustered.parquet'
    if customer_output.startswith("gs://"):
        customer_df.to_parquet(customer_output)
    else:
        customer_df.to_parquet(customer_output, index=False)
    print(f"Saved clustered customer data: {customer_output}")

    # Summary
    print("\n" + "=" * 60)
    print("CLUSTERING COMPLETE")
    print("=" * 60)
    print(f"\nFinal model: K-Means with K={optimal_k}")
    print(f"Silhouette score: {silhouette_score(X_scaled, labels_kmeans):.3f}")
    print("\nCluster segments:")
    for cluster_id, label in sorted(cluster_labels.items()):
        count = (customer_df['cluster_id'] == cluster_id).sum()
        pct = count / len(customer_df) * 100
        print(f"  {cluster_id} ({label}): {count:,} customers ({pct:.1f}%)")

    return {
        'customer_df': customer_df,
        'kmeans': kmeans,
        'scaler': scaler,
        'cluster_labels': cluster_labels,
        'comparison': comparison,
    }


def main():
    """Main entry point for Vertex AI training."""
    parser = argparse.ArgumentParser(description="Clustering Training Script for Vertex AI")
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to customer_segments.parquet (local or gs://)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save artifacts (local or gs://)",
    )
    parser.add_argument(
        "--optimal-k",
        type=int,
        default=4,
        help="Number of clusters for K-Means (default: 4)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="GCS bucket name (for Vertex AI environment)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("VERTEX AI TRAINING JOB STARTED")
    print("=" * 60)
    print(f"Input data: {args.input_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Optimal K: {args.optimal_k}")
    print(f"Bucket: {args.bucket}")
    print("=" * 60)

    # Run the pipeline
    run_clustering_pipeline(
        input_data=args.input_data,
        output_dir=args.output_dir,
        optimal_k=args.optimal_k,
    )

    print("\n" + "=" * 60)
    print("VERTEX AI TRAINING JOB COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
