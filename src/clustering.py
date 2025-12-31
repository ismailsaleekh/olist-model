"""
Clustering utilities for customer segmentation.

This module handles:
1. Data preparation for clustering (log transform, scaling)
2. K-Means clustering with optimal K selection
3. DBSCAN clustering with optimal eps selection
4. Hierarchical clustering with dendrogram
5. Gaussian Mixture Models with BIC/AIC selection
6. Clustering comparison and evaluation
7. Cluster profiling and business labeling

Day 2, Part 2 Implementation
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONSTANTS
# =============================================================================

# Features to use for clustering (selected based on business relevance and variance)
CLUSTERING_FEATURES = [
    'recency',            # Days since last order (core RFM)
    'frequency',          # Number of orders (core RFM) - will be log transformed
    'monetary',           # Total spend (core RFM) - will be log transformed
    'avg_review_score',   # Customer satisfaction signal
    'avg_delivery_days',  # Delivery experience
    'late_delivery_rate', # Service quality metric
]

# Features that need log transformation due to high skewness
LOG_TRANSFORM_FEATURES = ['frequency', 'monetary']

# Default business labels for 4 clusters (will be adjusted based on profiling)
DEFAULT_CLUSTER_LABELS = {
    0: 'champions',
    1: 'at_risk',
    2: 'potential_loyalists',
    3: 'needs_attention',
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_clustering_data(
    df: pd.DataFrame,
    features: List[str] = None,
    log_features: List[str] = None,
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Prepare customer data for clustering.

    Steps:
    1. Select clustering features
    2. Log transform skewed features
    3. Handle missing values
    4. StandardScaler normalization

    Args:
        df: Customer-level DataFrame
        features: List of features to use (default: CLUSTERING_FEATURES)
        log_features: Features to log transform (default: LOG_TRANSFORM_FEATURES)

    Returns:
        Tuple of (X_scaled, fitted_scaler, feature_names)
    """
    if features is None:
        features = CLUSTERING_FEATURES
    if log_features is None:
        log_features = LOG_TRANSFORM_FEATURES

    print("📊 Preparing clustering data")

    # Select features
    X = df[features].copy()
    feature_names = []

    # Log transform skewed features
    for col in features:
        if col in log_features:
            X[f'{col}_log'] = np.log1p(X[col])
            feature_names.append(f'{col}_log')
        else:
            feature_names.append(col)

    # Drop original skewed columns, keep log versions
    cols_to_use = [c for c in feature_names]
    X_final = X[[c for c in X.columns if c in cols_to_use or c not in log_features]]

    # Reorder to match feature_names
    X_final = X_final[[c if c in X_final.columns else c.replace('_log', '') for c in feature_names]]

    # Actually build the final dataframe correctly
    X_processed = pd.DataFrame()
    for col in features:
        if col in log_features:
            X_processed[f'{col}_log'] = np.log1p(X[col])
        else:
            X_processed[col] = X[col]

    feature_names = list(X_processed.columns)

    # Handle missing values with median
    X_processed = X_processed.fillna(X_processed.median())

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    print(f"  ✓ Selected {len(features)} features")
    print(f"  ✓ Log transformed: {log_features}")
    print(f"  ✓ Final features: {feature_names}")
    print(f"  ✓ Shape: {X_scaled.shape}")

    return X_scaled, scaler, feature_names


# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Find optimal number of clusters using multiple methods.

    Methods:
    - Elbow method (inertia)
    - Silhouette score
    - Davies-Bouldin index
    - Calinski-Harabasz index

    Args:
        X: Scaled feature matrix
        k_range: Range of K values to try
        random_state: Random seed

    Returns:
        DataFrame with metrics for each K
    """
    print("📊 Finding optimal K for K-Means")

    results = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
    }

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)

        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))

        print(f"  K={k}: Silhouette={results['silhouette'][-1]:.3f}, "
              f"DB={results['davies_bouldin'][-1]:.3f}, "
              f"CH={results['calinski_harabasz'][-1]:.0f}")

    return pd.DataFrame(results)


def perform_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """
    Perform K-Means clustering with specified K.

    Args:
        X: Scaled feature matrix
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Tuple of (fitted KMeans model, cluster labels)
    """
    print(f"📊 Training K-Means with K={n_clusters}")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"    Cluster {cluster}: {count:,} ({pct:.1f}%)")

    return kmeans, labels


# =============================================================================
# DBSCAN CLUSTERING
# =============================================================================

def find_optimal_eps(
    X: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Find optimal eps using k-nearest neighbors distance.

    Look for the "knee" in the k-distance graph.

    Args:
        X: Scaled feature matrix
        k: Number of neighbors to consider

    Returns:
        Sorted k-distances array
    """
    print(f"📊 Computing {k}-distance graph for DBSCAN eps selection")

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)

    # Sort k-th nearest neighbor distances
    k_distances = np.sort(distances[:, k-1])

    # Suggest eps based on percentiles
    p90 = np.percentile(k_distances, 90)
    p95 = np.percentile(k_distances, 95)
    p99 = np.percentile(k_distances, 99)

    print(f"  Distance percentiles:")
    print(f"    90th: {p90:.3f}")
    print(f"    95th: {p95:.3f}")
    print(f"    99th: {p99:.3f}")
    print(f"  Suggested eps range: {p90:.2f} - {p95:.2f}")

    return k_distances


def perform_dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int = 10,
) -> Tuple[DBSCAN, np.ndarray]:
    """
    Perform DBSCAN clustering.

    Args:
        X: Scaled feature matrix
        eps: Maximum distance between samples
        min_samples: Minimum samples in a neighborhood

    Returns:
        Tuple of (fitted DBSCAN model, cluster labels)
    """
    print(f"📊 Training DBSCAN (eps={eps}, min_samples={min_samples})")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X)

    # Count clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = n_noise / len(labels) * 100

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise:,} ({noise_pct:.1f}%)")

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        label = "Noise" if cluster == -1 else f"Cluster {cluster}"
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    return dbscan, labels


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def compute_linkage_matrix(
    X: np.ndarray,
    sample_size: int = 2000,
    method: str = 'ward',
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute linkage matrix for dendrogram visualization.

    Uses sampling for large datasets.

    Args:
        X: Scaled feature matrix
        sample_size: Number of samples for dendrogram
        method: Linkage method ('ward', 'complete', 'average')
        random_state: Random seed

    Returns:
        Tuple of (linkage_matrix, sample_indices)
    """
    print(f"📊 Computing linkage matrix (method={method})")

    # Sample for visualization if dataset is large
    if len(X) > sample_size:
        np.random.seed(random_state)
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]
        print(f"  Using {sample_size} samples for dendrogram")
    else:
        sample_idx = np.arange(len(X))
        X_sample = X

    # Compute linkage
    linkage_matrix = linkage(X_sample, method=method)

    print(f"  ✓ Linkage matrix computed")

    return linkage_matrix, sample_idx


def perform_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage_method: str = 'ward',
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Perform Agglomerative Hierarchical clustering.

    Args:
        X: Scaled feature matrix
        n_clusters: Number of clusters
        linkage_method: Linkage method

    Returns:
        Tuple of (fitted model, cluster labels)
    """
    print(f"📊 Training Hierarchical clustering (n={n_clusters}, linkage={linkage_method})")

    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
    )
    labels = agg.fit_predict(X)

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"    Cluster {cluster}: {count:,} ({pct:.1f}%)")

    return agg, labels


# =============================================================================
# GAUSSIAN MIXTURE MODELS
# =============================================================================

def find_optimal_gmm_components(
    X: np.ndarray,
    n_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Find optimal number of GMM components using BIC/AIC.

    Args:
        X: Scaled feature matrix
        n_range: Range of component counts to try
        random_state: Random seed

    Returns:
        DataFrame with BIC/AIC for each n
    """
    print("📊 Finding optimal GMM components")

    results = {'n': [], 'bic': [], 'aic': []}

    for n in n_range:
        gmm = GaussianMixture(
            n_components=n,
            random_state=random_state,
            n_init=5,
            max_iter=200,
        )
        gmm.fit(X)

        results['n'].append(n)
        results['bic'].append(gmm.bic(X))
        results['aic'].append(gmm.aic(X))

        print(f"  n={n}: BIC={results['bic'][-1]:.0f}, AIC={results['aic'][-1]:.0f}")

    return pd.DataFrame(results)


def perform_gmm(
    X: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    Perform Gaussian Mixture Model clustering.

    Args:
        X: Scaled feature matrix
        n_components: Number of mixture components
        random_state: Random seed

    Returns:
        Tuple of (fitted GMM model, cluster labels, cluster probabilities)
    """
    print(f"📊 Training GMM with n_components={n_components}")

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=5,
        max_iter=200,
    )
    gmm.fit(X)

    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    max_proba = probabilities.max(axis=1)

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print("  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        avg_proba = max_proba[labels == cluster].mean()
        print(f"    Cluster {cluster}: {count:,} ({pct:.1f}%), avg confidence: {avg_proba:.2f}")

    return gmm, labels, max_proba


# =============================================================================
# CLUSTERING EVALUATION
# =============================================================================

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    name: str,
) -> Dict:
    """
    Evaluate clustering quality with multiple metrics.

    Args:
        X: Scaled feature matrix
        labels: Cluster labels
        name: Name of the clustering method

    Returns:
        Dictionary with evaluation metrics
    """
    # Filter out noise points for DBSCAN
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
        return {
            'name': name,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
            'error': 'Less than 2 clusters',
        }

    return {
        'name': name,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette_score(X_eval, labels_eval),
        'davies_bouldin': davies_bouldin_score(X_eval, labels_eval),
        'calinski_harabasz': calinski_harabasz_score(X_eval, labels_eval),
    }


def compare_clustering_methods(
    X: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare multiple clustering methods.

    Args:
        X: Scaled feature matrix
        labels_dict: Dictionary mapping method names to labels

    Returns:
        DataFrame with comparison metrics
    """
    print("📊 Comparing clustering methods")

    results = []
    for name, labels in labels_dict.items():
        metrics = evaluate_clustering(X, labels, name)
        results.append(metrics)

    comparison_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("CLUSTERING COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    return comparison_df


# =============================================================================
# CLUSTER PROFILING
# =============================================================================

def profile_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    features: List[str],
) -> pd.DataFrame:
    """
    Create detailed cluster profiles.

    Args:
        df: DataFrame with cluster assignments
        cluster_col: Column name containing cluster labels
        features: Features to profile

    Returns:
        DataFrame with cluster profiles
    """
    print(f"📊 Profiling clusters by {cluster_col}")

    # Calculate statistics per cluster
    profiles = df.groupby(cluster_col)[features].agg(['mean', 'median', 'std']).round(2)

    # Add cluster sizes
    cluster_sizes = df[cluster_col].value_counts().sort_index()

    # Create summary DataFrame
    summary = df.groupby(cluster_col)[features].mean().round(2)
    summary['count'] = cluster_sizes
    summary['pct'] = (cluster_sizes / len(df) * 100).round(1)

    print("\nCluster Summary:")
    print(summary.to_string())

    return profiles, summary


def assign_business_labels(
    summary: pd.DataFrame,
    feature_weights: Dict[str, str] = None,
) -> Dict[int, str]:
    """
    Assign business labels based on cluster profiles.

    This uses heuristics based on RFM characteristics.

    Args:
        summary: Cluster summary DataFrame
        feature_weights: Optional custom feature interpretations

    Returns:
        Dictionary mapping cluster IDs to business labels
    """
    print("📊 Assigning business labels")

    labels = {}
    n_clusters = len(summary)

    # Normalize features for comparison
    summary_norm = summary.copy()
    for col in ['recency', 'frequency_log', 'monetary_log', 'avg_review_score']:
        if col in summary_norm.columns:
            min_val = summary_norm[col].min()
            max_val = summary_norm[col].max()
            if max_val > min_val:
                summary_norm[col] = (summary_norm[col] - min_val) / (max_val - min_val)

    for cluster_id in summary.index:
        row = summary_norm.loc[cluster_id]

        # Get recency (lower is better - more recent)
        recency = row.get('recency', 0.5)
        # Get frequency (higher is better)
        frequency = row.get('frequency_log', 0.5)
        # Get monetary (higher is better)
        monetary = row.get('monetary_log', 0.5)
        # Get review score (higher is better)
        review = row.get('avg_review_score', 0.5)

        # Scoring logic
        # Champions: recent, high frequency, high monetary
        # At-risk: not recent, any frequency/monetary
        # Potential loyalists: recent, medium everything
        # Needs attention: recent but low satisfaction

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

    # Ensure unique labels by adding suffix if needed
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
# VISUALIZATION
# =============================================================================

def plot_elbow_silhouette(
    k_metrics: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot elbow and silhouette analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Elbow plot
    axes[0].plot(k_metrics['k'], k_metrics['inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(k_metrics['k'], k_metrics['silhouette'], 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Davies-Bouldin plot
    axes[2].plot(k_metrics['k'], k_metrics['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[2].set_title('Davies-Bouldin (Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved plot to {save_path}")

    return fig


def plot_k_distance(
    k_distances: np.ndarray,
    suggested_eps: float = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot k-distance graph for DBSCAN eps selection."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(k_distances, 'b-', linewidth=1)
    ax.set_xlabel('Points (sorted by distance)', fontsize=12)
    ax.set_ylabel('K-th Nearest Neighbor Distance', fontsize=12)
    ax.set_title('K-Distance Graph (look for elbow/knee)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if suggested_eps:
        ax.axhline(y=suggested_eps, color='r', linestyle='--',
                   label=f'Suggested eps={suggested_eps:.2f}')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(15, 7))

    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=30,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10,
    )

    ax.set_xlabel('Sample Index (or cluster size)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_gmm_bic_aic(
    gmm_metrics: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot BIC/AIC for GMM component selection."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(gmm_metrics['n'], gmm_metrics['bic'], 'bo-', linewidth=2, markersize=8, label='BIC')
    ax.plot(gmm_metrics['n'], gmm_metrics['aic'], 'go-', linewidth=2, markersize=8, label='AIC')

    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('GMM Model Selection (Lower is Better)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_clusters_pca(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = 'Cluster Visualization (PCA)',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize clusters in 2D using PCA."""
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        label_name = f'Cluster {label}' if label != -1 else 'Noise'
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[color], label=label_name,
            alpha=0.6, s=20,
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cluster_profiles_radar(
    summary: pd.DataFrame,
    features: List[str],
    cluster_labels: Dict[int, str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create radar chart for cluster profiles."""
    from math import pi

    # Normalize features for radar chart
    summary_norm = summary[features].copy()
    for col in features:
        min_val = summary_norm[col].min()
        max_val = summary_norm[col].max()
        if max_val > min_val:
            summary_norm[col] = (summary_norm[col] - min_val) / (max_val - min_val)
        else:
            summary_norm[col] = 0.5

    # Number of features
    n_features = len(features)
    angles = [n / float(n_features) * 2 * pi for n in range(n_features)]
    angles += angles[:1]  # Complete the loop

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_norm)))

    for idx, (cluster_id, row) in enumerate(summary_norm.iterrows()):
        values = row.tolist()
        values += values[:1]  # Complete the loop

        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}') if cluster_labels else f'Cluster {cluster_id}'

        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Set feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    ax.set_title('Cluster Profiles (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cluster_distributions(
    df: pd.DataFrame,
    cluster_col: str,
    features: List[str],
    cluster_labels: Dict[int, str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot feature distributions by cluster."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        for cluster_id in sorted(df[cluster_col].unique()):
            if cluster_id == -1:
                continue
            data = df[df[cluster_col] == cluster_id][feature]
            label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}') if cluster_labels else f'Cluster {cluster_id}'
            ax.hist(data, bins=30, alpha=0.5, label=label, density=True)

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{feature} by Cluster', fontsize=12)
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions by Cluster', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_clustering_artifacts(
    scaler: StandardScaler,
    model: KMeans,
    profiles: pd.DataFrame,
    summary: pd.DataFrame,
    cluster_labels: Dict[int, str],
    metrics: pd.DataFrame,
    feature_names: List[str],
    models_dir: str = "models",
) -> None:
    """
    Save all clustering artifacts.

    Args:
        scaler: Fitted StandardScaler
        model: Fitted clustering model (K-Means)
        profiles: Cluster profiles DataFrame
        summary: Cluster summary DataFrame
        cluster_labels: Business labels mapping
        metrics: Comparison metrics DataFrame
        feature_names: List of feature names used
        models_dir: Output directory
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING CLUSTERING ARTIFACTS")
    print("=" * 60)

    # Save scaler
    joblib.dump(scaler, models_dir / "cluster_scaler.joblib")
    print(f"✓ Saved cluster_scaler.joblib")

    # Save model
    joblib.dump(model, models_dir / "clustering_model.joblib")
    print(f"✓ Saved clustering_model.joblib")

    # Save profiles
    summary.to_csv(models_dir / "cluster_profiles.csv")
    print(f"✓ Saved cluster_profiles.csv")

    # Save metrics
    metrics.to_csv(models_dir / "clustering_metrics.csv", index=False)
    print(f"✓ Saved clustering_metrics.csv")

    # Save labels mapping
    cluster_config = {
        'cluster_labels': cluster_labels,
        'feature_names': feature_names,
        'n_clusters': len(cluster_labels),
        'algorithm': 'KMeans',
    }
    with open(models_dir / "cluster_config.json", 'w') as f:
        json.dump(cluster_config, f, indent=2)
    print(f"✓ Saved cluster_config.json")

    print("\n✅ All clustering artifacts saved!")


# =============================================================================
# MAIN CLUSTERING PIPELINE
# =============================================================================

def run_clustering_pipeline(
    customer_df: pd.DataFrame,
    k_range: range = range(2, 9),
    optimal_k: int = None,
    random_state: int = 42,
) -> Dict:
    """
    Run the complete clustering pipeline.

    Args:
        customer_df: Customer-level DataFrame
        k_range: Range of K values to evaluate
        optimal_k: Pre-specified K (if None, will be determined)
        random_state: Random seed

    Returns:
        Dictionary with all results and artifacts
    """
    print("=" * 60)
    print("CLUSTERING PIPELINE")
    print("=" * 60)
    print(f"Customers: {len(customer_df):,}")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Prepare Data
    # =========================================================================
    X_scaled, scaler, feature_names = prepare_clustering_data(customer_df)

    # =========================================================================
    # STEP 2: K-Means Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)

    k_metrics = find_optimal_k(X_scaled, k_range, random_state)

    # Determine optimal K if not specified
    if optimal_k is None:
        # Use silhouette score to suggest K
        best_k_idx = k_metrics['silhouette'].idxmax()
        optimal_k = k_metrics.loc[best_k_idx, 'k']
        print(f"\n  Suggested optimal K: {optimal_k} (based on silhouette)")

    # Train K-Means with optimal K
    kmeans, labels_kmeans = perform_kmeans(X_scaled, optimal_k, random_state)

    # =========================================================================
    # STEP 3: DBSCAN Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTERING")
    print("=" * 60)

    k_distances = find_optimal_eps(X_scaled, k=5)

    # Use 95th percentile as eps
    eps = np.percentile(k_distances, 95)
    min_samples = max(10, len(X_scaled) // 1000)  # At least 10, or 0.1% of data

    dbscan, labels_dbscan = perform_dbscan(X_scaled, eps, min_samples)

    # =========================================================================
    # STEP 4: Hierarchical Clustering
    # =========================================================================
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 60)

    linkage_matrix, _ = compute_linkage_matrix(X_scaled)
    hier, labels_hier = perform_hierarchical(X_scaled, optimal_k)

    # =========================================================================
    # STEP 5: GMM Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("GAUSSIAN MIXTURE MODELS")
    print("=" * 60)

    gmm_metrics = find_optimal_gmm_components(X_scaled, k_range, random_state)
    gmm, labels_gmm, gmm_proba = perform_gmm(X_scaled, optimal_k, random_state)

    # =========================================================================
    # STEP 6: Compare All Methods
    # =========================================================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    labels_dict = {
        'K-Means': labels_kmeans,
        'DBSCAN': labels_dbscan,
        'Hierarchical': labels_hier,
        'GMM': labels_gmm,
    }

    comparison = compare_clustering_methods(X_scaled, labels_dict)

    # =========================================================================
    # STEP 7: Profile Clusters (using K-Means as primary)
    # =========================================================================
    print("\n" + "=" * 60)
    print("CLUSTER PROFILING")
    print("=" * 60)

    # Add cluster labels to customer dataframe
    customer_df = customer_df.copy()
    customer_df['cluster_id'] = labels_kmeans

    # Profile clusters
    profile_features = [f for f in feature_names]
    profiles, summary = profile_clusters(customer_df, 'cluster_id', profile_features)

    # Assign business labels
    cluster_labels = assign_business_labels(summary)
    customer_df['customer_segment'] = customer_df['cluster_id'].map(cluster_labels)

    # =========================================================================
    # Summary
    # =========================================================================
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
        'X_scaled': X_scaled,
        'scaler': scaler,
        'feature_names': feature_names,
        'k_metrics': k_metrics,
        'kmeans': kmeans,
        'labels_kmeans': labels_kmeans,
        'dbscan': dbscan,
        'labels_dbscan': labels_dbscan,
        'k_distances': k_distances,
        'linkage_matrix': linkage_matrix,
        'labels_hier': labels_hier,
        'gmm_metrics': gmm_metrics,
        'gmm': gmm,
        'labels_gmm': labels_gmm,
        'gmm_proba': gmm_proba,
        'comparison': comparison,
        'profiles': profiles,
        'summary': summary,
        'cluster_labels': cluster_labels,
        'optimal_k': optimal_k,
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def add_clusters_to_dataset(
    df: pd.DataFrame,
    customer_df: pd.DataFrame,
    cluster_col: str = 'cluster_id',
    segment_col: str = 'customer_segment',
) -> pd.DataFrame:
    """
    Merge cluster assignments to order-level dataset.

    Args:
        df: Order-level DataFrame
        customer_df: Customer DataFrame with cluster assignments
        cluster_col: Column name for cluster ID
        segment_col: Column name for business segment

    Returns:
        DataFrame with cluster columns added
    """
    merge_cols = ['customer_unique_id', cluster_col, segment_col]

    df = df.merge(
        customer_df[merge_cols],
        on='customer_unique_id',
        how='left',
    )

    # Handle customers not in training data
    df[cluster_col] = df[cluster_col].fillna(-1).astype(int)
    df[segment_col] = df[segment_col].fillna('new_customer')

    return df


def load_clustering_artifacts(models_dir: str = "models") -> Dict:
    """Load saved clustering artifacts."""
    models_dir = Path(models_dir)

    artifacts = {
        'scaler': joblib.load(models_dir / "cluster_scaler.joblib"),
        'model': joblib.load(models_dir / "clustering_model.joblib"),
        'profiles': pd.read_csv(models_dir / "cluster_profiles.csv", index_col=0),
        'metrics': pd.read_csv(models_dir / "clustering_metrics.csv"),
    }

    with open(models_dir / "cluster_config.json") as f:
        artifacts['config'] = json.load(f)

    print("✓ Loaded clustering artifacts")

    return artifacts
