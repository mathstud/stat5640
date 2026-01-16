import pandas as pd
import numpy as np
# clustering
from sklearn.preprocessing import normalize
import hdbscan
# parameter tuning
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score


# Load data
PARQUET_FILE = "sBERT_hybrid_embeddings.parquet"

# check embeddings file locally 
# encodings = pd.read_parquet("/Users/anya/Documents/STAT_5640_Capstone/Project_Data/Hybrid_Approach/" + PARQUET_FILE)
# print(encodings.head())

# CHANGE FILE PATH HERE
encodings = pd.read_parquet("/scratch/alpine/anle7858/Hybrid_Approach/" + PARQUET_FILE)

# Stack a list of 1D arrays into a 2D matrix for clustering
# X = np.vstack(encodings["embedding"].values)
# print(X.shape) # (num_chunks=29298, embedding_dim=768)

# Select all embedding columns (assuming they are named v0, v1, ..., v767)
embedding_cols = [f"v{i}" for i in range(768)]
X = encodings[embedding_cols].to_numpy()
print(f"The shape of X (num_chunks, embedding_dim) is: {(X.shape)}") # (num_chunks=13095, embedding_dim=768)

# normalize embeddings to unit length
X_norm = normalize(X, norm='l2')

# Define parameter grid
param_grid = {
    "MIN_CLUSTER_SIZE": [2, 4], 
    "MIN_SAMPLES": [1, 2, 3, 4],   # range from 1 to log(n) = log(13000)
    "CLUSTER_SELECTION_EPS": [0.0001, 0.001, 0.01, 0.1]
}

param_combinations = list(ParameterGrid(param_grid))


def evaluate_hdbscan(X, MIN_CLUSTER_SIZE, MIN_SAMPLES, EPS):
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            cluster_selection_epsilon=EPS,
            metric='euclidean',
            gen_min_span_tree=True
        ).fit(X)

        labels = clusterer.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = np.sum(labels == -1) / len(labels)
        # measure stability (both between 0 and 1)
        # relative_validity = clusterer.relative_validity_ # global stability - compare param combos
        # cluster_persistence = clusterer.cluster_persistence_ # cluster-level stability
        # mean_cluster_persistence = np.mean(cluster_persistence)

        silhouette = None
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)

        return {
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_samples": MIN_SAMPLES,
            "eps": EPS,
            "n_clusters": n_clusters,
            "noise_frac": round(noise_frac, 3),
            # "relative_validity": round(relative_validity, 3),
            # "mean_cluster_persistence": round(mean_cluster_persistence, 3),
            "silhouette": round(silhouette, 3) if silhouette else None
        }
    except Exception as e:
        return {
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_samples": MIN_SAMPLES,
            "eps": EPS,
            "n_clusters": None,
            "noise_frac": None,
            # "relative_validity": None,
            # "mean_cluster_persistence": None,
            "silhouette": None,
            "error": str(e)
        }


results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs = -1 (use all available CPU cores)
    delayed(evaluate_hdbscan)(
        X_norm,
        params['MIN_CLUSTER_SIZE'],
        params['MIN_SAMPLES'],
        params['CLUSTER_SELECTION_EPS']
    )
    for params in param_combinations
)

# convert to DF and view
tuning_results_df = pd.DataFrame(results)

SAVE_FILENAME = "sbert_hybrid_tuning_results.csv"
# CHANGE FILE PATH HERE
tuning_results_df.to_csv("/scratch/alpine/anle7858/Hybrid_Approach/" + SAVE_FILENAME, index=False)

print("Parallel search completed.")

# Next steps:
# View outputted csv
# choose best parameters based on metrics
# run HDBSCAN with best parameters (securebert_hybrid_clustering.py)
