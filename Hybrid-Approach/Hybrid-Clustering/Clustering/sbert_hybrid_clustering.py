import pandas as pd
# import numpy as np
# clustering
from sklearn.preprocessing import normalize
import hdbscan

# Load data
PARQUET_FILE = "sBERT_hybrid_embeddings.parquet"
# CHANGE FILE PATH HERE
encodings = pd.read_parquet("/scratch/alpine/anle7858/Hybrid_Approach/" + PARQUET_FILE)

# Select all embedding columns (assuming they are named v0, v1, ..., v767)
embedding_cols = [f"v{i}" for i in range(768)]
X = encodings[embedding_cols].to_numpy()
print(f"The shape of X (num_chunks, embedding_dim) is: {(X.shape)}") # (num_chunks=13095, embedding_dim=768)

# normalize embeddings to unit length
X_norm = normalize(X, norm='l2')

# HDBSCAN Parameters
MIN_CLUSTER_SIZE = 2          # default=5, smallest allowable cluster
MIN_SAMPLES = 4               # default=None, how conservative clustering is
CLUSTER_SELECTION_EPS = 0.0001  # default=0.0
                                  # flexibility parameter: higher number = the more allowance for different density clusters to be put together
                                  # ensures that clusters below the given threshold are not split up any further

# Define hdbscan clusterer
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    cluster_selection_epsilon=CLUSTER_SELECTION_EPS,
    metric='euclidean',
    gen_min_span_tree=True
)

clusterer.fit(X_norm)

# Get cluster IDs and membership probabilities for each chunk
labels = clusterer.labels_    # array of cluster IDs, -1 means noise
membership_probs = clusterer.probabilities_  # array of membership probabilities

# Append cluster IDs to parquet file
encodings["cluster_id"] = labels
encodings["membership_prob"] = membership_probs

# Save cluster_id and membership_probs to original csv
FILE_NAME="sbert_hybrid_with_cluster_id.csv"
# CHANGE FILE PATH HERE
encodings.to_csv("/scratch/alpine/anle7858/Hybrid_Approach/" + FILE_NAME, index=False)
print("Saved cluster IDs to sbert_hybrid_with_cluster_id.csv")
