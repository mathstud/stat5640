import pandas as pd
import numpy as np
# clustering
from sklearn.preprocessing import normalize
import hdbscan

# Load data
FILE_NAME = "SecureBERT_dependency_embeddings.csv"
# CHANGE FILE PATH HERE
encodings = pd.read_csv("/scratch/alpine/anle7858/Hybrid_Approach/" + FILE_NAME)

# Stack a list of 1D arrays into a 2D matrix for clustering
X = np.vstack(encodings["article_securebert_mean"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ") # convert embedding from str to numpy
))
print(f"The shape of X (num_chunks, embedding_dim) is: {(X.shape)}") # (num_chunks=13075, embedding_dim=768)

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
FILE_NAME="SecureBERT_hybrid_with_cluster_id.csv"
# CHANGE FILE PATH HERE
encodings.to_csv("/scratch/alpine/anle7858/Hybrid_Approach/" + FILE_NAME, index=False)
print("Saved cluster IDs to SecureBERT_hybrid_with_cluster_id.csv")
