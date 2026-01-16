import pandas as pd
import numpy as np

# Load chunk-level data
# Load data
PARQUET_FILE = "securebert_chunks_with_cluster_id.parquet"
df = pd.read_parquet("/scratch/alpine/anle7858/Direct_Approach/" + PARQUET_FILE)

def weighted_embed(group):
    vecs = np.stack(group["embedding"].to_numpy()) # [n_chunks, dim]
    w = (group["end_token"] - group["start_token"]).to_numpy(dtype=float)
    # guard against zeros / negatives
    w = np.clip(w, 1.0, None)
    w = w / w.sum()
    
    return (vecs * w[:, None]).sum(axis=0) # [dim]

# Collapse: one embedding per article (weighted)
article_df = (
    df.groupby("article_id")
    .apply(weighted_embed)
    .reset_index(name="embedding")
)

# Bring back representative metadata (pick what you prefer)
meta = (
    df.sort_values("date") # earliest date/title/source seen
    .groupby("article_id")
    .agg(title=("title","first"),
         source=("source","first"),
         date=("date","first"))
         .reset_index()
)

article_df = article_df.merge(meta, on="article_id", how="left")
article_df.head()

SAVE_FILENAME = "securebert_direct_embeddings.parquet"
article_df.to_parquet("/scratch/alpine/anle7858/Direct_Approach/" + SAVE_FILENAME, index=False)
