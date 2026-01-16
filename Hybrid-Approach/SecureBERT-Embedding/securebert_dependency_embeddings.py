import re
import pandas as pd
# %pip install torch
import torch
import torch.nn.functional as F
from torch import Tensor
import transformers
from transformers import RobertaTokenizerFast, RobertaModel
import numpy as np


# Initalize model
model_name = "ehsanaghaei/SecureBERT"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model= RobertaModel.from_pretrained(model_name, add_pooling_layer=False).eval()


df = pd.read_csv("/scratch/alpine/anle7858/article_event_templates.csv", encoding="latin1")
# print(df.head())


# split into lists (strip optional braces and surrounding spaces)
df["chunks"] = df["event_text"].str.strip("{}").str.split(r"\s*;\s*", regex=True)

# explode and add within-row ids (based on the original row index)
out = (
    df[["article_id", "chunks"]]
      .explode("chunks")
      .rename(columns={"chunks": "split_text"})
      .dropna()
      .assign(within_row_id=lambda d: d.groupby(level=0).cumcount())  # 0,1,2,...
      .reset_index(names="orig_row")  # original row number
)

# print(out.head())

#Creating a streamlined data frame
df = out
df = df.rename(columns={"within_row_id": "triple_id"})
df = df.drop(columns=["orig_row"])

# print(df.head())


### Loading in SecureBERT and then applying embedding to every text ###
# Input what text you want to encode.
def SecureBERT_embed(article_text):
    batch = tokenizer(article_text, return_tensors="pt", padding=False, truncation=False)
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)

    # out.hidden_states is a tuple: [embeddings, layer1, ..., layerN]
    hidden_states = out.hidden_states                # len = N_layers + 1
    last4 = hidden_states[-4:]                       # take last 4 layers
    # Average across the 4 layers -> [1, T, H]
    token_reps = torch.stack(last4, dim=0).mean(dim=0)

    # Mean-pool across tokens with attention mask
    attn = batch["attention_mask"].unsqueeze(-1).to(token_reps.dtype)  # [1, T, 1]
    masked = token_reps * attn                                         # zero out pads (if any)
    sent_emb = masked.sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)     # [1, H]
          # No normalization

    return sent_emb



df["triple_embeddings"] = df["split_text"].apply(SecureBERT_embed)
print(df.head())

#safeguard if the average doesn't work
article_securebert_embeddings_no_average_df =  df
article_securebert_embeddings_no_average_df.to_csv("/scratch/alpine/anle7858/Hybrid_Approach/SecureBERT_dependency_embeddings_no_average.csv", index=False)
print("Saved SecureBERT direct embeddings (no average) to SecureBERT_dependency_embeddings_no_average.csv")


# Average over each article to get one embedding.
print(df.head())


def parse_embedding(x):
    # If it's already a tensor -> convert to numpy
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    # If it's a numpy array -> return as is
    if isinstance(x, np.ndarray):
        return x

    # If it's a string -> attempt parsing
    if isinstance(x, str):
        # Remove the 'tensor([...])' wrapper
        cleaned = x.strip().replace("tensor(", "").rstrip(")")
        # Convert string of numbers -> numpy
        return np.array(eval(cleaned), dtype=float)

    raise TypeError(f"Unknown embedding type: {type(x)}")


def mean_vec(series):
    arrs = [np.asarray(x, dtype=float) for x in series]
    return np.mean(np.vstack(arrs), axis=0)

df["triple_embeddings"] = df["triple_embeddings"].apply(parse_embedding)



article_securebert_embeddings_df = (df
            # .groupby("article_id")["split_articles"]
            .groupby("article_id")["triple_embeddings"]
            .apply(mean_vec)
            .reset_index(name="article_securebert_mean"))

#check that it is the length of how many articles we have
print(article_securebert_embeddings_df.head())
print(f"Length of article_securebert_embeddings_df: {len(article_securebert_embeddings_df)}")

#save to csv
article_securebert_embeddings_df.to_csv("/scratch/alpine/anle7858/Hybrid_Approach/SecureBERT_dependency_embeddings.csv", index=False)
print("Saved SecureBERT dependency embeddings to SecureBERT_dependency_embeddings.csv")
