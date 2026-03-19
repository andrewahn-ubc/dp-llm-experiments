import sys, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer

# example usage:
# "python dedup_similarity.py input.csv"

inp = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "filtered.csv"
removed = sys.argv[3] if len(sys.argv) > 3 else "removed.csv"

df = pd.read_csv(inp)
s = df["goal"].astype(str).str.strip()
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
E = m.encode(s.tolist(), normalize_embeddings=True)
S = E @ E.T
np.fill_diagonal(S, -1)
k = min(5, len(df) - 1)
scores = np.mean(np.partition(S, -k, axis=1)[:, -k:], axis=1)
drop = np.argsort(scores)[-1460:]

df.iloc[np.setdiff1d(np.arange(len(df)), drop)].to_csv(out, index=False)
df.iloc[drop].assign(redundancy_score=scores[drop]).sort_values("redundancy_score", ascending=False).to_csv(removed, index=False)

print(df.iloc[drop][["goal"]].assign(redundancy_score=scores[drop]).sort_values("redundancy_score", ascending=False).to_string(index=False))