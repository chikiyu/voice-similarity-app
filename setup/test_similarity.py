"""
Sanity check: print top-3 most similar artists for each artist.
Good result: same-genre artists cluster together.
  e.g. Adele → Amy Winehouse, Whitney Houston (not Eminem)
"""
import numpy as np
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")

with open(os.path.join(ROOT, "artists_metadata.json"), encoding="utf-8") as f:
    artists = json.load(f)

data = np.load(os.path.join(ROOT, "embeddings.npz"))
emb = data["embeddings"]  # shape (N, 118), already L2 normalized

# Cosine similarity matrix (dot product of L2-normalized vectors)
sim_matrix = emb @ emb.T  # shape (N, N)

names = [a["name"] for a in artists]
genres = [", ".join(a.get("genres", [])[:2]) or "unknown" for a in artists]

print(f"{'Artist':<25} {'Genre':<30} → {'Top 3 similar'}")
print("=" * 100)

for i, name in enumerate(names):
    # Get top 3 (excluding self)
    row = sim_matrix[i].copy()
    row[i] = -1  # exclude self
    top_idx = np.argsort(row)[-3:][::-1]
    top_str = " | ".join(f"{names[j]} ({row[j]:.2f})" for j in top_idx)
    print(f"  {name:<25} {genres[i]:<30} → {top_str}")

print(f"\nTotal artists: {len(names)}")
print("If genres cluster (pop→pop, hip-hop→hip-hop), the pipeline is working.")
