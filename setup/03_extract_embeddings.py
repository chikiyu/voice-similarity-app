"""
Step 3: Extract audio features from downloaded clips and save embeddings.
Outputs:
  - embeddings.npz   (committed to git — small, ~60KB for 60 artists)
  - artists_metadata.json  (updated to only include artists with valid clips)
"""
import librosa
import numpy as np
import json
import os
import pickle
from sklearn.preprocessing import normalize, StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLIPS_DIR = os.path.join(ROOT, "audio_clips")
METADATA_PATH = os.path.join(ROOT, "artists_metadata.json")
EMBEDDINGS_PATH = os.path.join(ROOT, "embeddings.npz")
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")

def safe_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("'", "").replace(".", "")

def extract_features(path, duration=30):
    """
    Returns a 158-dim feature vector:
      MFCC (40 coef, mean+std)          = 80 features — vocal timbre
      MFCC delta (40 coef, mean+std)    = 80 features — vocal dynamics (rate of change)  [subset: first 20]
      Chroma (12 bins, mean+std)        = 24 features — harmonic profile
      Spectral contrast (7, mean+std)   = 14 features — vocal brightness
    Total = 198 features
    """
    y, sr = librosa.load(path, sr=22050, mono=True, duration=duration)

    if len(y) < sr * 3:
        return None  # Too short

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_d   = librosa.feature.delta(mfcc)            # delta = rate of change over time
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    vec = np.concatenate([
        np.mean(mfcc, axis=1),     np.std(mfcc, axis=1),      # 80
        np.mean(mfcc_d, axis=1),   np.std(mfcc_d, axis=1),    # 80
        np.mean(chroma, axis=1),   np.std(chroma, axis=1),     # 24
        np.mean(contrast, axis=1), np.std(contrast, axis=1),   # 14
    ])
    return vec  # shape (198,)

with open(METADATA_PATH, encoding="utf-8") as f:
    artists = json.load(f)

embeddings_list = []
valid_artists = []
skipped = []

print(f"Extracting features from {len(artists)} artists...\n")

for i, artist in enumerate(artists, 1):
    name = artist["name"]
    clip_path = os.path.join(CLIPS_DIR, f"{safe_name(name)}.wav")

    if not os.path.exists(clip_path):
        print(f"  [{i:02d}] skip  {name} (no audio file)")
        skipped.append(name)
        continue

    try:
        vec = extract_features(clip_path)
        if vec is None:
            print(f"  [{i:02d}] skip  {name} (clip too short)")
            skipped.append(name)
            continue

        embeddings_list.append(vec)
        valid_artists.append(artist)
        print(f"  [{i:02d}] ✓     {name:30} vec={vec.shape}")

    except Exception as e:
        print(f"  [{i:02d}] ✗     {name} — {e}")
        skipped.append(name)

if not embeddings_list:
    print("\nERROR: No embeddings generated. Run 02_download_clips.py first.")
    exit(1)

# 1. Stack raw features
raw_matrix = np.array(embeddings_list, dtype=np.float32)  # shape (N, 198)

# 2. StandardScaler: removes common spectral baseline shared by all songs.
#    Each feature → mean=0, std=1 across all artists.
#    This makes differences between artists more discriminative.
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(raw_matrix)

# 3. L2 normalize (required for cosine similarity via dot product)
matrix = normalize(scaled_matrix, norm="l2").astype(np.float32)

# Save embeddings + scaler (scaler is needed at inference time)
np.savez(EMBEDDINGS_PATH, embeddings=matrix)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Update metadata to only include valid artists (in same order as embeddings)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(valid_artists, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"  Artists processed: {len(valid_artists)}")
print(f"  Skipped:           {len(skipped)}")
print(f"  Feature dim:       {raw_matrix.shape[1]} raw → {matrix.shape[1]} after scaling")
print(f"  Embedding shape:   {matrix.shape}")
print(f"  Saved → embeddings.npz ({os.path.getsize(EMBEDDINGS_PATH)//1024} KB)")
print(f"  Saved → scaler.pkl")
print(f"\nNext: run setup/test_similarity.py to verify quality")
