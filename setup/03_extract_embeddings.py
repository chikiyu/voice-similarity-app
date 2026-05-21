"""
Step 3: Extract audio features → embeddings.npz + scaler.pkl + embeddings_explain.npz

Feature vector (206 dims):
  MFCC 40 (mean+std)        = 80  — vocal timbre
  Delta MFCC 40 (mean+std)  = 80  — vocal dynamics
  Chroma STFT 12 (mean+std) = 24  — pitch/harmony profile
  Spectral Contrast 7 m+s   = 14  — brightness
  Spectral Centroid m+s     =  2  — vocal brightness
  Spectral Rolloff m+s      =  2  — high-freq content
  ZCR m+s                   =  2  — voice vs unvoiced
  RMS m+s                   =  2  — energy envelope
Total: 206

Optimization: single STFT computed once, reused for all features.
Duration reduced to 15s (faster inference, sufficient for features).
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
EXPLAIN_PATH = os.path.join(ROOT, "embeddings_explain.npz")
SCALER_PATH = os.path.join(ROOT, "scaler.pkl")

# Feature group boundaries (for per-group similarity explanations in app.py)
FEATURE_GROUPS = {
    "Voice Timbre":    (0,   80),   # MFCC mean+std
    "Vocal Dynamics":  (80,  160),  # Delta MFCC mean+std
    "Pitch & Harmony": (160, 184),  # Chroma mean+std
    "Tone & Energy":   (184, 206),  # contrast+centroid+rolloff+ZCR+RMS
}

N_FFT = 2048
HOP_LENGTH = 512
DURATION = 15  # seconds — sufficient for features, faster than 30s


def safe_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("'", "").replace(".", "").replace("$", "S").replace("!", "")


def extract_features(path):
    """Extract 206-dim feature vector. Uses single STFT for speed."""
    y, sr = librosa.load(path, sr=22050, mono=True, duration=DURATION)
    if len(y) < sr * 3:
        return None

    # ── Single STFT ──
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(D)
    S_power = S_mag ** 2

    # MFCC via mel spectrogram from power STFT
    mel_S = librosa.feature.melspectrogram(S=S_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel_S)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=40)
    mfcc_d = librosa.feature.delta(mfcc)

    # Spectral features — all reuse S_mag
    chroma   = librosa.feature.chroma_stft(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff  = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms      = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    vec = np.concatenate([
        np.mean(mfcc, axis=1),      np.std(mfcc, axis=1),      # 80  [0:80]
        np.mean(mfcc_d, axis=1),    np.std(mfcc_d, axis=1),    # 80  [80:160]
        np.mean(chroma, axis=1),    np.std(chroma, axis=1),     # 24  [160:184]
        np.mean(contrast, axis=1),  np.std(contrast, axis=1),   # 14  [184:198]
        [np.mean(centroid), np.std(centroid)],                  # 2   [198:200]
        [np.mean(rolloff),  np.std(rolloff)],                   # 2   [200:202]
        [np.mean(zcr),      np.std(zcr)],                       # 2   [202:204]
        [np.mean(rms),      np.std(rms)],                       # 2   [204:206]
    ])
    return vec.astype(np.float32)  # shape (206,)


with open(METADATA_PATH, encoding="utf-8") as f:
    artists = json.load(f)

embeddings_list = []
valid_artists = []
skipped = []

print(f"Extracting features from {len(artists)} artists (duration={DURATION}s)...\n")

for i, artist in enumerate(artists, 1):
    name = artist["name"]
    sname = safe_name(artist.get("search_name", name))
    clip_path = os.path.join(CLIPS_DIR, f"{sname}.wav")

    # Try alternate filename patterns if primary doesn't exist
    if not os.path.exists(clip_path):
        alt = safe_name(name)
        clip_path = os.path.join(CLIPS_DIR, f"{alt}.wav")

    if not os.path.exists(clip_path):
        print(f"  [{i:03d}] skip  {name} (no audio file — run 02_download_clips.py first)")
        skipped.append(name)
        continue

    try:
        vec = extract_features(clip_path)
        if vec is None:
            print(f"  [{i:03d}] skip  {name} (clip too short)")
            skipped.append(name)
            continue
        embeddings_list.append(vec)
        valid_artists.append(artist)
        print(f"  [{i:03d}] ✓     {name:35} {vec.shape}")
    except Exception as e:
        print(f"  [{i:03d}] ✗     {name} — {e}")
        skipped.append(name)

if not embeddings_list:
    print("\nERROR: No embeddings generated.")
    exit(1)

raw_matrix = np.array(embeddings_list, dtype=np.float32)  # (N, 206)

# StandardScaler: removes common spectral baseline, highlights differences
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(raw_matrix)

# L2-normalized for cosine similarity (main embeddings)
final_matrix = normalize(scaled_matrix, norm="l2").astype(np.float32)

# Save
np.savez(EMBEDDINGS_PATH, embeddings=final_matrix, feature_groups=json.dumps(FEATURE_GROUPS))
np.savez(EXPLAIN_PATH, embeddings=scaled_matrix.astype(np.float32))  # for per-group breakdown
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(valid_artists, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"  Artists: {len(valid_artists)} | Skipped: {len(skipped)}")
print(f"  Shape:   {final_matrix.shape}")
print(f"  Saved:   embeddings.npz ({os.path.getsize(EMBEDDINGS_PATH)//1024}KB) + embeddings_explain.npz + scaler.pkl")
print(f"\n  Feature groups:")
for g, (s, e) in FEATURE_GROUPS.items():
    print(f"    [{s:3d}:{e:3d}] {g}")
