# Voice Similarity App — Complete Technical Documentation

> How the entire system works, from audio to result. Read this to understand the stack, the ML pipeline, and every design decision.

---

## Table of Contents

1. [What the App Does](#1-what-the-app-does)
2. [Full Stack](#2-full-stack)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Data Pipeline (Setup Phase)](#4-data-pipeline-setup-phase)
5. [Feature Engineering — The Core of the ML](#5-feature-engineering)
6. [Normalization and Similarity Math](#6-normalization-and-similarity-math)
7. [Inference Flow (Runtime)](#7-inference-flow-runtime)
8. [Per-Group Explanation Logic](#8-per-group-explanation-logic)
9. [Gradio UI Structure](#9-gradio-ui-structure)
10. [HuggingFace Spaces Deployment](#10-huggingface-spaces-deployment)
11. [File Map](#11-file-map)
12. [How to Add More Artists](#12-how-to-add-more-artists)
13. [Limitations and Known Issues](#13-limitations-and-known-issues)
14. [Future Improvements](#14-future-improvements)

---

## 1. What the App Does

The user records or uploads audio of themselves singing. The app extracts acoustic features from the audio, then compares them against pre-computed feature vectors for 102 artists using cosine similarity. The top 5 most similar artists are returned with an overall match percentage and a breakdown by feature group (timbre, dynamics, harmony, energy).

**The key insight:** you don't need to "train" a model in the traditional supervised sense. You compare a feature vector representation of the user's voice against pre-computed feature vectors of artists. The quality of the system lives entirely in the quality of the feature extraction.

---

## 2. Full Stack

### Runtime (what runs in the deployed app)

| Component | Library/Tool | Purpose |
|---|---|---|
| Feature extraction | `librosa 0.10+` | Audio analysis — MFCC, Chroma, Spectral features |
| Normalization + similarity | `scikit-learn 1.3+` | StandardScaler, L2 normalize, cosine similarity |
| Numerical computation | `numpy` | Feature vectors, matrix operations |
| Audio format handling | `ffmpeg` (system) | Convert microphone WebM/OGG to WAV before processing |
| Audio loading | `soundfile` (via librosa) | Read WAV files efficiently |
| Web UI + audio widget | `gradio 5.x` | Browser interface, microphone input, HTML output |
| Serialization | `pickle` (stdlib) | Load pre-fitted StandardScaler |
| Hosting | HuggingFace Spaces (CPU Basic, free) | Serves the app, auto-wakes on visit |

### Setup (one-time pipeline, not deployed)

| Component | Library/Tool | Purpose |
|---|---|---|
| Artist metadata | `Deezer API` (free, no auth) | Artist name, image URL, genre, fan count |
| Audio download | `yt-dlp` | Download 30s clips from YouTube |
| Audio trimming | `ffmpeg` | Trim to 30s starting at 60s (skip intros) |
| Feature extraction | `librosa` | Same pipeline as runtime |
| Scaler fitting | `scikit-learn StandardScaler` | Fit on artist corpus |
| Data persistence | `numpy .npz`, `pickle` | Save embeddings + scaler for the app |

### Infrastructure

| Component | Service | Notes |
|---|---|---|
| Demo hosting | HuggingFace Spaces | Free, sleeps after 48h inactivity (~30s cold start) |
| Code repository | GitHub | `github.com/chikiyu/voice-similarity-app` |
| Audio clips | Local only | NOT committed — only the extracted features are saved |

---

## 3. High-Level Architecture

```
────────────────── SETUP PHASE (run once locally) ──────────────────

  Deezer API ──→ artists_metadata.json   (name, image, genres, Spotify URL)
  YouTube (yt-dlp + ffmpeg) ──→ audio_clips/*.wav   (30s per artist, local only)
  librosa ──→ raw feature matrix (102 × 206)
  StandardScaler.fit_transform() ──→ scaled matrix (102 × 206)
  L2 normalize ──→ embeddings.npz  (102 × 206, for cosine similarity)
               └─→ embeddings_explain.npz  (102 × 206, scaled only, for group breakdown)
  scaler.pkl ──→ saved for inference

────────────────── RUNTIME (every user request) ────────────────────

  User audio
    │
    ▼ ffmpeg (ensure valid WAV, 22050 Hz mono)
    │
    ▼ librosa.load() → y (waveform array)
    │
    ▼ librosa.stft() → STFT matrix D  ← computed ONCE, reused for all features
    │
    ├─→ MFCC + delta MFCC
    ├─→ Chroma STFT
    ├─→ Spectral Contrast
    ├─→ Spectral Centroid + Rolloff
    └─→ ZCR + RMS
    │
    ▼ concatenate + reshape → (1, 206) raw vector
    │
    ▼ SCALER.transform() → (1, 206) scaled
    │
    ├──────────────────────────────────────→ embeddings_explain comparison
    │                                         (per-group breakdown)
    ▼ normalize(norm='l2') → (1, 206) unit vector
    │
    ▼ dot product with ARTIST_EMBEDDINGS (102, 206) → (102,) similarities
    │
    ▼ argsort → top 5 artist indices
    │
    ▼ Gradio HTML output (cards + bars)
```

---

## 4. Data Pipeline (Setup Phase)

### Step 1 — Artist metadata (`setup/01_get_artists.py`)

**What it does:** fetches artist metadata from the Deezer API (free, no authentication required) for each artist in the hardcoded `GENRE_MAP` dictionary.

**Why Deezer and not Spotify:** Spotify deprecated the Search API and audio features for new developer apps (November 2024). New Spotify apps require a Premium subscription for API access. Deezer's API is fully public and returns artist images (CDN URLs), fan counts, and artist IDs with no auth token.

**What it produces per artist:**
```json
{
  "name": "Adele",
  "id": "75798",
  "genres": ["soul", "pop"],
  "popularity": 15391837,
  "image_url": "https://e-cdns-images.dzcdn.net/images/artist/..._xl.jpg",
  "spotify_url": "https://open.spotify.com/search/Adele",
  "deezer_url": "https://www.deezer.com/artist/75798",
  "search_name": "Adele"
}
```

**Caching:** the script checks for already-fetched entries to avoid re-hitting the API. Re-running is safe.

**Note on `search_name` vs `name`:** `search_name` is what we searched with (e.g., "Pink"), `name` is what Deezer returned (e.g., "P!nk"). The audio filename uses `safe_name(search_name)` to avoid filesystem issues.

---

### Step 2 — Audio download (`setup/02_download_clips.py`)

**What it does:** for each artist, searches YouTube using `yt-dlp` and downloads the top result, then trims it to a 30-second clip starting at 60 seconds (to skip intros).

**Why skip the first 60 seconds:** most pop songs have 20-60 second intros with minimal vocals. Starting at 60s puts us in the main verse/chorus where the artist's voice is prominent.

**yt-dlp query format:**
```
ytsearch1:Adele official song vocals
```
The `ytsearch1:` prefix tells yt-dlp to take the first YouTube search result. No direct URL needed.

**Audio spec after ffmpeg trimming:**
- Format: WAV (uncompressed PCM)
- Sample rate: 22050 Hz (librosa's default)
- Channels: 1 (mono)
- Duration: 30 seconds
- Start offset: 60s from the original (or 30% in if song is short)

**Why 22050 Hz:** it's the standard for music analysis. The highest audible frequency in music is ~20kHz, and by Nyquist's theorem you need 2× that to represent it — so 40kHz minimum. 44100 Hz is the CD standard; 22050 Hz is half of that, which is sufficient for vocal feature extraction while using half the storage and compute.

**Key constraint:** audio clips are NEVER committed to git or deployed to HF Spaces. Only the extracted feature vectors are saved. This avoids copyright issues and keeps the repo small.

---

### Step 3 — Feature extraction (`setup/03_extract_embeddings.py`)

This is the most important step. See [Section 5](#5-feature-engineering) for the full explanation.

**Output files:**
- `embeddings.npz` — StandardScaler-transformed + L2-normalized feature matrix, shape `(102, 206)`. Used for cosine similarity search.
- `embeddings_explain.npz` — StandardScaler-transformed but NOT L2-normalized, shape `(102, 206)`. Used to compute per-group similarity breakdowns.
- `scaler.pkl` — the fitted `StandardScaler` object. Must be applied to user audio at inference time with the same parameters.

**Why two embedding files:**
The L2-normalized embeddings are optimal for cosine similarity (dot product of two unit vectors = cosine similarity). But for per-group breakdown, you need to normalize each group independently — and you can't do that correctly if the whole vector is already L2-normalized as a unit.

---

## 5. Feature Engineering

The feature vector is 206 dimensions, built from one shared STFT computation.

### Why compute STFT once?

The Short-Time Fourier Transform (STFT) is the most expensive computation in the pipeline. Multiple librosa features (MFCC, Chroma, Spectral Contrast, Centroid, Rolloff) all internally need the STFT. If you call them independently, librosa recomputes the STFT each time.

**Optimization:** compute STFT once, pass `S=` (the magnitude spectrogram) to all subsequent features:

```python
D        = librosa.stft(y, n_fft=2048, hop_length=512)
S_mag    = np.abs(D)          # magnitude spectrogram
S_power  = S_mag ** 2         # power spectrogram

# All spectral features reuse S_mag — no redundant computation
chroma   = librosa.feature.chroma_stft(S=S_mag, sr=sr, ...)
contrast = librosa.feature.spectral_contrast(S=S_mag, sr=sr, ...)
centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, ...)
rolloff  = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, ...)
```

This saves ~40% of processing time.

---

### Feature groups and their meaning

| Group | Features | Indices | What it captures |
|---|---|---|---|
| **Voice Timbre** | MFCC 40 (mean + std) | 0–79 | The unique "color" or "texture" of a voice. MFCC coefficients describe the shape of the vocal tract filter — this is what makes Adele sound different from Eminem even on the same note. |
| **Vocal Dynamics** | Delta MFCC 40 (mean + std) | 80–159 | How the voice changes over time. Delta = first derivative of MFCC over time. Captures the speed and pattern of vocal movements — fast rap vs. slow ballad. |
| **Pitch & Harmony** | Chroma STFT 12 (mean + std) | 160–183 | Which pitch classes (C, C#, D...) are most used. Captures whether someone sings in major/minor keys, their preferred harmonic patterns. |
| **Tone & Energy** | Spectral Contrast + Centroid + Rolloff + ZCR + RMS (all mean + std) | 184–205 | How bright, dark, loud, and energetic the voice is. High spectral centroid = bright/sharp voice (Ariana Grande). Low = dark/deep voice (Johnny Cash). |

---

### Feature definitions

**MFCC (Mel-Frequency Cepstral Coefficients)**

MFCCs are the industry standard for speech and music analysis. They capture the "shape" of the spectrum in a way that correlates with human auditory perception.

1. Compute power spectrogram from STFT
2. Apply mel filter bank (128 triangular filters spaced on mel scale)
3. Take log of energies → log-mel spectrogram
4. Apply Discrete Cosine Transform (DCT) → MFCC coefficients

The mel scale is a perceptual scale of pitches — equal spacing on the mel scale corresponds to equal perceived pitch differences. This makes MFCCs robust to pitch variation while capturing timbral identity.

We use 40 coefficients (n_mfcc=40) instead of the common 13 because more coefficients capture finer timbral detail. The tradeoff is more dimensionality, but StandardScaler handles this.

**Delta MFCC**

The first-order temporal derivative of the MFCC sequence. If MFCC[t] is the feature at time t, then delta MFCC[t] ≈ MFCC[t+1] - MFCC[t-1]. This captures the rate of change of the vocal characteristics over time — vocal expressiveness, vibrato, attack patterns.

**Chroma features**

Projects the full spectrum onto 12 bins, one per pitch class (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). Captures harmonic content independent of octave. A singer who tends to sing in minor keys will have different chroma distributions than one who sings in major keys.

**Spectral Contrast**

Measures the difference between peaks and valleys in the spectrum, per frequency sub-band. High contrast = clear harmonic structure (a clean singing voice). Low contrast = noisy signal (distorted guitar, spoken word). 7 sub-bands computed from the STFT magnitude.

**Spectral Centroid**

The "center of mass" of the spectrum — the frequency around which most of the spectral energy is concentrated. High centroid → bright, sharp voice. Low centroid → dark, full voice. A single number per frame.

**Spectral Rolloff**

The frequency below which 85% of the total spectral energy is contained. Related to centroid but captures the upper edge of the spectrum. Useful for distinguishing between high-pitched voices (Mariah Carey, Ariana Grande) and lower-voiced singers (Johnny Cash, Nick Cave).

**Zero Crossing Rate (ZCR)**

How many times per second the audio waveform crosses zero (changes sign). High ZCR → noisy or percussive signal (rap). Low ZCR → tonal singing (opera, ballad). Also captures breathiness in the voice.

**RMS Energy**

Root Mean Square energy — the overall loudness of the signal per frame. Mean and std of RMS capture both average loudness and dynamic range of the performance.

---

### Why mean and std for each feature?

Each feature (MFCC, Chroma, etc.) produces a 2D matrix: `(n_features, n_frames)` where n_frames depends on audio duration. To get a fixed-size vector, we aggregate:
- **Mean over time:** the average characteristic over the whole clip
- **Std over time:** how much that characteristic varies

Together, mean + std describe both "what kind of voice" and "how variable/expressive it is". Using just the mean loses information about vocal dynamics.

---

## 6. Normalization and Similarity Math

### StandardScaler

After extracting all 206 features for all 102 artists, we have a raw matrix of shape `(102, 206)`.

The problem: features have very different scales. MFCC values might range from -500 to +500. RMS might range from 0 to 0.3. ZCR from 0 to 1. Cosine similarity is affected by scale — large-valued features dominate.

`StandardScaler` transforms each feature to have **mean = 0, std = 1** across the artist corpus:

```
x_scaled = (x - mean_feature) / std_feature
```

This means each of the 206 features contributes equally to the similarity. It also removes the "common spectral baseline" — the fact that all commercially produced music sounds somewhat similar because of mixing, compression, and mastering. After scaling, what remains are the differences between artists.

**Critical:** the scaler is fitted on the artist corpus and then applied (`.transform()`) to the user's audio at inference. The user's audio is transformed with the same mean and std values as the artists. This ensures the comparison is in the same feature space.

### L2 Normalization

After scaling, each 206-dim vector is L2-normalized to a unit vector:

```
x_normalized = x_scaled / ||x_scaled||₂
```

After this, **cosine similarity = dot product**:

```
cosine_similarity(u, a) = u · a   (since ||u|| = ||a|| = 1)
```

This allows the similarity between the user vector and all 102 artist vectors to be computed in one matrix multiplication:

```python
similarities = ARTIST_EMBEDDINGS @ user_vector.T   # shape (102,)
```

This is O(102 × 206) — essentially instant.

### Why cosine similarity and not Euclidean distance?

Euclidean distance measures absolute differences. Cosine similarity measures **angular similarity** — it captures how similar the "direction" of two vectors is, regardless of magnitude.

For audio features, two identical voices recorded at different volumes would have different Euclidean distances (different RMS energy) but identical cosine similarity (same direction in feature space). Cosine similarity is more robust to amplitude variation, which matters because users record at different microphone levels.

### Interpreting the similarity scores

Raw cosine similarity ranges from -1 to +1. After L2 normalization, the values cluster between 0.1 and 0.7 for this dataset. For display, we map to [0, 100] using:

```python
pct = max(5, min(100, int((score + 1) / 2 * 100)))
```

This maps [-1, 1] → [0, 100] linearly. A score of 0.7 = 85%, a score of 0.3 = 65%.

---

## 7. Inference Flow (Runtime)

When a user submits audio, this sequence runs:

```python
# 1. Ensure valid WAV (handles browser WebM/OGG from microphone)
wav_path, cleanup = to_wav(audio_path)
# ffmpeg: -ar 22050 -ac 1 -f wav

# 2. Load audio
y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=15)
# 15 seconds is sufficient for reliable features

# 3. Compute STFT once
D       = librosa.stft(y, n_fft=2048, hop_length=512)
S_mag   = np.abs(D)
S_power = S_mag ** 2

# 4. Extract all features from shared STFT
mel_S    = librosa.feature.melspectrogram(S=S_power, ...)
log_mel  = librosa.power_to_db(mel_S)
mfcc     = librosa.feature.mfcc(S=log_mel, n_mfcc=40)
mfcc_d   = librosa.feature.delta(mfcc)
chroma   = librosa.feature.chroma_stft(S=S_mag, ...)
contrast = librosa.feature.spectral_contrast(S=S_mag, ...)
centroid = librosa.feature.spectral_centroid(S=S_mag, ...)
rolloff  = librosa.feature.spectral_rolloff(S=S_mag, ...)
zcr      = librosa.feature.zero_crossing_rate(y, ...)
rms      = librosa.feature.rms(y=y, ...)

# 5. Build 206-dim vector
vec = np.concatenate([mean+std for each feature])  # shape (1, 206)

# 6. Apply pre-fitted StandardScaler
scaled = SCALER.transform(vec)   # same mean/std as training corpus

# 7. L2 normalize
normed = normalize(scaled, norm='l2')   # unit vector

# 8. Cosine similarity via dot product
similarities = ARTIST_EMBEDDINGS @ normed.T   # (102,)

# 9. Top 5
top5 = np.argsort(similarities)[-5:][::-1]
```

**Total processing time:** ~0.5-1.5 seconds on CPU, depending on server load.

---

## 8. Per-Group Explanation Logic

The feature vector is divided into 4 logical groups:

```python
FEATURE_GROUPS = {
    "Voice Timbre":    (0,   80),
    "Vocal Dynamics":  (80,  160),
    "Pitch & Harmony": (160, 184),
    "Tone & Energy":   (184, 206),
}
```

For each matched artist, per-group similarity is computed using **`embeddings_explain.npz`** (scaled but NOT L2-normalized):

```python
def group_similarity(user_scaled, artist_idx, start, end):
    u = user_scaled[0, start:end]         # user's sub-vector for this group
    a = ARTIST_EXPLAIN[artist_idx, start:end]  # artist's sub-vector

    # Normalize each group independently — this IS cosine sim for the sub-space
    u_n = u / (||u|| + ε)
    a_n = a / (||a|| + ε)
    return max(0, dot(u_n, a_n))
```

**Why not use the L2-normalized full vector?**
If you take a slice of an already L2-normalized vector, the slice is NOT a unit vector. Computing a dot product of non-unit slices does not give cosine similarity. You need to normalize each group independently to get a valid similarity score for that group.

**Why `embeddings_explain.npz` instead of raw features?**
The StandardScaler has already been applied, so the per-group comparisons are still in the normalized feature space. Without the scaler, large-scale features (like MFCC values) would dominate the per-group similarity.

---

## 9. Gradio UI Structure

```
gr.Blocks
  ├── gr.Markdown (title + description)
  ├── gr.Audio(sources=["microphone","upload"], type="filepath")
  │     └── type="filepath" → returns path to a temp file
  │         The audio widget handles recording via WebRTC in the browser
  ├── gr.Button("Find my match →", variant="primary")
  │     └── triggers compare_voice(audio) → returns HTML string
  └── gr.HTML()
        └── receives the results card HTML
```

**Why `type="filepath"` instead of `type="numpy"`?**
`type="numpy"` returns a tuple `(sample_rate, array)` which bypasses format conversion. `type="filepath"` returns the path to a temp file saved by Gradio — this allows ffmpeg to process the file directly, handling any format the browser might send (WebM, OGG, WAV).

**The `to_wav()` function:**
When recording from a microphone, Chrome typically sends WebM-format audio. Firefox sends OGG. librosa/soundfile may not load these directly. The `to_wav()` function runs ffmpeg to normalize to 22050 Hz mono WAV before any processing — this is why the microphone works reliably.

**Output as HTML:**
Instead of a `gr.Dataframe` or structured output, the results are rendered as an HTML string. This gives full control over the visual design (artist images, colored bars, hover effects, Spotify links) without being constrained by Gradio's component styling.

---

## 10. HuggingFace Spaces Deployment

**Space type:** Gradio SDK (managed runtime)

**What happens when a user visits the space:**
1. If sleeping (no visitors in 48h): cold start ~30-60 seconds (builds and loads the app)
2. If awake: ~2s load (Python module loading + numpy file loading)
3. On submit: 0.5-1.5s processing (feature extraction + similarity search)

**Files that live in the HF Space repo:**
```
app.py                  ← entry point (Gradio runs this)
requirements.txt        ← librosa, numpy, scikit-learn, soundfile
README.md               ← HF Space configuration (YAML frontmatter)
artists_metadata.json   ← 102 artists, images, Spotify URLs
embeddings.npz          ← (102, 206) L2-normalized embeddings
embeddings_explain.npz  ← (102, 206) scaled-only embeddings
scaler.pkl              ← fitted StandardScaler
```

**What does NOT live in the HF Space repo:**
```
audio_clips/            ← 102 × 1.3MB WAV files (~135MB) — too large, copyright risk
setup/                  ← setup scripts not needed at runtime
```

**HF Space configuration (README.md frontmatter):**
```yaml
---
title: Voice Similarity App
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
---
```

The `sdk_version` tells HuggingFace which Gradio version to install. Since `gradio` is NOT in requirements.txt (HF manages it), this is the only place to specify the Gradio version. If you add `gradio` to requirements.txt, it can conflict.

**Deployment workflow:**
```python
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.upload_file(path_or_fileobj="file.npz", path_in_repo="file.npz",
                repo_id="ayayon/voice-similarity-app", repo_type="space")
```
Files are uploaded one by one via the HuggingFace Hub Python library. The Space rebuilds automatically after each push to the repo.

---

## 11. File Map

```
voice-similarity-app/
│
├── app.py                     ← Main Gradio app. Entry point for HF Spaces.
│                                 Loads: artists_metadata.json, embeddings.npz,
│                                        embeddings_explain.npz, scaler.pkl
│                                 Functions:
│                                   to_wav()           — ffmpeg audio normalization
│                                   extract_features() — 206-dim vector from audio
│                                   group_similarity() — per-group cosine sim
│                                   compare_voice()    — main pipeline
│                                   _results_html()    — render HTML output
│
├── requirements.txt           ← Runtime deps for HF Spaces (no gradio — HF manages it)
│
├── README.md                  ← HF Spaces config (YAML frontmatter) + project description
│
├── DOCUMENTATION.md           ← This file
│
├── artists_metadata.json      ← 102 artists with: name, genres, image_url, spotify_url
│
├── embeddings.npz             ← Pre-computed artist embeddings, shape (102, 206)
│                                 Key "embeddings": StandardScaler + L2-normalized
│
├── embeddings_explain.npz     ← Same artists, shape (102, 206)
│                                 Key "embeddings": StandardScaler only (for per-group sim)
│
├── scaler.pkl                 ← Fitted sklearn.preprocessing.StandardScaler
│                                 Must be used with .transform() on user audio
│
└── setup/
    ├── 01_get_artists.py      ← Deezer API → artists_metadata.json
    ├── 02_download_clips.py   ← yt-dlp + ffmpeg → audio_clips/*.wav
    ├── 03_extract_embeddings.py ← librosa → embeddings.npz + scaler.pkl
    └── test_similarity.py     ← Sanity check: prints top-3 similar for each artist
```

---

## 12. How to Add More Artists

**Step 1:** Add the artist to `GENRE_MAP` in `setup/01_get_artists.py`:
```python
GENRE_MAP = {
    ...
    "Rosalia": ["flamenco", "urban latin"],  # add here
    ...
}
```

**Step 2:** Run the pipeline:
```bash
python3 setup/01_get_artists.py       # fetches metadata from Deezer
python3 setup/02_download_clips.py    # downloads audio (skips existing)
python3 setup/03_extract_embeddings.py  # re-extracts ALL embeddings (scaler refits)
```

**Why re-extract all embeddings when adding new artists?**
The StandardScaler is fitted on the entire corpus. Adding a new artist changes the mean and std of each feature, which changes the scaler. All embeddings must be re-computed with the new scaler. The scaler saved in `scaler.pkl` must also be updated.

**Step 3:** Upload updated files to HF Spaces:
```python
from huggingface_hub import HfApi
api = HfApi(token="your_token")
for f in ["artists_metadata.json", "embeddings.npz", "embeddings_explain.npz", "scaler.pkl"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f,
                    repo_id="ayayon/voice-similarity-app", repo_type="space")
```

**Troubleshooting missing audio files:**
If an artist is skipped in step 3 with "no audio file", check the filename:
- The script uses `safe_name(search_name)` which strips `!`, `$`, spaces→`_`, etc.
- Example: `"A$AP Rocky"` → `"ASAP_Rocky"` → looks for `audio_clips/ASAP_Rocky.wav`
- If yt-dlp failed, retry manually: `yt-dlp "ytsearch1:Artist Name official song" -x --audio-format wav -o "audio_clips/Artist_Name.%(ext)s"`
- Then: `ffmpeg -i audio_clips/Artist_Name.wav -ss 60 -t 30 -ac 1 -ar 22050 audio_clips/SafeName.wav -y`

---

## 13. Limitations and Known Issues

### Audio quality issue: instruments contaminate features
The artist embeddings are computed from full songs, which include instruments. This means the features capture the song production style, not just the vocal characteristics. Two artists with similar production styles (e.g., same producer, same era) may show high similarity even if their voices are distinct.

**Workaround:** use clips from acoustic/stripped versions, or live vocal performances without heavy production. Simply change the yt-dlp search query to `"Artist Name acoustic a cappella"`.

**Long-term fix:** vocal source separation (see Section 14).

### Microphone cold start in browser
On the first record attempt in Chrome, the browser requests microphone permission which can cause a brief page stall. This is browser behavior, not an app bug.

### HuggingFace cold start
After 48 hours without visitors, the Space goes to sleep. The first visit triggers a rebuild which takes ~30-60 seconds. A loading spinner appears. This is a limitation of the free tier — upgrade to $9/month "persistent" hardware to eliminate this.

### "Page not responding" on very long recordings
If a user records more than ~45 seconds, the audio processing takes longer and Chrome may show a brief "page not responding" warning. The warning resolves on its own. Recording 15-25 seconds avoids this entirely.

### Similarity scores are relative, not absolute
A "match" of 75% doesn't mean your voice is objectively 75% similar to that artist. It means, within the corpus of 102 artists, that artist's feature vector is most similar to yours. With different artists in the database, the same voice could get different scores. The ranking is more meaningful than the absolute percentage.

---

## 14. Future Improvements

### Vocal isolation with Spleeter
[Spleeter](https://github.com/deezer/spleeter) (by Deezer, the same company whose API we use) separates audio into vocals + accompaniment. Running spleeter on artist clips before feature extraction would give pure vocal embeddings.

```python
from spleeter.separator import Separator
sep = Separator("spleeter:2stems")
sep.separate_to_file("artist.wav", "output/")
# → output/artist/vocals.wav (voice only)
```

**Impact:** would dramatically improve accuracy of matches. This is the single highest-impact improvement.
**Why not done now:** spleeter requires TensorFlow (~500MB) and takes ~10s per clip. Would add 20 minutes to the setup and require GPU or longer wait on HF Spaces.

### Pre-trained speaker embedding models
Models like **ECAPA-TDNN** or **wav2vec 2.0** (both open source) produce 256-dim speaker embeddings trained on millions of voice samples. These embeddings are far more powerful than hand-crafted MFCC features.

```python
# Example with speechbrain ECAPA-TDNN
from speechbrain.pretrained import SpeakerRecognition
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
embedding = model.encode_batch(audio_tensor)
```

**Caveat:** ECAPA-TDNN is trained on speech, not singing. Performance on singing voice is unknown without testing. MERT (Music Encoder Representations from Transformers) is trained on music and would be a better fit — but the model is ~1GB.

### Expanding the artist database
The setup pipeline is designed to scale. Adding 100+ more artists requires:
1. Adding to `GENRE_MAP`
2. Running `02_download_clips.py` (downloads only new artists)
3. Running `03_extract_embeddings.py` (re-extracts all — ~5 min for 200 artists)
4. Re-uploading embeddings to HF

### Genre-based filtering
Add a UI option to compare only within a genre (e.g., "find my closest Latin singer"). Would require filtering the artist matrix before similarity search.

### Multi-clip averaging
Currently each artist has one 30-second clip. Using multiple clips from different songs and averaging the feature vectors would produce a more robust artist profile.

```python
clips = ["song1.wav", "song2.wav", "song3.wav"]
vecs = [extract_features(c) for c in clips]
artist_embedding = np.mean(vecs, axis=0)
```
