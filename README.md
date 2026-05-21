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

# 🎙️ Voice Similarity App — Who do you sound like?

> Record your voice singing → ML compares your vocal features against 60+ artists → Top 5 most similar singers

**[→ Try it live on Hugging Face Spaces](https://huggingface.co/spaces/ayayon/voice-similarity-app)**

---

## How it works

1. Record 15–30 seconds of yourself singing (or upload an audio file)
2. The app extracts **MFCC + Chroma + Spectral Contrast** features using `librosa`
3. Your 118-dimensional feature vector is compared to pre-computed artist embeddings via **cosine similarity**
4. Top 5 most vocally similar artists are returned with a similarity score

```
audio (.wav/mp3) → librosa feature extraction → 118-dim L2-normalized vector
    → cosine similarity against 60+ artist embeddings → Top 5 results
```

## Feature vector breakdown

| Feature | Dimensions | What it captures |
|---|---|---|
| MFCC (40 coefficients, mean + std) | 80 | Vocal timbre — the "color" of your voice |
| Chroma (12 bins, mean + std) | 24 | Harmonic/pitch profile |
| Spectral Contrast (7 bands, mean + std) | 14 | Vocal brightness vs. depth |
| **Total** | **118** | |

## Artists covered (60+)

Pop · R&B · Rock · Hip-Hop · Latin · Indie · Classic

Adele, Beyoncé, Ariana Grande, Taylor Swift, Ed Sheeran, The Weeknd, Bruno Mars,
Michael Jackson, Whitney Houston, Amy Winehouse, Eminem, Drake, Kendrick Lamar,
Bad Bunny, Shakira, J Balvin, Rosalía, Daddy Yankee, and more.

## Stack

```
Python · librosa · scikit-learn · Gradio · Hugging Face Spaces · spotipy
```

## Run locally

```bash
git clone https://github.com/chikiyu/voice-similarity-app
cd voice-similarity-app
pip install -r requirements.txt
python app.py
```

## Reproduce the artist embeddings

```bash
# Requires: Spotify API credentials + yt-dlp + ffmpeg
pip install yt-dlp spotipy python-dotenv
python setup/01_get_artists.py    # fetch metadata from Spotify
python setup/02_download_clips.py # download 30s clips via yt-dlp
python setup/03_extract_embeddings.py  # extract features → embeddings.npz
python setup/test_similarity.py   # sanity check
```

## Limitations

- Results are based on **timbral similarity**, not pitch accuracy or musical skill
- Artist embeddings are extracted from full songs (including instrumentals), which adds background noise to the comparison. Future improvement: vocal isolation with Spleeter
- Best results when you sing clearly for 15+ seconds

## Author

[Milton Pachari](https://github.com/chikiyu) — CS grad · Cusco, Perú
