import gradio as gr
import librosa
import numpy as np
from sklearn.preprocessing import normalize
import json
import os
import pickle

# ---------------------------------------------------------------------------
# Load pre-computed data (committed to repo — persists on HF Spaces)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "artists_metadata.json"), encoding="utf-8") as f:
    ARTISTS = json.load(f)

data = np.load(os.path.join(BASE_DIR, "embeddings.npz"))
ARTIST_EMBEDDINGS = data["embeddings"]  # shape (N, 198), scaled + L2 normalized

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

# ---------------------------------------------------------------------------
# Feature extraction — must match setup/03_extract_embeddings.py exactly
# ---------------------------------------------------------------------------
def extract_features(audio_path: str) -> np.ndarray | None:
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
    except Exception:
        return None

    if len(y) < sr * 3:
        return None

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_d   = librosa.feature.delta(mfcc)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    vec = np.concatenate([
        np.mean(mfcc, axis=1),      np.std(mfcc, axis=1),
        np.mean(mfcc_d, axis=1),    np.std(mfcc_d, axis=1),
        np.mean(chroma, axis=1),    np.std(chroma, axis=1),
        np.mean(contrast, axis=1),  np.std(contrast, axis=1),
    ]).astype(np.float32).reshape(1, -1)  # shape (1, 198)

    # Apply same StandardScaler fitted on artist embeddings
    vec = SCALER.transform(vec)
    return vec  # shape (1, 198)

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def compare_voice(audio):
    if audio is None:
        return _error_html("Please record or upload an audio file.")

    features = extract_features(audio)

    if features is None:
        return _error_html(
            "Audio too short or unreadable. "
            "Please record at least 5 seconds of yourself singing."
        )

    features = normalize(features, norm="l2")

    # Cosine similarity (dot product, both sides are L2 normalized)
    similarities = (ARTIST_EMBEDDINGS @ features.T).flatten()

    top5_idx = np.argsort(similarities)[-5:][::-1]

    return _results_html(top5_idx, similarities)

# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
MEDALS = ["🥇", "🥈", "🥉", "4th", "5th"]
BAR_COLOR = "#1DB954"

def _results_html(top5_idx, similarities):
    cards = ""
    for rank, idx in enumerate(top5_idx):
        artist = ARTISTS[idx]
        score  = float(similarities[idx])
        name   = artist["name"]
        genres = ", ".join(artist.get("genres", [])[:2]) or "Various"
        img    = artist.get("image_url", "")
        pct    = max(5, min(100, int((score + 1) / 2 * 100)))  # map [-1,1] → [0,100]
        spotify_url = artist.get("spotify_url", "#")

        img_tag = (
            f'<img src="{img}" width="56" height="56" '
            f'style="border-radius:50%;object-fit:cover;flex-shrink:0;" />'
            if img else
            '<div style="width:56px;height:56px;border-radius:50%;'
            'background:#333;flex-shrink:0;"></div>'
        )

        cards += f"""
        <a href="{spotify_url}" target="_blank" style="text-decoration:none;color:inherit;">
          <div style="display:flex;align-items:center;gap:14px;padding:12px 16px;
                      margin:8px 0;background:#1a1a1a;border-radius:12px;
                      transition:background 0.2s;"
               onmouseover="this.style.background='#222'"
               onmouseout="this.style.background='#1a1a1a'">
            <span style="font-size:1.6em;min-width:32px;text-align:center;">{MEDALS[rank]}</span>
            {img_tag}
            <div style="flex:1;min-width:0;">
              <div style="font-weight:700;font-size:1em;color:#fff;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
              <div style="font-size:0.78em;color:#aaa;margin:2px 0;">{genres}</div>
              <div style="background:#333;border-radius:4px;height:6px;margin-top:6px;">
                <div style="background:{BAR_COLOR};width:{pct}%;height:6px;
                            border-radius:4px;transition:width 0.5s;"></div>
              </div>
            </div>
            <div style="font-size:1em;font-weight:700;color:{BAR_COLOR};
                        min-width:44px;text-align:right;">{pct}%</div>
          </div>
        </a>"""

    return f"""
    <div style="font-family:'Segoe UI',sans-serif;max-width:520px;padding:4px;">
      <h3 style="color:{BAR_COLOR};margin:0 0 12px;">🎤 Your voice sounds like...</h3>
      {cards}
      <p style="color:#555;font-size:0.72em;margin-top:14px;">
        Results based on timbral similarity (MFCC + Chroma features).
        Click an artist to open their Spotify profile.
      </p>
    </div>"""

def _error_html(msg):
    return f"""
    <div style="font-family:'Segoe UI',sans-serif;padding:16px;
                background:#2a1a1a;border-radius:8px;color:#ff6b6b;">
      ⚠️ {msg}
    </div>"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Voice Similarity — Who do you sound like?",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 620px !important; margin: auto; }"
) as demo:

    gr.Markdown("""
    # 🎙️ Who do you sound like?
    Record yourself singing **15–30 seconds** of any song → ML compares your vocal features
    against **{n} artists** → see your **Top 5 most similar singers**.
    """.format(n=len(ARTISTS)))

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record or upload your voice"
        )

    btn = gr.Button("🔍 Find my singer match", variant="primary", size="lg")
    output = gr.HTML(label="Your matches")

    btn.click(fn=compare_voice, inputs=audio_input, outputs=output)

    gr.Markdown("""
    ---
    **How it works:** Audio → MFCC (40 coefficients) + Chroma + Spectral Contrast features
    → 118-dim vector → cosine similarity against pre-computed artist embeddings.
    Built with [librosa](https://librosa.org) + [scikit-learn](https://scikit-learn.org) + [Gradio](https://gradio.app).
    [GitHub](https://github.com/chikiyu/voice-similarity-app)
    """)

if __name__ == "__main__":
    demo.launch()
