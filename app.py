import gradio as gr
import librosa
import numpy as np
from sklearn.preprocessing import normalize
import json
import os
import pickle
import subprocess
import tempfile

# ── Load pre-computed data ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "artists_metadata.json"), encoding="utf-8") as f:
    ARTISTS = json.load(f)

_emb = np.load(os.path.join(BASE_DIR, "embeddings.npz"))
ARTIST_EMBEDDINGS = _emb["embeddings"]          # (N, 206) L2-normalized

_exp = np.load(os.path.join(BASE_DIR, "embeddings_explain.npz"))
ARTIST_EXPLAIN = _exp["embeddings"]             # (N, 206) scaled, not L2-norm

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

# Feature group boundaries — must match 03_extract_embeddings.py
FEATURE_GROUPS = {
    "Voice Timbre":    (0,   80),
    "Vocal Dynamics":  (80,  160),
    "Pitch & Harmony": (160, 184),
    "Tone & Energy":   (184, 206),
}

N_FFT = 2048
HOP_LENGTH = 512
DURATION = 15

# ── Audio helpers ─────────────────────────────────────────────────────────────

def to_wav(audio_path: str) -> tuple[str, bool]:
    """Ensure audio is a valid WAV at 22050 Hz mono. Returns (path, needs_cleanup)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path,
         "-ar", "22050", "-ac", "1", "-f", "wav",
         tmp.name, "-y", "-loglevel", "error"],
        capture_output=True, timeout=30
    )
    if result.returncode == 0 and os.path.getsize(tmp.name) > 1000:
        return tmp.name, True
    os.unlink(tmp.name)
    return audio_path, False

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(audio_path: str) -> np.ndarray | None:
    """Extract 206-dim feature vector using single shared STFT."""
    wav_path, cleanup = to_wav(audio_path)
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=DURATION)
    finally:
        if cleanup:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    if len(y) < sr * 3:
        return None

    # Single STFT — reused for all spectral features
    D        = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag    = np.abs(D)
    S_power  = S_mag ** 2

    mel_S    = librosa.feature.melspectrogram(S=S_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel  = librosa.power_to_db(mel_S)
    mfcc     = librosa.feature.mfcc(S=log_mel, n_mfcc=40)
    mfcc_d   = librosa.feature.delta(mfcc)
    chroma   = librosa.feature.chroma_stft(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff  = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms      = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    vec = np.concatenate([
        np.mean(mfcc, axis=1),      np.std(mfcc, axis=1),
        np.mean(mfcc_d, axis=1),    np.std(mfcc_d, axis=1),
        np.mean(chroma, axis=1),    np.std(chroma, axis=1),
        np.mean(contrast, axis=1),  np.std(contrast, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(rolloff),  np.std(rolloff)],
        [np.mean(zcr),      np.std(zcr)],
        [np.mean(rms),      np.std(rms)],
    ]).astype(np.float32).reshape(1, -1)

    return vec  # (1, 206) raw


def group_similarity(user_scaled: np.ndarray, artist_idx: int, start: int, end: int) -> float:
    """Cosine similarity on a feature sub-group (normalized independently)."""
    u = user_scaled[0, start:end]
    a = ARTIST_EXPLAIN[artist_idx, start:end]
    u_n = u / (np.linalg.norm(u) + 1e-8)
    a_n = a / (np.linalg.norm(a) + 1e-8)
    return max(0.0, float(np.dot(u_n, a_n)))


# ── Main compare function ─────────────────────────────────────────────────────

def compare_voice(audio):
    if audio is None:
        return _msg("Record or upload a voice sample to get started.")

    raw = extract_features(audio)
    if raw is None:
        return _msg("Audio too short — please record at least 5 seconds of yourself singing.")

    # Scale → L2 normalize
    scaled = SCALER.transform(raw)          # (1, 206) — for per-group breakdown
    normed = normalize(scaled, norm="l2")   # (1, 206) — for cosine sim

    # Overall cosine similarity
    similarities = (ARTIST_EMBEDDINGS @ normed.T).flatten()
    top5 = np.argsort(similarities)[-5:][::-1]

    return _results_html(top5, similarities, scaled)


# ── HTML rendering ────────────────────────────────────────────────────────────

MEDALS = ["1st", "2nd", "3rd", "4th", "5th"]
ACC = "#16a34a"   # green accent

def _pct_bar(value: float, color: str = ACC, height: int = 4) -> str:
    p = max(2, min(100, int(value * 100)))
    return (
        f'<div style="background:#e5e7eb;border-radius:2px;height:{height}px;margin:3px 0;">'
        f'<div style="background:{color};width:{p}%;height:{height}px;border-radius:2px;"></div></div>'
    )

def _results_html(top5, similarities, user_scaled):
    cards = ""
    for rank, idx in enumerate(top5):
        artist = ARTISTS[idx]
        score  = float(similarities[idx])
        name   = artist["name"]
        genres = " · ".join(artist.get("genres", [])[:2]) or "Various"
        img    = artist.get("image_url", "")
        url    = artist.get("spotify_url", f"https://open.spotify.com/search/{name}")
        overall_pct = max(5, min(100, int((score + 1) / 2 * 100)))

        # Per-group breakdown
        group_bars = ""
        for g_name, (s, e) in FEATURE_GROUPS.items():
            g_sim = group_similarity(user_scaled, idx, s, e)
            g_pct = int(g_sim * 100)
            group_bars += (
                f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
                f'<span style="font-size:0.68em;color:#6b7280;width:90px;flex-shrink:0;">{g_name}</span>'
                f'<div style="flex:1;">{_pct_bar(g_sim, ACC, 4)}</div>'
                f'<span style="font-size:0.68em;color:#374151;width:28px;text-align:right;">{g_pct}%</span>'
                f'</div>'
            )

        img_tag = (
            f'<img src="{img}" width="48" height="48" '
            f'style="border-radius:50%;object-fit:cover;flex-shrink:0;" />'
            if img else
            '<div style="width:48px;height:48px;border-radius:50%;background:#e5e7eb;flex-shrink:0;"></div>'
        )

        cards += f"""
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;margin:8px 0;background:#fff;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            {img_tag}
            <div style="flex:1;min-width:0;">
              <div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;">
                <div>
                  <span style="font-size:0.75em;color:#9ca3af;font-weight:500;letter-spacing:0.05em;">{MEDALS[rank].upper()}</span>
                  <a href="{url}" target="_blank"
                     style="display:block;font-weight:700;font-size:1em;color:#111827;text-decoration:none;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                     title="Open on Spotify">{name} ↗</a>
                  <span style="font-size:0.75em;color:#6b7280;">{genres}</span>
                </div>
                <span style="font-size:1.1em;font-weight:700;color:{ACC};flex-shrink:0;">{overall_pct}%</span>
              </div>
              {_pct_bar(score, ACC, 6)}
            </div>
          </div>
          <div style="border-top:1px solid #f3f4f6;padding-top:8px;">
            {group_bars}
          </div>
        </div>"""

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:540px;">
      <p style="font-size:0.8em;color:#6b7280;margin:0 0 10px;">
        Compared against <strong>{len(ARTISTS)}</strong> artists · click any name to open Spotify
      </p>
      {cards}
      <p style="font-size:0.7em;color:#9ca3af;margin-top:12px;">
        Based on MFCC + delta + Chroma + Spectral features · results reflect vocal timbre similarity,
        not singing skill
      </p>
    </div>"""


def _msg(text: str) -> str:
    return f'<p style="font-family:system-ui;color:#6b7280;padding:12px 0;">{text}</p>'


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 580px !important; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="Who do you sound like?", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown("## 🎙️ Who do you sound like?\nSing 15–30 seconds → find your closest artist match among **100 artists**.")

    audio_in = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Record or upload your voice",
        show_download_button=False,
    )
    btn = gr.Button("Find my match →", variant="primary")
    out = gr.HTML()

    btn.click(fn=compare_voice, inputs=audio_in, outputs=out)
    audio_in.change(fn=lambda _: None, inputs=audio_in, outputs=[])  # reset on new upload

    gr.Markdown(
        "<small style='color:#9ca3af;'>Stack: librosa · scikit-learn · Gradio · "
        "[GitHub](https://github.com/chikiyu/voice-similarity-app)</small>"
    )

if __name__ == "__main__":
    demo.launch()
