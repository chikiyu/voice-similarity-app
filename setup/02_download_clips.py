"""
Step 2: Download 30-second audio clips for each artist via yt-dlp + ffmpeg.
Only used during setup — audio files are NEVER committed to git.
Outputs: audio_clips/<ArtistName>.wav
"""
import subprocess
import json
import os
import glob

ROOT = os.path.join(os.path.dirname(__file__), "..")
METADATA_PATH = os.path.join(ROOT, "artists_metadata.json")
CLIPS_DIR = os.path.join(ROOT, "audio_clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

with open(METADATA_PATH, encoding="utf-8") as f:
    artists = json.load(f)

def safe_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("'", "").replace(".", "")

def download_clip(artist_name, out_path):
    """Download 30s clip starting at 60s (skip typical intros)."""
    query = f"ytsearch1:{artist_name} official song vocals"
    temp_template = os.path.join(CLIPS_DIR, f"_tmp_{safe_name(artist_name)}.%(ext)s")

    # Download best audio
    result = subprocess.run(
        ["yt-dlp", query,
         "-x", "--audio-format", "wav",
         "--audio-quality", "5",
         "-o", temp_template,
         "--no-playlist",
         "--quiet",
         "--no-warnings"],
        capture_output=True, text=True, timeout=90
    )

    # Find the downloaded temp file
    pattern = os.path.join(CLIPS_DIR, f"_tmp_{safe_name(artist_name)}.*")
    temp_files = glob.glob(pattern)

    if not temp_files:
        return False, "yt-dlp download failed"

    raw = temp_files[0]

    # Get duration of the downloaded file
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", raw],
        capture_output=True, text=True
    )
    try:
        duration = float(probe.stdout.strip())
    except:
        duration = 120.0

    # Choose start time: 60s in, or 30% into the song if shorter
    start = min(60, max(0, int(duration * 0.3)))
    clip_len = 30  # seconds

    # If song is too short for start+30s, use beginning
    if start + clip_len > duration:
        start = max(0, int(duration * 0.1))

    trim_result = subprocess.run(
        ["ffmpeg",
         "-i", raw,
         "-ss", str(start),
         "-t", str(clip_len),
         "-ac", "1",       # mono
         "-ar", "22050",   # sample rate librosa uses
         out_path,
         "-y",
         "-loglevel", "error"],
        capture_output=True, text=True
    )

    os.remove(raw)

    if trim_result.returncode != 0 or not os.path.exists(out_path):
        return False, trim_result.stderr

    # Verify output is not empty
    if os.path.getsize(out_path) < 10_000:  # < 10KB is suspicious
        os.remove(out_path)
        return False, "Output file too small"

    return True, f"start={start}s, len={clip_len}s"

print(f"Downloading clips for {len(artists)} artists...\n")
ok_count = 0
fail_count = 0
failed_artists = []

for i, artist in enumerate(artists, 1):
    name = artist["name"]
    out_path = os.path.join(CLIPS_DIR, f"{safe_name(name)}.wav")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        print(f"  [{i:02d}/{len(artists)}] skip  {name} (already exists)")
        ok_count += 1
        continue

    print(f"  [{i:02d}/{len(artists)}] dl    {name}...", end=" ", flush=True)
    try:
        success, info = download_clip(name, out_path)
        if success:
            print(f"✓ ({info})")
            ok_count += 1
        else:
            print(f"✗ ({info})")
            fail_count += 1
            failed_artists.append(name)
    except subprocess.TimeoutExpired:
        print(f"✗ (timeout)")
        fail_count += 1
        failed_artists.append(name)
    except Exception as e:
        print(f"✗ ({e})")
        fail_count += 1
        failed_artists.append(name)

print(f"\n{'='*50}")
print(f"  OK:     {ok_count}")
print(f"  Failed: {fail_count}")
if failed_artists:
    print(f"  Failed artists: {failed_artists}")
print(f"\nNext: run setup/03_extract_embeddings.py")
