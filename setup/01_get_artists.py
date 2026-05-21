"""
Step 1: Fetch artist metadata from Deezer API (free, no auth required).
Outputs: artists_metadata.json in the project root.
Deezer gives us: name, image, URL. Genres are from a hardcoded map.
"""
import urllib.request
import urllib.parse
import json
import os
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT = os.path.join(ROOT, "artists_metadata.json")

# Genre mapping for the artist list
GENRE_MAP = {
    "Adele": ["soul", "pop"],
    "Ed Sheeran": ["pop", "folk pop"],
    "Ariana Grande": ["pop", "r&b"],
    "Beyonce": ["r&b", "pop"],
    "Taylor Swift": ["pop", "country pop"],
    "Bruno Mars": ["pop", "funk"],
    "Dua Lipa": ["dance pop", "pop"],
    "Lady Gaga": ["electropop", "pop"],
    "Billie Eilish": ["indie pop", "alternative"],
    "The Weeknd": ["r&b", "pop"],
    "Michael Jackson": ["pop", "r&b"],
    "Whitney Houston": ["r&b", "soul"],
    "Mariah Carey": ["r&b", "pop"],
    "Sam Smith": ["pop", "soul"],
    "John Legend": ["r&b", "soul"],
    "Amy Winehouse": ["soul", "jazz"],
    "Frank Ocean": ["r&b", "neo soul"],
    "Freddie Mercury": ["rock", "glam rock"],
    "Coldplay": ["alternative rock", "pop rock"],
    "Radiohead": ["alternative rock", "art rock"],
    "Linkin Park": ["nu metal", "alternative rock"],
    "Nirvana": ["grunge", "alternative rock"],
    "Foo Fighters": ["alternative rock", "hard rock"],
    "U2": ["rock", "post-punk"],
    "Eminem": ["hip-hop", "rap"],
    "Drake": ["hip-hop", "r&b"],
    "Kendrick Lamar": ["hip-hop", "rap"],
    "Jay-Z": ["hip-hop", "rap"],
    "Cardi B": ["hip-hop", "rap"],
    "Nicki Minaj": ["hip-hop", "rap"],
    "Travis Scott": ["hip-hop", "trap"],
    "Post Malone": ["hip-hop", "pop rap"],
    "Bad Bunny": ["latin trap", "reggaeton"],
    "Shakira": ["pop latina", "latin"],
    "J Balvin": ["reggaeton", "latin pop"],
    "Rosalia": ["flamenco", "urban latin"],
    "Maluma": ["reggaeton", "latin pop"],
    "Marc Anthony": ["salsa", "latin pop"],
    "Daddy Yankee": ["reggaeton", "latin hip-hop"],
    "Luis Fonsi": ["latin pop", "pop"],
    "Ozuna": ["reggaeton", "latin pop"],
    "Karol G": ["reggaeton", "latin pop"],
    "Hozier": ["indie folk", "blues rock"],
    "Lorde": ["indie pop", "alternative"],
    "Phoebe Bridgers": ["indie folk", "emo"],
    "Elvis Presley": ["rock and roll", "pop"],
    "David Bowie": ["glam rock", "pop rock"],
    "Prince": ["funk", "pop"],
    "Johnny Cash": ["country", "folk"],
    "Bob Dylan": ["folk", "rock"],
    "Rihanna": ["pop", "r&b"],
    "Harry Styles": ["pop", "soft rock"],
    "Olivia Rodrigo": ["pop", "alternative"],
    "Doja Cat": ["pop rap", "r&b"],
    "SZA": ["r&b", "alternative r&b"],
    "Tyler the Creator": ["hip-hop", "neo soul"],
    "Shawn Mendes": ["pop", "soft rock"],
    "Camila Cabello": ["pop", "latin pop"],
    "Charlie Puth": ["pop", "r&b"],
    "Khalid": ["r&b", "indie pop"],
}

ARTISTS = list(GENRE_MAP.keys())


def search_deezer(artist_name):
    """Search Deezer API (free, no auth)."""
    query = urllib.parse.quote(artist_name)
    url = f"https://api.deezer.com/search/artist?q={query}&limit=1"
    req = urllib.request.Request(url, headers={"User-Agent": "VoiceSimilarityApp/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    items = data.get("data", [])
    if not items:
        return None
    a = items[0]
    return {
        "name": a["name"],
        "id": str(a["id"]),
        "genres": GENRE_MAP.get(artist_name, []),
        "popularity": a.get("nb_fan", 0),
        "image_url": a.get("picture_xl") or a.get("picture_big") or a.get("picture", ""),
        "spotify_url": f"https://www.deezer.com/artist/{a['id']}",
        "search_name": artist_name,
    }


metadata = []
failed = []

print(f"Fetching metadata for {len(ARTISTS)} artists from Deezer...\n")

for i, name in enumerate(ARTISTS, 1):
    try:
        result = search_deezer(name)
        if result:
            metadata.append(result)
            print(f"  [{i:02d}] ✓ {result['name']:30} fans={result['popularity']:>10,}  {result['genres'][:2]}")
        else:
            print(f"  [{i:02d}] ✗ {name} — not found on Deezer")
            # Still add with manual data so the artist appears in the app
            metadata.append({
                "name": name,
                "id": "",
                "genres": GENRE_MAP.get(name, []),
                "popularity": 0,
                "image_url": "",
                "spotify_url": f"https://www.deezer.com/search/{urllib.parse.quote(name)}",
                "search_name": name,
            })
        time.sleep(0.3)  # be polite to the API
    except Exception as e:
        print(f"  [{i:02d}] ✗ {name} — error: {e}")
        failed.append(name)
        metadata.append({
            "name": name,
            "id": "",
            "genres": GENRE_MAP.get(name, []),
            "popularity": 0,
            "image_url": "",
            "spotify_url": "",
            "search_name": name,
        })

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"  Saved: {len(metadata)} artists → artists_metadata.json")
if failed:
    print(f"  Errors: {failed}")
print(f"\nNext: run setup/02_download_clips.py")
