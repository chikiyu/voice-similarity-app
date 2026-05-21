"""
Step 1: Fetch artist metadata from Deezer API (free, no auth required).
Outputs: artists_metadata.json in the project root.
"""
import urllib.request
import urllib.parse
import json
import os
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT = os.path.join(ROOT, "artists_metadata.json")

GENRE_MAP = {
    # Pop
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
    "Justin Timberlake": ["pop", "r&b"],
    "Usher": ["r&b", "pop"],
    "Katy Perry": ["pop", "electropop"],
    "Pink": ["pop rock", "pop"],
    "Selena Gomez": ["pop", "dance pop"],
    "Sabrina Carpenter": ["pop", "indie pop"],
    "Conan Gray": ["indie pop", "pop"],
    "Meghan Trainor": ["pop", "doo-wop pop"],
    # R&B / Soul
    "Michael Jackson": ["pop", "r&b"],
    "Whitney Houston": ["r&b", "soul"],
    "Mariah Carey": ["r&b", "pop"],
    "Sam Smith": ["pop", "soul"],
    "John Legend": ["r&b", "soul"],
    "Amy Winehouse": ["soul", "jazz"],
    "Frank Ocean": ["r&b", "neo soul"],
    "Alicia Keys": ["r&b", "soul"],
    "HER": ["r&b", "soul"],
    "Giveon": ["r&b", "soul"],
    "Daniel Caesar": ["r&b", "neo soul"],
    "Omar Apollo": ["r&b", "indie pop"],
    "Norah Jones": ["jazz", "soul"],
    "Stevie Wonder": ["soul", "r&b"],
    "Aretha Franklin": ["soul", "gospel"],
    "Ray Charles": ["soul", "r&b"],
    # Rock
    "Freddie Mercury": ["rock", "glam rock"],
    "Coldplay": ["alternative rock", "pop rock"],
    "Radiohead": ["alternative rock", "art rock"],
    "Linkin Park": ["nu metal", "alternative rock"],
    "Nirvana": ["grunge", "alternative rock"],
    "Foo Fighters": ["alternative rock", "hard rock"],
    "U2": ["rock", "post-punk"],
    "Arctic Monkeys": ["indie rock", "alternative"],
    "Muse": ["alternative rock", "progressive rock"],
    "Tame Impala": ["psychedelic pop", "indie"],
    "The Killers": ["indie rock", "alternative"],
    "Red Hot Chili Peppers": ["rock", "funk rock"],
    "Green Day": ["punk rock", "alternative rock"],
    "Pearl Jam": ["grunge", "alternative rock"],
    "Florence and the Machine": ["indie rock", "art rock"],
    "Jack White": ["blues rock", "alternative"],
    # Hip-Hop
    "Eminem": ["hip-hop", "rap"],
    "Drake": ["hip-hop", "r&b"],
    "Kendrick Lamar": ["hip-hop", "rap"],
    "Jay-Z": ["hip-hop", "rap"],
    "Cardi B": ["hip-hop", "rap"],
    "Nicki Minaj": ["hip-hop", "rap"],
    "Travis Scott": ["hip-hop", "trap"],
    "Post Malone": ["hip-hop", "pop rap"],
    "J. Cole": ["hip-hop", "conscious rap"],
    "Kanye West": ["hip-hop", "rap"],
    "Lil Wayne": ["hip-hop", "rap"],
    "21 Savage": ["hip-hop", "trap"],
    "A$AP Rocky": ["hip-hop", "trap"],
    # Latin
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
    "Rauw Alejandro": ["reggaeton", "urban latin"],
    "Becky G": ["latin pop", "reggaeton"],
    "Sebastian Yatra": ["latin pop", "pop"],
    "Maria Becerra": ["reggaeton", "latin pop"],
    "Anuel AA": ["reggaeton", "trap latino"],
    "Sech": ["reggaeton", "latin pop"],
    # Indie / Folk
    "Hozier": ["indie folk", "blues rock"],
    "Lorde": ["indie pop", "alternative"],
    "Phoebe Bridgers": ["indie folk", "emo"],
    "Bon Iver": ["indie folk", "folk"],
    "Vampire Weekend": ["indie pop", "indie rock"],
    "Fleet Foxes": ["indie folk", "folk rock"],
    "Cigarettes After Sex": ["ambient pop", "dream pop"],
    # Classic
    "Elvis Presley": ["rock and roll", "pop"],
    "David Bowie": ["glam rock", "pop rock"],
    "Prince": ["funk", "pop"],
    "Johnny Cash": ["country", "folk"],
    "Bob Dylan": ["folk", "rock"],
    "James Brown": ["funk", "soul"],
    # Extra pop
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
        "spotify_url": f"https://open.spotify.com/search/{urllib.parse.quote(artist_name)}",
        "deezer_url": f"https://www.deezer.com/artist/{a['id']}",
        "search_name": artist_name,
    }


# Load existing metadata to preserve already-fetched entries
existing = {}
if os.path.exists(OUTPUT):
    with open(OUTPUT, encoding="utf-8") as f:
        for entry in json.load(f):
            existing[entry.get("search_name", entry["name"])] = entry

metadata = []
new_count = 0
skip_count = 0

print(f"Processing {len(ARTISTS)} artists (existing: {len(existing)})...\n")

for i, name in enumerate(ARTISTS, 1):
    if name in existing:
        entry = existing[name]
        # Update spotify_url if missing (old entries didn't have it)
        if "spotify_url" not in entry or "deezer.com" in entry.get("spotify_url", ""):
            entry["spotify_url"] = f"https://open.spotify.com/search/{urllib.parse.quote(name)}"
        metadata.append(entry)
        skip_count += 1
        print(f"  [{i:03d}] skip {name} (cached)")
        continue

    try:
        result = search_deezer(name)
        if result:
            metadata.append(result)
            new_count += 1
            print(f"  [{i:03d}] ✓    {result['name']:30} {result['genres'][:2]}")
        else:
            entry = {
                "name": name, "id": "", "genres": GENRE_MAP.get(name, []),
                "popularity": 0, "image_url": "",
                "spotify_url": f"https://open.spotify.com/search/{urllib.parse.quote(name)}",
                "deezer_url": "", "search_name": name,
            }
            metadata.append(entry)
            new_count += 1
            print(f"  [{i:03d}] ~    {name} (not found on Deezer, added manually)")
        time.sleep(0.2)
    except Exception as e:
        entry = {
            "name": name, "id": "", "genres": GENRE_MAP.get(name, []),
            "popularity": 0, "image_url": "",
            "spotify_url": f"https://open.spotify.com/search/{urllib.parse.quote(name)}",
            "deezer_url": "", "search_name": name,
        }
        metadata.append(entry)
        new_count += 1
        print(f"  [{i:03d}] ✗    {name} — {e}")

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"  Total:    {len(metadata)} artists → artists_metadata.json")
print(f"  New:      {new_count} | Cached: {skip_count}")
print(f"\nNext: run setup/02_download_clips.py")
