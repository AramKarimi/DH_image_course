"""Download public-domain cultural-heritage sample images from Wikimedia Commons.

Usage:
    python scripts/download_data.py

Images are saved to data/raw/. All files are public domain or CC-licensed
works hosted on Wikimedia Commons. We request a resized version (max width
1280px) to keep the dataset lightweight for teaching.
"""

import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# (local filename, Wikimedia Commons file title)
IMAGES = [
    ("mona_lisa.jpg",
     "Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg"),
    ("starry_night.jpg",
     "Van Gogh - Starry Night - Google Art Project.jpg"),
    ("girl_pearl_earring.jpg",
     "1665 Girl with a Pearl Earring.jpg"),
    ("great_wave.jpg",
     "Tsunami by hokusai 19th century.jpg"),
    ("the_scream.jpg",
     "Edvard Munch, 1893, The Scream, oil, tempera and pastel on cardboard, "
     "91 x 73 cm, National Gallery of Norway.jpg"),
    ("american_gothic.jpg",
     "Grant Wood - American Gothic - Google Art Project.jpg"),
    ("rosetta_stone.jpg",
     "Rosetta Stone.JPG"),
    ("nefertiti.jpg",
     "Nofretete Neues Museum.jpg"),
    ("stonehenge.jpg",
     "Stonehenge2007 07 30.jpg"),
    ("louvre_crowd.jpg",
     "Crowd looking at the Mona Lisa at the Louvre.jpg"),
]

# Wikimedia requires a descriptive User-Agent for API/file requests.
HEADERS = {
    "User-Agent": "DH-ImageProcessing-Course/1.0 (educational use; "
                  "https://commons.wikimedia.org)"
}


def download(filename: str, commons_title: str, width: int = 1280) -> bool:
    dest = DATA_DIR / filename
    if dest.exists():
        print(f"  [skip] {filename} already exists")
        return True
    quoted = urllib.parse.quote(commons_title)
    url = (f"https://commons.wikimedia.org/wiki/Special:FilePath/"
           f"{quoted}?width={width}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  [ok]   {filename} ({len(data) // 1024} KB)")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  [FAIL] {filename}: {exc}")
        return False


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(IMAGES)} images to {DATA_DIR}\n")
    failures = sum(not download(name, title) for name, title in IMAGES)
    print(f"\nDone. {len(IMAGES) - failures}/{len(IMAGES)} images available.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
