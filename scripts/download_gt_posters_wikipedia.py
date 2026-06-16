#!/usr/bin/env python3
"""Download a small traced GT poster subset from Wikipedia file pages."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import ssl
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


POSTERS = [
    {
        "poster_id": 2,
        "poster_name": "The Dark Knight (2008)",
        "page_title": "The Dark Knight",
        "file_title": "File:The Dark Knight (2008 film).jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg",
    },
    {
        "poster_id": 3,
        "poster_name": "The Matrix (1999)",
        "page_title": "The Matrix",
        "file_title": "File:The Matrix.png",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/d/db/The_Matrix.png",
    },
    {
        "poster_id": 7,
        "poster_name": "Inception (2010)",
        "page_title": "Inception",
        "file_title": "File:Inception (2010) theatrical poster.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/2/2e/Inception_%282010%29_theatrical_poster.jpg",
    },
    {
        "poster_id": 9,
        "poster_name": "Star Wars: Episode IV - A New Hope (1977)",
        "page_title": "Star Wars (film)",
        "file_title": "File:StarWarsMoviePoster1977.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/8/87/StarWarsMoviePoster1977.jpg",
    },
    {
        "poster_id": 10,
        "poster_name": "Star Wars: Episode V - The Empire Strikes Back (1980)",
        "page_title": "The Empire Strikes Back",
        "file_title": "File:The Empire Strikes Back (1980 film).jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/3/3f/The_Empire_Strikes_Back_%281980_film%29.jpg",
    },
    {
        "poster_id": 14,
        "poster_name": "Harry Potter and the Chamber of Secrets (2002)",
        "page_title": "Harry Potter and the Chamber of Secrets (film)",
        "file_title": "File:Harry Potter and the Chamber of Secrets movie.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/c/c0/Harry_Potter_and_the_Chamber_of_Secrets_movie.jpg",
    },
    {
        "poster_id": 15,
        "poster_name": "Back to the Future (1985)",
        "page_title": "Back to the Future",
        "file_title": "File:Back to the Future.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/d/d2/Back_to_the_Future.jpg",
    },
    {
        "poster_id": 17,
        "poster_name": "The Lord of the Rings: The Return of the King (2003)",
        "page_title": "The Lord of the Rings: The Return of the King",
        "file_title": "File:Lord Rings Return King.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/4/48/Lord_Rings_Return_King.jpg",
    },
    {
        "poster_id": 19,
        "poster_name": "The Lord of the Rings: The Fellowship of the Ring (2001)",
        "page_title": "The Lord of the Rings: The Fellowship of the Ring",
        "file_title": "File:Lord Rings Fellowship Ring.jpg",
        "image_url": "https://upload.wikimedia.org/wikipedia/en/f/fb/Lord_Rings_Fellowship_Ring.jpg",
    },
]


DEFAULT_OUT_DIR = Path("data/modal_aphasia_posters/gt_posters_wikipedia_20260531")
USER_AGENT = "AAAI2027-entropy-research/0.1 (local poster GT comparison)"
SSL_CONTEXT = ssl._create_unverified_context()


def slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def api_query(params: dict[str, str]) -> dict:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45, context=SSL_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=45, context=SSL_CONTEXT) as response:
                path.write_bytes(response.read())
            return
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code != 429:
                raise
            time.sleep(10 * (attempt + 1))
        except urllib.error.URLError as exc:
            last_exc = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download after retries: {url}") from last_exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    downloaded_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for poster in POSTERS:
        image_url = poster["image_url"]
        ext = Path(urllib.parse.urlparse(image_url).path).suffix.lower() or ".jpg"
        local_name = f"poster_{poster['poster_id']:02d}_{slug(poster['poster_name'])}{ext}"
        local_path = out_dir / local_name
        download(image_url, local_path)
        time.sleep(1)
        with Image.open(local_path) as image:
            width, height = image.size
        file_page = "https://en.wikipedia.org/wiki/" + poster["file_title"].replace(" ", "_")
        manifest.append(
            {
                **poster,
                "source_page_url": "https://en.wikipedia.org/wiki/" + poster["page_title"].replace(" ", "_"),
                "file_page_url": file_page,
                "image_url": image_url,
                "mime": mimetypes.guess_type(str(local_path))[0] or "",
                "width": width,
                "height": height,
                "sha1": "",
                "downloaded_at": downloaded_at,
                "local_path": str(local_path),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[INFO] downloaded {len(manifest)} poster files to {out_dir}")


if __name__ == "__main__":
    main()
