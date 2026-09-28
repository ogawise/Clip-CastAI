"""
ClipCast AI - Component 1: Content Sourcing Module.

Searches Pexels for videos in a category, keeps the ones that meet a minimum
duration and resolution, downloads them to ./downloads/, and records each
clip in a local SQLite database (clips.db).

Usage:
    python source.py --category dogs --count 5
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# --- Configuration ----------------------------------------------------------

MIN_DURATION_SECONDS = 5
# Checked against the video's SHORT side so portrait and landscape are treated
# the same: 1080 means "at least 1080p" (1920x1080 and 1080x1920 both pass).
MIN_SHORT_SIDE_PX = 1080

# Paths are anchored to this file, not the current directory, so the script
# behaves the same no matter where you run it from.
BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DB_PATH = BASE_DIR / "clips.db"

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
PEXELS_LICENSE = "Pexels License"  # every Pexels video uses the same license
PEXELS_PER_PAGE = 80  # API maximum
MAX_PAGES = 5  # safety cap per run; Pexels allows 200 requests/hour

REQUEST_TIMEOUT = 15  # seconds to wait on an API call
DOWNLOAD_TIMEOUT = 60  # seconds of no data before a download gives up

log = logging.getLogger("clipcast.source")


# --- Database ---------------------------------------------------------------

def init_db(db_path):
    """Open the database, creating the clips table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clips (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            source        TEXT NOT NULL,
            clip_id       TEXT NOT NULL,
            license_type  TEXT NOT NULL,
            category      TEXT NOT NULL,
            downloaded_at TEXT NOT NULL,
            local_path    TEXT NOT NULL,
            UNIQUE (source, clip_id)
        )
        """
    )
    conn.commit()
    return conn


def is_already_downloaded(conn, source, clip_id):
    row = conn.execute(
        "SELECT 1 FROM clips WHERE source = ? AND clip_id = ?",
        (source, str(clip_id)),
    ).fetchone()
    return row is not None


def record_clip(conn, source, clip_id, license_type, category, local_path):
    conn.execute(
        """
        INSERT INTO clips (source, clip_id, license_type, category, downloaded_at, local_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            str(clip_id),
            license_type,
            category,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            str(local_path),
        ),
    )
    conn.commit()


# --- Pexels -----------------------------------------------------------------

def best_pexels_rendition(video):
    """Return the highest-resolution MP4 in video["video_files"], or None.

    Pexels doesn't sort video_files and "quality" can be null, so we compare
    actual pixel counts instead of trusting the label.
    """
    files = [
        f
        for f in video.get("video_files", [])
        if f.get("file_type") == "video/mp4"
        and f.get("width")
        and f.get("height")
        and f.get("link")
    ]
    if not files:
        return None
    return max(files, key=lambda f: f["width"] * f["height"])


def passes_filters(duration, width, height):
    return (
        duration >= MIN_DURATION_SECONDS
        and min(width, height) >= MIN_SHORT_SIDE_PX
    )


def search_pexels(api_key, category):
    """Yield Pexels videos for `category` that pass the filters, page by page.

    This is a generator: it only fetches the next page when the caller asks for
    more clips, so a run that needs 5 clips usually makes a single API call.

    Each yielded dict has the same keys regardless of provider, so a future
    search_pixabay() just needs to yield the same shape:
        {"source", "clip_id", "license_type", "download_url",
         "width", "height", "duration"}
    """
    headers = {"Authorization": api_key}  # raw key, no "Bearer" prefix

    for page in range(1, MAX_PAGES + 1):
        params = {"query": category, "per_page": PEXELS_PER_PAGE, "page": page}
        try:
            resp = requests.get(
                PEXELS_SEARCH_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 401:
                log.error("Pexels rejected the API key (401). Check PEXELS_API_KEY in .env.")
                return
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            # ValueError covers a response body that isn't valid JSON.
            log.warning("Pexels search failed on page %d: %s", page, exc)
            return

        videos = data.get("videos", [])
        if page == 1:
            log.info("Pexels reports %d total results for %r", data.get("total_results", 0), category)

        matches = []
        for video in videos:
            rendition = best_pexels_rendition(video)
            if rendition is None:
                continue
            duration = video.get("duration") or 0
            if not passes_filters(duration, rendition["width"], rendition["height"]):
                continue
            matches.append(
                {
                    "source": "pexels",
                    "clip_id": video["id"],
                    "license_type": PEXELS_LICENSE,
                    "download_url": rendition["link"],
                    "width": rendition["width"],
                    "height": rendition["height"],
                    "duration": duration,
                }
            )

        log.info("Page %d: %d of %d videos passed filters", page, len(matches), len(videos))
        yield from matches

        if not data.get("next_page"):
            return  # no more results on Pexels' side

    log.warning("Stopped after MAX_PAGES (%d) pages of results.", MAX_PAGES)


# --- Downloading ------------------------------------------------------------

def download_file(url, dest):
    """Stream `url` to `dest`. Returns True on success, False on failure.

    Writes to a .part file first and renames at the end, so an interrupted
    download never leaves behind something that looks like a finished clip.
    """
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        tmp.replace(dest)
        return True
    except (requests.RequestException, OSError) as exc:
        log.warning("Download failed for %s: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return False


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download royalty-free video clips from Pexels.")
    parser.add_argument("--category", required=True, help='search term, e.g. "dogs"')
    parser.add_argument("--count", type=int, default=5, help="number of NEW clips to download (default 5)")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        log.error("PEXELS_API_KEY is not set. Add it to .env (see .env.example).")
        return 1

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    conn = init_db(DB_PATH)
    downloaded = 0

    try:
        for clip in search_pexels(api_key, args.category):
            label = f"{clip['source']} {clip['clip_id']}"
            if is_already_downloaded(conn, clip["source"], clip["clip_id"]):
                log.info("Skipping %s (already downloaded)", label)
                continue

            dest = DOWNLOAD_DIR / f"{clip['source']}_{clip['clip_id']}.mp4"
            log.info(
                "Downloading %s (%dx%d, %ds)...",
                label, clip["width"], clip["height"], clip["duration"],
            )
            if not download_file(clip["download_url"], dest):
                continue

            try:
                record_clip(
                    conn, clip["source"], clip["clip_id"], clip["license_type"],
                    args.category, dest,
                )
            except sqlite3.Error as exc:
                log.warning("Downloaded %s but could not record it in the DB: %s", label, exc)
                continue

            downloaded += 1
            log.info("Saved %s -> %s (%d/%d)", label, dest.name, downloaded, args.count)
            if downloaded >= args.count:
                break
    finally:
        conn.close()

    if downloaded == 0:
        log.warning("No clips downloaded for %r.", args.category)
    elif downloaded < args.count:
        log.warning(
            "Only downloaded %d of %d requested clips for %r (not enough matches).",
            downloaded, args.count, args.category,
        )
    else:
        log.info("Done: downloaded %d clips for %r.", downloaded, args.category)
    return 0


if __name__ == "__main__":
    sys.exit(main())
