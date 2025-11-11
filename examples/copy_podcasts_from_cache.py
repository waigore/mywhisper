#!/usr/bin/env python3
"""
copy_podcasts_from_cache.py
Copy downloaded Podcasts.app episodes to a user-chosen folder.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import plistlib
import sys
import re
import os
import subprocess
import sqlite3
from functools import lru_cache
from typing import Optional, Iterable, Dict, Any

# ----------------------------------------------------------------------
# Configurable paths
# ----------------------------------------------------------------------
CACHE_ROOT = Path(
    "~/Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Library/Cache"
).expanduser()

DEFAULT_OUTPUT_DIR = Path("~/Documents/Podcasts")
ENV_OUTPUT_DIR = os.getenv("PODCASTS_DIR")
if ENV_OUTPUT_DIR:
    DEFAULT_OUTPUT_DIR = Path(ENV_OUTPUT_DIR)

DB_PATH = Path(
    os.getenv(
        "PODCASTS_DB",
        "~/Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Documents/MTLibrary.sqlite",
    )
).expanduser()

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".aac", ".wav", ".mp4"}

# ----------------------------------------------------------------------
def sanitize(name: str) -> str:
    """Make a string safe for filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", name.strip())

# ----------------------------------------------------------------------
def _find_audio_file(container: Path) -> Optional[Path]:
    """Return the first audio file inside `container` or the file itself."""
    if container.is_file() and container.suffix.lower() in AUDIO_EXTENSIONS:
        return container

    if container.is_dir():
        for candidate in sorted(container.iterdir()):
            if candidate.suffix.lower() in AUDIO_EXTENSIONS:
                return candidate

    return None


def _first_non_empty(candidates: Iterable[Optional[str]]) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate).strip()
        if text and text.lower() not in {"null", "(null)"}:
            return text
    return ""


def _extract_mdls_metadata(src_audio: Path) -> dict[str, str]:
    """Use macOS Spotlight metadata (`mdls`) to enrich title/author information."""
    try:
        result = subprocess.run(
            [
                "mdls",
                "-name",
                "kMDItemAlbum",
                "-name",
                "kMDItemTitle",
                "-name",
                "kMDItemAuthors",
                str(src_audio),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {}
    except subprocess.CalledProcessError:
        return {}

    meta: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        cleaned = value.strip().strip('"')
        if cleaned.startswith("(") and cleaned.endswith(")"):
            # simple handling for mdls array output
            cleaned = ",".join(
                part.strip().strip('"')
                for part in cleaned.strip("()").split(",")
                if part.strip()
            )
        cleaned = cleaned.strip()
        if cleaned and cleaned.lower() not in {"(null)", "null"}:
            meta[key.strip()] = cleaned
    return meta


@lru_cache(maxsize=128)
def _lookup_episode_metadata(audio_filename: str) -> Optional[Dict[str, Any]]:
    if not DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"  ⚠️  Unable to open podcasts database: {exc}", file=sys.stderr)
        return None

    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT
                e.ZTITLE AS episode_title,
                e.ZAUTHOR AS episode_author,
                e.ZGUID AS episode_guid,
                e.ZPUBDATE AS episode_pubdate,
                e.ZSEASONNUMBER AS season_number,
                e.ZEPISODENUMBER AS episode_number,
                p.ZTITLE AS podcast_title
            FROM ZMTEPISODE e
            LEFT JOIN ZMTPODCAST p ON e.ZPODCAST = p.Z_PK
            WHERE e.ZASSETURL LIKE ?
            ORDER BY e.ZPUBDATE DESC
            LIMIT 1
        """
        cursor = conn.execute(query, (f"%/{audio_filename}",))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        print(f"  ⚠️  Database lookup failed for {audio_filename}: {exc}", file=sys.stderr)
        return None
    finally:
        conn.close()


def _coredata_ts_to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime(2001, 1, 1) + timedelta(seconds=seconds)
    except OverflowError:
        return None


def _strip_suffix(text: str, suffix: str) -> str:
    if suffix and text.lower().endswith(suffix.lower()):
        return text[: -len(suffix)]
    return text


def extract_episode(cache_entry: Path, dest_root: Path) -> bool:
    """Copy one episode (dir or standalone file) to dest_root."""
    print(f"→ Processing {cache_entry}")

    metadata: dict[str, object] = {}

    if cache_entry.is_dir():
        metadata_plist = cache_entry / "metadata.plist"
        if metadata_plist.exists():
            try:
                with metadata_plist.open("rb") as f:
                    metadata = plistlib.load(f)
            except Exception as exc:
                print(
                    f"  ⚠️  Failed to read metadata for {cache_entry.name}: {exc}",
                    file=sys.stderr,
                )
        else:
            print(
                f"  ⚠️  No metadata.plist in {cache_entry.name}; using fallback names.",
                file=sys.stderr,
            )

    src_audio = _find_audio_file(cache_entry)
    if src_audio is None:
        print(f"  ⚠️  No audio in {cache_entry.name}", file=sys.stderr)
        return False

    mdls_meta = _extract_mdls_metadata(src_audio)
    db_meta = _lookup_episode_metadata(src_audio.name)

    # ---- basic fields -------------------------------------------------
    show_title = _first_non_empty(
        (
            metadata.get("podcastTitle"),
            (db_meta or {}).get("podcast_title"),
            mdls_meta.get("kMDItemAlbum"),
            cache_entry.name if cache_entry.is_dir() else None,
            src_audio.parent.name if src_audio.parent != CACHE_ROOT else None,
            "Unknown Show",
        )
    )

    raw_episode_title = _first_non_empty(
        (
            metadata.get("episodeTitle"),
            (db_meta or {}).get("episode_title"),
            mdls_meta.get("kMDItemTitle"),
            src_audio.stem,
        )
    )

    ep_title = _strip_suffix(raw_episode_title, src_audio.suffix)
    pub_date_ts = metadata.get("releaseDate") if metadata else None  # seconds since epoch
    if pub_date_ts is None and db_meta:
        pub_dt = _coredata_ts_to_datetime(db_meta.get("episode_pubdate"))
        if pub_dt:
            pub_date_ts = pub_dt.timestamp()
    guid = _first_non_empty(
        (
            metadata.get("episodeGuid"),
            (db_meta or {}).get("episode_guid"),
        )
    )

    author = _first_non_empty(
        (
            metadata.get("author"),
            (db_meta or {}).get("episode_author"),
            mdls_meta.get("kMDItemAuthors"),
        )
    )

    if db_meta:
        print(
            "  ↳ Enriched with Podcasts.db metadata: "
            f"show='{db_meta.get('podcast_title')}', title='{db_meta.get('episode_title')}'"
        )
    elif mdls_meta:
        print(
            f"  ↳ Enriched with mdls metadata: "
            f"Album='{mdls_meta.get('kMDItemAlbum')}', Title='{mdls_meta.get('kMDItemTitle')}'"
        )

    # ---- build destination path ---------------------------------------
    pub_str = ""
    if pub_date_ts:
        try:
            pub_dt = datetime.fromtimestamp(float(pub_date_ts))
            pub_str = pub_dt.strftime("%Y-%m-%d")
        except Exception:
            pub_str = ""

    if not pub_str:
        try:
            pub_dt = datetime.fromtimestamp(src_audio.stat().st_mtime)
            pub_str = pub_dt.strftime("%Y-%m-%d")
        except Exception:
            pub_str = ""

    safe_show = sanitize(show_title)
    safe_ep = sanitize(ep_title)
    safe_author = sanitize(author) if author else ""

    # Example:  "My Show/2024-05-12 - Episode Name.m4a"
    dest_dir = dest_root / safe_show
    dest_dir.mkdir(parents=True, exist_ok=True)

    base_parts = []
    if pub_str:
        base_parts.append(pub_str)
    numbering_tokens: list[str] = []
    if db_meta:
        season = db_meta.get("season_number")
        episode_number = db_meta.get("episode_number")
        if isinstance(season, (int, float)) and season > 0:
            numbering_tokens.append(f"S{int(season):02d}")
        if isinstance(episode_number, (int, float)) and episode_number > 0:
            numbering_tokens.append(f"E{int(episode_number):02d}")

    title_segment = safe_ep
    if numbering_tokens:
        title_segment = " ".join(numbering_tokens) + " - " + safe_ep

    base_parts.append(title_segment)
    if safe_author:
        base_parts.append(f"({safe_author})")

    base_name = " - ".join(part for part in base_parts if part)
    dest_path = dest_dir / f"{base_name}{src_audio.suffix}"

    # avoid overwriting existing files
    counter = 1
    original_dest = dest_path
    while dest_path.exists():
        dest_path = dest_dir / f"{base_name} ({counter}){src_audio.suffix}"
        counter += 1

    # ---- copy ---------------------------------------------------------
    print(f"  ↳ Copying audio → {dest_path}")
    shutil.copy2(src_audio, dest_path)

    # optional: store the GUID in a side-car file for later dedup
    if guid:
        sidecar = dest_path.with_suffix(".guid")
        sidecar.write_text(guid)
        print(f"  ↳ Wrote GUID sidecar → {sidecar}")

    return True

# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract downloaded Podcasts.app episodes to a folder."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Destination folder (will be created if needed)",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move instead of copy (deletes original cache files)",
    )
    args = parser.parse_args()

    dest_root: Path = Path(args.output_dir).expanduser()
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_root = dest_root.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    if not CACHE_ROOT.is_dir():
        print(f"❌ Podcasts cache not found at {CACHE_ROOT}", file=sys.stderr)
        sys.exit(1)

    print(f"Cache root: {CACHE_ROOT}")
    if ENV_OUTPUT_DIR:
        print(
            f"Destination root: {dest_root} (from PODCASTS_DIR environment variable)"
        )
    else:
        print(f"Destination root: {dest_root}")

    episode_entries: list[Path] = []
    for entry in CACHE_ROOT.iterdir():
        if entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
            episode_entries.append(entry)
        elif entry.is_dir():
            if any(
                child.suffix.lower() in AUDIO_EXTENSIONS for child in entry.iterdir()
            ) or (entry / "metadata.plist").exists():
                episode_entries.append(entry)

    print(f"Found {len(episode_entries)} cached items…")

    for ep_entry in episode_entries:
        try:
            copied = extract_episode(ep_entry, dest_root)
            if copied and args.move:
                if ep_entry.is_dir():
                    shutil.rmtree(ep_entry)  # delete after successful copy
                else:
                    ep_entry.unlink()
        except Exception as e:
            print(f"  ⚠️  Failed {ep_entry.name}: {e}", file=sys.stderr)

    print("✅ Done!")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()