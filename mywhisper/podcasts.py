"""
Podcast catalog management for mywhisper.
"""

from __future__ import annotations

import json
import plistlib
import re
import shutil
import logging
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

from .config import ensure_data_subdir, resolve_data_root
from .config import generate_artefact_key
from .models import PodcastEpisode

LOGGER = logging.getLogger("mywhisper.podcasts")

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    show_title TEXT NOT NULL,
    episode_title TEXT NOT NULL,
    author TEXT,
    guid TEXT,
    published_at TEXT,
    cache_path TEXT,
    audio_path TEXT NOT NULL,
    duration_sec REAL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS artefacts (
    artefact_key TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_show_title ON episodes(show_title);
CREATE INDEX IF NOT EXISTS idx_episodes_guid ON episodes(guid);
CREATE INDEX IF NOT EXISTS idx_episodes_published_at ON episodes(published_at);
"""


AUDIO_EXTENSIONS = {".m4a", ".mp3", ".aac", ".wav", ".mp4"}


def _sanitize(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", text.strip())


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


@lru_cache(maxsize=256)
def _cached_lookup_episode_metadata(db_path: str, audio_filename: str) -> Optional[Dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return None

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        LOGGER.debug("Unable to open podcasts database %s: %s", path, exc)
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
                e.ZITEMDESCRIPTION AS item_description,
                e.ZITEMDESCRIPTIONWITHOUTHTML AS item_description_plain,
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
        LOGGER.debug("Database lookup failed for %s: %s", audio_filename, exc)
        return None
    finally:
        conn.close()


def _lookup_episode_metadata(db_path: Optional[Path], audio_filename: str) -> Optional[Dict[str, Any]]:
    if not db_path:
        return None
    return _cached_lookup_episode_metadata(str(db_path), audio_filename)


def _coredata_ts_to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    try:
        # Core Data timestamps are seconds since 2001-01-01.
        return datetime(2001, 1, 1) + timedelta(seconds=seconds)
    except Exception:
        return None


def _extract_mdls_metadata(src_audio: Path) -> Dict[str, str]:
    """
    Use macOS Spotlight metadata (`mdls`) to populate album/title fields.
    """

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
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    meta: Dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        cleaned = value.strip().strip('"')
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = ",".join(
                part.strip().strip('"')
                for part in cleaned.strip("()").split(",")
                if part.strip()
            )
        cleaned = cleaned.strip()
        if cleaned and cleaned.lower() not in {"(null)", "null"}:
            meta[key.strip()] = cleaned
    return meta


class PodcastCatalog:
    """
    SQLite-backed catalog of podcast episodes.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.data_root = resolve_data_root(data_root)
        catalog_dir = ensure_data_subdir("podcasts", self.data_root)
        self.db_path = (db_path or catalog_dir / "catalog.db").resolve()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_episode(self, episode: PodcastEpisode) -> None:
        metadata = dict(episode.metadata or {})
        if episode.description is not None:
            metadata.setdefault("description", episode.description)
        metadata.setdefault("episode_key", episode.episode_key)
        cache_path = metadata.get("cache_path")

        payload = {
            "id": episode.episode_id,
            "show_title": episode.show_title,
            "episode_title": episode.episode_title,
            "author": episode.author,
            "guid": episode.guid,
            "published_at": _to_iso(episode.published_at),
            "cache_path": cache_path,
            "audio_path": str(episode.source_path),
            "duration_sec": episode.duration_sec,
            "metadata_json": json.dumps(metadata),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes
                (id, show_title, episode_title, author, guid, published_at, cache_path, audio_path, duration_sec, metadata_json)
                VALUES
                (:id, :show_title, :episode_title, :author, :guid, :published_at, :cache_path, :audio_path, :duration_sec, :metadata_json)
                ON CONFLICT(id) DO UPDATE SET
                    show_title=excluded.show_title,
                    episode_title=excluded.episode_title,
                    author=excluded.author,
                    guid=excluded.guid,
                    published_at=excluded.published_at,
                    cache_path=excluded.cache_path,
                    audio_path=excluded.audio_path,
                    duration_sec=excluded.duration_sec,
                    metadata_json=excluded.metadata_json
                """,
                payload,
            )

    def record_artefact(self, episode_id: str, kind: str, path: Path, artefact_key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artefacts (artefact_key, episode_id, kind, path)
                VALUES (?, ?, ?, ?)
                """,
                (artefact_key, episode_id, kind, str(path)),
            )

    def get_episode(self, identifier: str) -> Optional[PodcastEpisode]:
        query = """
            SELECT id, show_title, episode_title, author, guid, published_at, audio_path, duration_sec, metadata_json
            FROM episodes
            WHERE id = :identifier OR guid = :identifier OR audio_path = :identifier
            LIMIT 1
        """
        with self._connect() as conn:
            cursor = conn.execute(query, {"identifier": identifier})
            row = cursor.fetchone()
        if not row:
            return None
        metadata = json.loads(row[8] or "{}")
        description = metadata.get("description")
        published_at = datetime.fromisoformat(row[5]) if row[5] else None
        return PodcastEpisode(
            episode_id=row[0],
            show_title=row[1],
            episode_title=row[2],
            author=row[3],
            guid=row[4],
            published_at=published_at,
            description=description,
            source_path=Path(row[6]),
            duration_sec=row[7],
            metadata=metadata,
        )

    def list_episodes(
        self,
        show_title: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> Iterable[PodcastEpisode]:
        query = """
            SELECT id, show_title, episode_title, author, guid, published_at, audio_path, duration_sec, metadata_json
            FROM episodes
            WHERE (:show_title IS NULL OR show_title = :show_title)
              AND (:since IS NULL OR published_at >= :since)
            ORDER BY published_at DESC
        """
        params = {
            "show_title": show_title,
            "since": _to_iso(since) if since else None,
        }
        with self._connect() as conn:
            for row in conn.execute(query, params):
                metadata = json.loads(row[8] or "{}")
                published_at = datetime.fromisoformat(row[5]) if row[5] else None
                yield PodcastEpisode(
                    episode_id=row[0],
                    show_title=row[1],
                    episode_title=row[2],
                    author=row[3],
                    guid=row[4],
                    published_at=published_at,
                    description=metadata.get("description"),
                    source_path=Path(row[6]),
                    duration_sec=row[7],
                    metadata=metadata,
                )


@dataclass(slots=True)
class EpisodeMetadata:
    """
    Metadata collected from the Apple Podcasts cache.
    """

    cache_entry: Path
    audio_path: Path
    show_title: str
    episode_title: str
    author: Optional[str] = None
    guid: Optional[str] = None
    published_at: Optional[datetime] = None
    description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_episode(self) -> PodcastEpisode:
        metadata = dict(self.extra or {})
        metadata.update(
            {
                "cache_path": str(self.cache_entry),
            }
        )
        if self.description:
            metadata.setdefault("description", self.description)
        return PodcastEpisode(
            episode_id=self.guid or self.audio_path.stem,
            show_title=self.show_title,
            episode_title=self.episode_title,
            author=self.author,
            guid=self.guid,
            published_at=self.published_at,
            description=self.description,
            source_path=self.audio_path,
            metadata=metadata,
        )


class ApplePodcastsImporter:
    """
    Scan the Apple Podcasts cache and register episodes in the catalog.
    """

    def __init__(
        self,
        cache_root: Path,
        catalog: PodcastCatalog,
        output_dir: Optional[Path] = None,
        move: bool = False,
        logger: Optional[logging.Logger] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self.cache_root = cache_root
        self.catalog = catalog
        self.output_dir = (output_dir or ensure_data_subdir("podcasts")).resolve()
        self.move = move
        self.logger = logger or LOGGER
        self.db_path = db_path

    def scan(self) -> Generator[EpisodeMetadata, None, None]:
        if not self.cache_root.exists():
            raise FileNotFoundError(f"Cache root not found at {self.cache_root}")

        entries = sorted(self.cache_root.iterdir())
        for entry in entries:
            metadata = self._load_entry(entry)
            if metadata:
                yield metadata

    def register_in_catalog(self) -> Generator[PodcastEpisode, None, None]:
        for metadata in self.scan():
            try:
                dest_path = self._copy_episode(media=metadata)
            except Exception as exc:
                self.logger.error("Failed to copy %s: %s", metadata.audio_path, exc)
                continue

            episode = metadata.to_episode()
            episode.source_path = dest_path
            duration = self._duration_seconds(dest_path)
            if duration:
                episode.duration_sec = duration

            episode.metadata.update(
                {
                    "original_cache_path": str(metadata.audio_path),
                    "imported_at": datetime.utcnow().isoformat(),
                }
            )

            self.catalog.upsert_episode(episode)
            artefact_key = generate_artefact_key()
            self.catalog.record_artefact(
                episode_id=episode.episode_id,
                kind="audio",
                path=dest_path,
                artefact_key=artefact_key,
            )

            if self.move:
                self._cleanup_source(metadata.cache_entry)

            yield episode

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_entry(self, entry: Path) -> Optional[EpisodeMetadata]:
        audio_path = self._find_audio_file(entry)
        if audio_path is None:
            return None

        plist_data: Dict[str, Any] = {}
        if entry.is_dir():
            plist_path = entry / "metadata.plist"
            if plist_path.exists():
                try:
                    with plist_path.open("rb") as handle:
                        plist_data = plistlib.load(handle)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.debug("Failed to read %s: %s", plist_path, exc)

        mdls_meta = _extract_mdls_metadata(audio_path)
        db_meta = _lookup_episode_metadata(self.db_path, audio_path.name)

        show_title = self._first_non_empty(
            plist_data.get("podcastTitle"),
            (db_meta or {}).get("podcast_title"),
            mdls_meta.get("kMDItemAlbum"),
            entry.name if entry.is_dir() else audio_path.parent.name,
            "Unknown Show",
        )
        episode_title = self._first_non_empty(
            plist_data.get("episodeTitle"),
            (db_meta or {}).get("episode_title"),
            mdls_meta.get("kMDItemTitle"),
            audio_path.stem,
        )
        author = self._first_non_empty(
            plist_data.get("author"),
            plist_data.get("podcastAuthor"),
            (db_meta or {}).get("episode_author"),
            mdls_meta.get("kMDItemAuthors"),
        )
        guid = self._first_non_empty(
            plist_data.get("episodeGuid"),
            plist_data.get("guid"),
            (db_meta or {}).get("episode_guid"),
        )
        published_at = self._parse_timestamp(plist_data.get("releaseDate"))
        if published_at is None and db_meta:
            published_at = _coredata_ts_to_datetime(db_meta.get("episode_pubdate"))

        db_description_plain = (db_meta or {}).get("item_description_plain")
        db_description_html = (db_meta or {}).get("item_description")

        description = self._first_non_empty(
            plist_data.get("episodeDescription"),
            plist_data.get("description"),
            plist_data.get("longDescription"),
            plist_data.get("subtitle"),
            plist_data.get("summary"),
            db_description_plain,
            db_description_html,
        )

        extra = dict(plist_data)
        extra["cache_entry"] = str(entry)
        if db_meta:
            extra.setdefault("podcasts_db", db_meta)
        if mdls_meta:
            extra.setdefault("mdls", mdls_meta)
        if description:
            extra.setdefault("description", description)
        elif db_description_plain:
            extra.setdefault("description", db_description_plain)
        elif db_description_html:
            extra.setdefault("description", db_description_html)

        return EpisodeMetadata(
            cache_entry=entry,
            audio_path=audio_path,
            show_title=show_title,
            episode_title=episode_title,
            author=author,
            guid=guid,
            published_at=published_at,
            description=description or None,
            extra=extra,
        )

    def _copy_episode(self, media: EpisodeMetadata) -> Path:
        destination_dir = self.output_dir / _sanitize(media.show_title or "Unknown Show")
        destination_dir.mkdir(parents=True, exist_ok=True)

        published_part = (
            media.published_at.strftime("%Y-%m-%d") if media.published_at else ""
        )
        filename_parts = [part for part in [published_part, _sanitize(media.episode_title)] if part]
        if media.author:
            filename_parts.append(f"({ _sanitize(media.author) })")

        base_name = " - ".join(filename_parts) if filename_parts else _sanitize(media.audio_path.stem)
        dest_path = self._unique_destination(destination_dir, base_name, media.audio_path.suffix)

        if self.move:
            shutil.move(str(media.audio_path), str(dest_path))
        else:
            shutil.copy2(media.audio_path, dest_path)

        return dest_path

    def _unique_destination(self, directory: Path, base_name: str, suffix: str) -> Path:
        counter = 0
        candidate = directory / f"{base_name}{suffix}"
        while candidate.exists():
            counter += 1
            candidate = directory / f"{base_name} ({counter}){suffix}"
        return candidate

    def _duration_seconds(self, audio_path: Path) -> Optional[float]:
        try:
            import torchaudio  # type: ignore
        except Exception:  # pragma: no cover
            return None

        try:
            info = torchaudio.info(str(audio_path))
        except Exception:  # pragma: no cover - best effort
            return None
        if not info.num_frames or not info.sample_rate:
            return None
        return info.num_frames / info.sample_rate

    def _cleanup_source(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)

    def _find_audio_file(self, entry: Path) -> Optional[Path]:
        if entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
            return entry
        if entry.is_dir():
            for candidate in entry.iterdir():
                if candidate.suffix.lower() in AUDIO_EXTENSIONS:
                    return candidate
        return None

    @staticmethod
    def _first_non_empty(*values: Optional[str]) -> str:
        for value in values:
            if not value:
                continue
            text = str(value).strip()
            if text and text.lower() not in {"null", "(null)"}:
                return text
        return ""

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        try:
            return datetime.fromtimestamp(seconds)
        except (OverflowError, OSError):
            return None

