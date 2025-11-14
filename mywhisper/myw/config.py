from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ..config import ensure_data_subdir, resolve_data_root

DEFAULT_PODCAST_CACHE = Path(
    "~/Library/Group Containers/243LU875E5.groups.com.apple.podcasts/Library/Cache"
).expanduser()


class ConfigError(RuntimeError):
    """Raised when configuration is invalid or incomplete."""


@dataclass(slots=True)
class MywConfig:
    data_dir: Path
    db_path: Path
    podcast_cache_path: Path
    podcast_db_path: Path
    log_level: str
    whisper_model: Optional[str] = None
    device: Optional[str] = None
    ollama_model: str = "llama3"
    spacy_model: str = "en_core_web_sm"
    hf_token: Optional[str] = None


def load_config(env_path: Optional[Path] = None) -> MywConfig:
    """
    Load application configuration from environment variables and defaults.
    """

    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    data_dir = _resolve_path(os.getenv("MYW_DATA_DIR"))
    data_root = resolve_data_root(data_dir)

    db_path = _resolve_path(os.getenv("MYW_DB_PATH"), fallback=data_root / "myw.db")
    podcast_cache = _resolve_path(
        os.getenv("MYW_PODCAST_CACHE_PATH"),
        fallback=DEFAULT_PODCAST_CACHE,
    )
    podcast_db_env = os.getenv("MYW_PODCAST_DB_PATH") or os.getenv("PODCASTS_DB")
    podcast_db = _resolve_path(
        podcast_db_env,
        fallback=_default_podcasts_db_for_cache(podcast_cache),
    )

    if not podcast_cache.exists():
        raise ConfigError(
            f"Apple Podcasts cache not found at {podcast_cache}. "
            "Set MYW_PODCAST_CACHE_PATH to a valid directory."
        )

    ensure_data_subdir("logs", data_root)

    whisper_model = os.getenv("MYW_WHISPER_MODEL")
    if whisper_model:
        whisper_model = str(Path(whisper_model).expanduser().resolve())

    device = os.getenv("MYW_DEVICE")

    return MywConfig(
        data_dir=data_root,
        db_path=db_path,
        podcast_cache_path=podcast_cache,
        podcast_db_path=podcast_db,
        log_level=os.getenv("MYW_LOG_LEVEL", "INFO"),
        whisper_model=whisper_model,
        device=device,
        ollama_model=os.getenv("MYW_OLLAMA_MODEL", "llama3"),
        spacy_model=os.getenv("MYW_SPACY_MODEL", "en_core_web_sm"),
        hf_token=os.getenv("MYW_HF_TOKEN"),
    )


def _resolve_path(value: Optional[str], fallback: Optional[Path] = None) -> Path:
    if value:
        path = Path(value).expanduser()
    elif fallback is not None:
        path = fallback
    else:
        path = Path.cwd()
    return path.resolve()


def _default_podcasts_db_for_cache(cache_path: Path) -> Path:
    """
    Return the default Podcasts SQLite database path for a given cache root.
    """

    # Default layout on macOS: cache root under `.../Library/Cache`, database under sibling `Documents/MTLibrary.sqlite`.
    documents_dir = cache_path.parent.parent / "Documents"
    return (documents_dir / "MTLibrary.sqlite").resolve()

