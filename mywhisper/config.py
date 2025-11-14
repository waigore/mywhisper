"""
Configuration helpers for mywhisper pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_DATA_ROOT = Path(
    os.getenv("MYWHISPER_DATA_ROOT", Path("data"))
).expanduser().resolve()


def ensure_data_subdir(name: str, root: Optional[Path] = None) -> Path:
    """
    Ensure a subdirectory exists under the configured data root.

    Parameters
    ----------
    name:
        Relative directory name (e.g. ``\"transcripts\"``).
    root:
        Optional override for the base directory. Defaults to
        :data:`DEFAULT_DATA_ROOT`.
    """

    base = (root or DEFAULT_DATA_ROOT).resolve()
    target = base / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def generate_artefact_key() -> str:
    """
    Generate an eight-character artefact key for temporary files.
    """

    import secrets

    return secrets.token_hex(4).upper()


def derive_episode_key(episode_id: str) -> str:
    """
    Derive a deterministic eight-digit episode key from an identifier.
    """

    if not episode_id:
        raise ValueError("episode_id must be a non-empty string.")

    import hashlib

    digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:6], "big") % 10**8
    return f"{value:08d}"


def resolve_data_root(path: Optional[Path] = None) -> Path:
    """
    Resolve the data root, defaulting to :data:`DEFAULT_DATA_ROOT`.
    """

    return (path or DEFAULT_DATA_ROOT).resolve()


