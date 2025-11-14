from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from .models import PipelineCheckpoint

SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    episode_id TEXT NOT NULL,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    details_json TEXT NOT NULL,
    artefact_paths_json TEXT NOT NULL,
    elapsed REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (episode_id, step)
);

CREATE INDEX IF NOT EXISTS idx_pipeline_checkpoints_status
ON pipeline_checkpoints(status);
"""


class CheckpointStore:
    """
    Persist pipeline checkpoints to an on-disk SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def upsert(self, checkpoint: PipelineCheckpoint) -> None:
        payload = self._serialize(checkpoint)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_checkpoints (
                    episode_id,
                    step,
                    status,
                    stage,
                    message,
                    payload_json,
                    details_json,
                    artefact_paths_json,
                    elapsed,
                    updated_at
                ) VALUES (
                    :episode_id,
                    :step,
                    :status,
                    :stage,
                    :message,
                    :payload_json,
                    :details_json,
                    :artefact_paths_json,
                    :elapsed,
                    :updated_at
                )
                ON CONFLICT(episode_id, step) DO UPDATE SET
                    status=excluded.status,
                    stage=excluded.stage,
                    message=excluded.message,
                    payload_json=excluded.payload_json,
                    details_json=excluded.details_json,
                    artefact_paths_json=excluded.artefact_paths_json,
                    elapsed=excluded.elapsed,
                    updated_at=excluded.updated_at
                """,
                payload,
            )

    def get_step(self, episode_id: str, step: str) -> Optional[PipelineCheckpoint]:
        query = """
            SELECT episode_id, step, status, stage, message, payload_json, details_json,
                   artefact_paths_json, elapsed, updated_at
            FROM pipeline_checkpoints
            WHERE episode_id = :episode_id AND step = :step
            LIMIT 1
        """
        with self._connect() as conn:
            cursor = conn.execute(query, {"episode_id": episode_id, "step": step})
            row = cursor.fetchone()
        return self._deserialize(row) if row else None

    def get_episode(self, episode_id: str) -> Iterable[PipelineCheckpoint]:
        query = """
            SELECT episode_id, step, status, stage, message, payload_json, details_json,
                   artefact_paths_json, elapsed, updated_at
            FROM pipeline_checkpoints
            WHERE episode_id = :episode_id
            ORDER BY updated_at ASC
        """
        with self._connect() as conn:
            rows = conn.execute(query, {"episode_id": episode_id}).fetchall()
        return [self._deserialize(row) for row in rows]

    def delete_episode(self, episode_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM pipeline_checkpoints WHERE episode_id = :episode_id",
                {"episode_id": episode_id},
            )

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _serialize(checkpoint: PipelineCheckpoint) -> Dict[str, object]:
        return {
            "episode_id": checkpoint.episode_id,
            "step": checkpoint.step,
            "status": checkpoint.status,
            "stage": checkpoint.stage,
            "message": checkpoint.message,
            "payload_json": json.dumps(checkpoint.payload),
            "details_json": json.dumps(checkpoint.details),
            "artefact_paths_json": json.dumps(checkpoint.artefact_paths),
            "elapsed": checkpoint.elapsed,
            "updated_at": checkpoint.updated_at.isoformat(),
        }

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> PipelineCheckpoint:
        return PipelineCheckpoint(
            episode_id=row["episode_id"],
            step=row["step"],
            status=row["status"],
            stage=row["stage"],
            message=row["message"],
            payload=json.loads(row["payload_json"]),
            details=json.loads(row["details_json"]),
            artefact_paths=json.loads(row["artefact_paths_json"]),
            elapsed=row["elapsed"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

