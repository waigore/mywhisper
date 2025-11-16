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

-- Add pipeline_id column if missing
"""

STATUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_status (
    episode_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    status TEXT NOT NULL,
    current_step TEXT,
    last_completed_step TEXT,
    progress REAL,
    remarks TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pipeline_status_pipeline
ON pipeline_status(pipeline_id);
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
                    pipeline_id,
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
                    :pipeline_id,
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
                    pipeline_id=COALESCE(excluded.pipeline_id, pipeline_checkpoints.pipeline_id),
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
            SELECT pipeline_id, episode_id, step, status, stage, message, payload_json, details_json,
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
            SELECT pipeline_id, episode_id, step, status, stage, message, payload_json, details_json,
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
            # Attempt to add pipeline_id column if it doesn't exist
            try:
                conn.execute("ALTER TABLE pipeline_checkpoints ADD COLUMN pipeline_id TEXT")
            except sqlite3.OperationalError:
                pass
            conn.executescript(STATUS_SCHEMA)

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
            "pipeline_id": checkpoint.pipeline_id,
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
            pipeline_id=row["pipeline_id"] if "pipeline_id" in row.keys() else None,
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

    # ----------------------------
    # Pipeline status operations
    # ----------------------------
    def get_pipeline_status(self, episode_id: str) -> Optional[Dict[str, object]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT episode_id, pipeline_id, status, current_step, last_completed_step,
                       progress, remarks, updated_at
                FROM pipeline_status
                WHERE episode_id = :episode_id
                LIMIT 1
                """,
                {"episode_id": episode_id},
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def set_pipeline_status(
        self,
        episode_id: str,
        pipeline_id: str,
        status: str,
        *,
        current_step: Optional[str] = None,
        last_completed_step: Optional[str] = None,
        progress: Optional[float] = None,
        remarks: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_status (
                    episode_id, pipeline_id, status, current_step, last_completed_step,
                    progress, remarks, updated_at
                ) VALUES (
                    :episode_id, :pipeline_id, :status, :current_step, :last_completed_step,
                    :progress, :remarks, :updated_at
                )
                ON CONFLICT(episode_id) DO UPDATE SET
                    pipeline_id=excluded.pipeline_id,
                    status=excluded.status,
                    current_step=excluded.current_step,
                    last_completed_step=COALESCE(excluded.last_completed_step, pipeline_status.last_completed_step),
                    progress=excluded.progress,
                    remarks=excluded.remarks,
                    updated_at=excluded.updated_at
                """,
                {
                    "episode_id": episode_id,
                    "pipeline_id": pipeline_id,
                    "status": status,
                    "current_step": current_step,
                    "last_completed_step": last_completed_step,
                    "progress": progress,
                    "remarks": remarks,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )

    def delete_pipeline(self, episode_id: str, pipeline_id: Optional[str] = None) -> None:
        with self._connect() as conn:
            if pipeline_id:
                conn.execute(
                    "DELETE FROM pipeline_checkpoints WHERE episode_id = :episode_id AND pipeline_id = :pipeline_id",
                    {"episode_id": episode_id, "pipeline_id": pipeline_id},
                )
            else:
                conn.execute(
                    "DELETE FROM pipeline_checkpoints WHERE episode_id = :episode_id",
                    {"episode_id": episode_id},
                )
            conn.execute("DELETE FROM pipeline_status WHERE episode_id = :episode_id", {"episode_id": episode_id})

