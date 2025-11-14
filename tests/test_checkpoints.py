from __future__ import annotations

from pathlib import Path

from mywhisper.checkpoints import CheckpointStore, PipelineEventAdapter
from mywhisper.models import PipelineEvent


def test_checkpoint_store_roundtrip(tmp_path):
    db_path = tmp_path / "myw.db"
    store = CheckpointStore(db_path)
    adapter = PipelineEventAdapter(store, episode_id="episode-1")

    transcript_path = tmp_path / "transcript.json"
    event = PipelineEvent(
        stage="start",
        message="Starting transcription",
        step_name="transcribe",
        episode_id="episode-1",
        payload={"step": "started"},
        artefact_paths={"transcript": transcript_path},
        checkpoint={"status": "started", "step": "transcribe"},
        elapsed=0.1,
    )

    adapter.process(event)

    stored = store.get_step("episode-1", "transcribe")
    assert stored is not None
    assert stored.step == "transcribe"
    assert stored.status == "started"
    assert stored.artefact_paths["transcript"] == str(transcript_path)

    all_steps = list(store.get_episode("episode-1"))
    assert len(all_steps) == 1
    assert all_steps[0].message == "Starting transcription"

