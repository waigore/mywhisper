from __future__ import annotations

from pathlib import Path

import pytest

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


def test_checkpoint_store_delete_episode(tmp_path):
    db_path = tmp_path / "myw.db"
    store = CheckpointStore(db_path)
    adapter = PipelineEventAdapter(store, episode_id="episode-1")

    event = PipelineEvent(
        stage="start",
        message="Test",
        step_name="transcribe",
        episode_id="episode-1",
        payload={},
        artefact_paths={},
        checkpoint={"status": "started", "step": "transcribe"},
        elapsed=0.1,
    )

    adapter.process(event)
    assert store.get_step("episode-1", "transcribe") is not None

    store.delete_episode("episode-1")
    assert store.get_step("episode-1", "transcribe") is None
    assert len(list(store.get_episode("episode-1"))) == 0


def test_checkpoint_store_pipeline_status(tmp_path):
    db_path = tmp_path / "myw.db"
    store = CheckpointStore(db_path)

    # Test setting pipeline status
    store.set_pipeline_status(
        episode_id="episode-1",
        pipeline_id="pipeline-1",
        status="in_progress",
        current_step="transcribe",
        last_completed_step="diarize",
        progress=0.5,
        remarks="Processing",
    )

    # Test getting pipeline status
    status = store.get_pipeline_status("episode-1")
    assert status is not None
    assert status["episode_id"] == "episode-1"
    assert status["pipeline_id"] == "pipeline-1"
    assert status["status"] == "in_progress"
    assert status["current_step"] == "transcribe"
    assert status["last_completed_step"] == "diarize"
    assert status["progress"] == 0.5
    assert status["remarks"] == "Processing"

    # Test getting non-existent status
    assert store.get_pipeline_status("non-existent") is None

    # Test updating pipeline status
    store.set_pipeline_status(
        episode_id="episode-1",
        pipeline_id="pipeline-1",
        status="completed",
        current_step=None,
        progress=1.0,
    )
    status = store.get_pipeline_status("episode-1")
    assert status["status"] == "completed"
    assert status["progress"] == 1.0


def test_checkpoint_store_delete_pipeline(tmp_path):
    db_path = tmp_path / "myw.db"
    store = CheckpointStore(db_path)
    
    # Create adapter with pipeline_id
    adapter1 = PipelineEventAdapter(store, episode_id="episode-1", pipeline_id="pipeline-1")
    adapter2 = PipelineEventAdapter(store, episode_id="episode-1", pipeline_id="pipeline-2")

    # Create some checkpoints with pipeline-1
    event1 = PipelineEvent(
        stage="start",
        message="Test 1",
        step_name="step1",
        episode_id="episode-1",
        payload={},
        artefact_paths={},
        checkpoint={"status": "started", "step": "step1"},
        elapsed=0.1,
    )
    adapter1.process(event1)

    event2 = PipelineEvent(
        stage="start",
        message="Test 2",
        step_name="step2",
        episode_id="episode-1",
        payload={},
        artefact_paths={},
        checkpoint={"status": "started", "step": "step2"},
        elapsed=0.1,
    )
    adapter1.process(event2)

    # Create checkpoint with pipeline-2 (different step to avoid conflicts)
    event3 = PipelineEvent(
        stage="start",
        message="Test 3",
        step_name="step3",
        episode_id="episode-1",
        payload={},
        artefact_paths={},
        checkpoint={"status": "started", "step": "step3"},
        elapsed=0.1,
    )
    adapter2.process(event3)

    # Set pipeline status
    store.set_pipeline_status(
        episode_id="episode-1",
        pipeline_id="pipeline-1",
        status="in_progress",
    )

    # Verify checkpoints exist
    assert store.get_step("episode-1", "step1") is not None
    assert store.get_step("episode-1", "step2") is not None
    assert store.get_step("episode-1", "step3") is not None

    # Delete pipeline with specific pipeline_id (should only delete pipeline-1 checkpoints)
    store.delete_pipeline("episode-1", pipeline_id="pipeline-1")
    # pipeline-1 checkpoints should be deleted
    assert store.get_step("episode-1", "step1") is None
    assert store.get_step("episode-1", "step2") is None
    # pipeline-2 checkpoint should still exist
    assert store.get_step("episode-1", "step3") is not None
    # Pipeline status is deleted regardless of pipeline_id filter
    assert store.get_pipeline_status("episode-1") is None
    
    # Set status again for next test
    store.set_pipeline_status(
        episode_id="episode-1",
        pipeline_id="pipeline-2",
        status="in_progress",
    )
    
    # Delete pipeline without pipeline_id (should delete all remaining checkpoints and status)
    store.delete_pipeline("episode-1")
    # All checkpoints should be gone now
    assert store.get_step("episode-1", "step3") is None
    assert store.get_pipeline_status("episode-1") is None


def test_checkpoint_store_schema_migration(tmp_path):
    """Test that schema migration handles existing pipeline_id column gracefully"""
    db_path = tmp_path / "myw.db"
    
    # Create store first time (creates schema)
    store1 = CheckpointStore(db_path)
    
    # Create store second time (should handle existing pipeline_id column)
    store2 = CheckpointStore(db_path)
    
    # Should work without errors
    assert store2.get_step("test", "test") is None


def test_pipeline_event_adapter_requires_episode_id(tmp_path):
    """Test that PipelineEventAdapter raises ValueError when episode_id is missing"""
    db_path = tmp_path / "myw.db"
    store = CheckpointStore(db_path)
    adapter = PipelineEventAdapter(store)  # No episode_id provided
    
    event = PipelineEvent(
        stage="start",
        message="Test",
        step_name="test",
        episode_id=None,  # No episode_id
        payload={},
        artefact_paths={},
        checkpoint={},
        elapsed=0.1,
    )
    
    with pytest.raises(ValueError, match="episode_id must be provided"):
        adapter.process(event)

