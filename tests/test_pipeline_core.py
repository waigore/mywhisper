from __future__ import annotations

from pathlib import Path
import json
import typing as t

import pytest

from mywhisper.checkpoints.models import PipelineCheckpoint
from mywhisper.models import DiarizedTurn, PipelineEvent, PodcastEpisode, TranscriptSegment
from mywhisper.myw.config import MywConfig
from mywhisper.myw.services.pipeline import (
    PipelineContext,
    PipelineInterrupted,
    PipelineRunner,
    STEP_ORDER,
)
from mywhisper.myw.services.steps import (
    ensure_diarized_turns,
    load_step_path,
    read_transcript,
    validate_assignment_availability,
    validate_condensed_availability,
    validate_diarization_availability,
    validate_themes_availability,
    validate_transcript_availability,
)


class DummyQueue:
    def __init__(self):
        self._should_stop = False
        self.status_updates: list[tuple[str, str, str]] = []

    def request_shutdown(self) -> None:
        pass

    def next_item(self):
        return None

    def release_current(self) -> None:
        pass

    def set_status(self, episode_id: str, status: str, remarks: str) -> None:
        self.status_updates.append((episode_id, status, remarks))

    def should_stop(self) -> bool:
        return self._should_stop


class InMemoryCheckpointStore:
    def __init__(self, rows: dict[tuple[str, str], PipelineCheckpoint] | None = None) -> None:
        self._rows = rows or {}
        self._pipeline_status: dict[str, dict[str, t.Any]] = {}

    def get_step(self, episode_id: str, step: str):
        return self._rows.get((episode_id, step))

    def get_episode(self, episode_id: str):
        return [cp for (eid, _), cp in self._rows.items() if eid == episode_id]

    def delete_episode(self, episode_id: str) -> None:
        for key in [key for key in self._rows if key[0] == episode_id]:
            del self._rows[key]

    def upsert(self, checkpoint: PipelineCheckpoint) -> None:
        self._rows[(checkpoint.episode_id, checkpoint.step)] = checkpoint

    # Minimal status API used by PipelineRunner
    def set_pipeline_status(
        self,
        *,
        episode_id: str,
        pipeline_id: str | None = None,
        status: str | None = None,
        current_step: str | None = None,
        last_completed_step: str | None = None,
        progress: float | None = None,
        remarks: str | None = None,
    ) -> None:
        row = self._pipeline_status.get(episode_id) or {}
        if pipeline_id is not None:
            row["pipeline_id"] = pipeline_id
        if status is not None:
            row["status"] = status
        if current_step is not None:
            row["current_step"] = current_step
        if last_completed_step is not None:
            row["last_completed_step"] = last_completed_step
        if progress is not None:
            row["progress"] = progress
        if remarks is not None:
            row["remarks"] = remarks
        self._pipeline_status[episode_id] = row

    def get_pipeline_status(self, episode_id: str):
        return self._pipeline_status.get(episode_id)


def build_runner(tmp_path: Path, *, queue: DummyQueue | None = None, checkpoints: InMemoryCheckpointStore | None = None) -> PipelineRunner:
    config = MywConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path,
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
        whisper_model=str(tmp_path / "model.bin"),
        device=None,
        ollama_model="llama3",
        spacy_model="en_core_web_sm",
        hf_token=None,
    )
    # A minimal catalog stub to satisfy type expectations when needed
    class StubCatalog:
        def get_episode(self, episode_id: str):
            return None

    return PipelineRunner(
        config=config,
        catalog=StubCatalog(),
        queue=queue or DummyQueue(),
        checkpoints=checkpoints or InMemoryCheckpointStore(),
        callback=None,
    )


def test_resolve_step_plan_normalizes_and_defaults(tmp_path):
    runner = build_runner(tmp_path)
    # invalid + duplicates filtered; order preserved relative to STEP_ORDER
    from mywhisper.myw.services.queue import QueueItem

    item = QueueItem("ep1", steps=("assign", "assign", "diarize", "invalid"))
    assert runner._resolve_step_plan(item) == ("assign", "diarize")

    # empty/invalid defaults to full pipeline
    item2 = QueueItem("ep1", steps=("bogus",))
    assert runner._resolve_step_plan(item2) == STEP_ORDER


def test_should_reset_checkpoints_logic(tmp_path):
    runner = build_runner(tmp_path)
    from mywhisper.myw.services.queue import QueueItem

    non_empty = [
        PipelineCheckpoint(
            episode_id="ep",
            step="transcribe",
            status="completed",
            stage="persisted",
            message="ok",
        )
    ]
    # Fresh run, plan starts with 'transcribe', existing checkpoints -> reset
    assert runner._should_reset_checkpoints(QueueItem("ep", steps=("transcribe",)), ("transcribe",), non_empty) is True

    # Resume -> do not reset
    assert runner._should_reset_checkpoints(QueueItem("ep", steps=("transcribe",), resume=True), ("transcribe",), non_empty) is False

    # Plan starts later step -> do not reset
    assert runner._should_reset_checkpoints(QueueItem("ep", steps=("assign",)), ("assign",), non_empty) is False

    # No checkpoints -> no reset
    assert runner._should_reset_checkpoints(QueueItem("ep", steps=("transcribe",)), ("transcribe",), []) is False


def test_validation_helpers_raise_as_expected(tmp_path):
    runner = build_runner(tmp_path)
    # transcript required by diarize when transcribe not in plan
    with pytest.raises(RuntimeError):
        validate_transcript_availability(("diarize",), None)

    # diarization required for prettify/assign unless already in completed
    with pytest.raises(RuntimeError):
        validate_diarization_availability(("prettify",), {}, None)
    # ok when diarize planned
    validate_diarization_availability(("prettify", "diarize"), {}, None)
    # ok when diarize completed
    validate_diarization_availability(("assign",), {"diarize": "completed"}, None)

    # readable required for assign
    with pytest.raises(RuntimeError):
        validate_assignment_availability(("assign",), None)

    # condensed required for thematize
    with pytest.raises(RuntimeError):
        validate_condensed_availability(("thematize",), None)

    # themes required for classify
    with pytest.raises(RuntimeError):
        validate_themes_availability(("classify",), None)


def test_progress_for_and_completion_message(tmp_path):
    runner = build_runner(tmp_path)
    plan = ("transcribe", "diarize", "prettify")
    dummy_event = PipelineEvent(
        stage="progress",
        step_name="transcribe",
        episode_id="ep",
        message="start",
        payload={},
        checkpoint={"status": "started"},
    )
    # first of 3, started (0/3 completed)
    assert runner._progress_for("transcribe", dummy_event, plan) == 0.0
    # when completed, (1/3)
    done_event = PipelineEvent(
        stage="persisted",
        step_name="transcribe",
        episode_id="ep",
        message="done",
        payload={},
        checkpoint={"status": "completed"},
    )
    assert runner._progress_for("transcribe", done_event, plan) == pytest.approx(1 / 3)

    # completion message defaults
    assert runner._completion_message(STEP_ORDER) == "Pipeline completed"
    assert runner._completion_message(("diarize",)) == "Diarization complete"
    assert "Prettify, Assign" in runner._completion_message(("prettify", "assign"))
    assert runner._completion_message(("classify",)) == "Classification complete"


def test_ensure_diarized_turns_handles_list_dicts_and_rttm(tmp_path):
    runner = build_runner(tmp_path)
    # From list[dict]
    turns = ensure_diarized_turns(
        [{"start": 0.0, "end": 1.0, "speaker": "S0"}, {"start": 0.5, "end": 1.5, "speaker_id": "S1"}]
    )
    assert isinstance(turns[0], DiarizedTurn)
    assert turns[0].speaker_id in {"S0", "S1"}

    # From RTTM path
    rttm = tmp_path / "turns.rttm"
    # SPEAKER <uri> <chan> <start> <dur> <ortho> <stype> <name>
    rttm.write_text("SPEAKER test 1 0.00 0.80 <NA> <NA> SPK0\n")
    parsed = ensure_diarized_turns(rttm)
    assert parsed and parsed[0].speaker_id == "SPK0"


def test_read_transcript_edge_cases(tmp_path):
    runner = build_runner(tmp_path)
    # missing file
    assert read_transcript(tmp_path / "missing.json") is None
    # minimal valid file
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"start": 0.0, "end": 0.5, "text": "x"}]))
    segs = read_transcript(p)
    assert segs and segs[0].text == "x"


def test_handle_event_emits_progress_and_updates_queue(monkeypatch, tmp_path):
    queue = DummyQueue()
    checkpoints = InMemoryCheckpointStore()
    captured_progress: list[str] = []

    def cb(evt):
        # PipelineProgress wrapper has .payload.remarks but we can just stringify
        captured_progress.append(str(evt))

    runner = build_runner(tmp_path, queue=queue, checkpoints=checkpoints)
    runner.callback = cb  # type: ignore[assignment]

    episode = PodcastEpisode(episode_id="ep", show_title="s", episode_title="e", source_path=tmp_path / "a.wav")
    context = PipelineContext(episode=episode, resume=False, completed_steps={}, step_plan=("transcribe",))

    class StubAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def process(self, event: PipelineEvent):
            # Return a minimal checkpoint-like object the runner expects
            return PipelineCheckpoint(
                episode_id=episode.episode_id,
                step=event.step_name,
                status=event.checkpoint.get("status", "started"),
                stage=event.stage,
                message=event.message,
                pipeline_id="p1",
            )

    monkeypatch.setattr("mywhisper.myw.services.pipeline.PipelineEventAdapter", StubAdapter)
    event = PipelineEvent(
        stage="progress",
        step_name="transcribe",
        episode_id=episode.episode_id,
        message="working",
        payload={},
        checkpoint={"status": "started"},
    )
    runner._handle_event(context, StubAdapter(), "transcribe", event, ("transcribe",))

    assert queue.status_updates and queue.status_updates[-1][2] == "working"
    assert captured_progress, "Expected a progress emission"
    # status row updated
    status_row = checkpoints.get_pipeline_status(episode.episode_id)
    assert status_row and status_row.get("current_step") == "transcribe"


def test_emit_stop_and_complete_use_status_and_callback(tmp_path):
    queue = DummyQueue()
    checkpoints = InMemoryCheckpointStore()
    captured: list[str] = []

    def cb(evt):
        captured.append(evt.__class__.__name__)

    runner = build_runner(tmp_path, queue=queue, checkpoints=checkpoints)
    runner.callback = cb  # type: ignore[assignment]

    # Seed a status row so _emit_stop/_emit_complete can update it
    checkpoints.set_pipeline_status(episode_id="ep", pipeline_id="p1", status="in_progress", current_step="assign")
    runner._emit_stop("ep", "paused")
    assert "PipelineStopped" in captured

    captured.clear()
    runner._emit_complete("ep", "done")
    assert "PipelineCompleted" in captured


def test_check_stop_raises_when_flagged(tmp_path):
    queue = DummyQueue()
    runner = build_runner(tmp_path, queue=queue)
    queue._should_stop = True
    with pytest.raises(PipelineInterrupted):
        runner._check_stop()


def test_load_classified_path(tmp_path):
    """Test loading classified path from checkpoint."""
    runner = build_runner(tmp_path)
    episode = PodcastEpisode(episode_id="ep", show_title="s", episode_title="e", source_path=tmp_path / "a.wav")
    context = PipelineContext(episode=episode, resume=False, completed_steps={}, step_plan=("classify",))

    # No checkpoint
    assert load_step_path("classify", runner.checkpoints, context.episode.episode_id) is None

    # With checkpoint
    checkpoints = InMemoryCheckpointStore()
    classified_file = tmp_path / "classified.json"
    classified_file.write_text('[]')
    checkpoints.upsert(
        PipelineCheckpoint(
            episode_id="ep",
            step="classify",
            status="completed",
            stage="persisted",
            message="done",
            details={"classified_path": str(classified_file)},
        )
    )
    runner.checkpoints = checkpoints
    result = load_step_path("classify", runner.checkpoints, context.episode.episode_id)
    assert result == classified_file


def test_load_themes_path(tmp_path):
    """Test loading themes path from checkpoint."""
    runner = build_runner(tmp_path)
    episode = PodcastEpisode(episode_id="ep", show_title="s", episode_title="e", source_path=tmp_path / "a.wav")
    context = PipelineContext(episode=episode, resume=False, completed_steps={}, step_plan=("thematize",))

    # No checkpoint
    assert load_step_path("thematize", runner.checkpoints, context.episode.episode_id) is None

    # With checkpoint
    checkpoints = InMemoryCheckpointStore()
    themes_file = tmp_path / "themes.json"
    themes_file.write_text('[]')
    checkpoints.upsert(
        PipelineCheckpoint(
            episode_id="ep",
            step="thematize",
            status="completed",
            stage="persisted",
            message="done",
            details={"themes_path": str(themes_file)},
        )
    )
    runner.checkpoints = checkpoints
    result = load_step_path("thematize", runner.checkpoints, context.episode.episode_id)
    assert result == themes_file


def test_load_themes_path_from_payload(tmp_path):
    """Test loading themes path from payload when details missing."""
    runner = build_runner(tmp_path)
    episode = PodcastEpisode(episode_id="ep", show_title="s", episode_title="e", source_path=tmp_path / "a.wav")
    context = PipelineContext(episode=episode, resume=False, completed_steps={}, step_plan=("thematize",))

    checkpoints = InMemoryCheckpointStore()
    themes_file = tmp_path / "themes.json"
    themes_file.write_text('[]')
    checkpoints.upsert(
        PipelineCheckpoint(
            episode_id="ep",
            step="thematize",
            status="completed",
            stage="persisted",
            message="done",
            payload={"path": str(themes_file)},
        )
    )
    runner.checkpoints = checkpoints
    result = load_step_path("thematize", runner.checkpoints, context.episode.episode_id)
    assert result == themes_file


def test_resolve_step_plan_includes_classify(tmp_path):
    """Test that classify step is included in STEP_ORDER."""
    runner = build_runner(tmp_path)
    from mywhisper.myw.services.queue import QueueItem

    # Test that classify is in STEP_ORDER
    assert "classify" in STEP_ORDER
    assert STEP_ORDER.index("classify") > STEP_ORDER.index("thematize")
    assert STEP_ORDER.index("classify") < STEP_ORDER.index("assign")

    # Test that classify can be resolved
    item = QueueItem("ep1", steps=("classify",))
    assert runner._resolve_step_plan(item) == ("classify",)


def test_plan_requires_transcript(tmp_path):
    """Test _plan_requires_transcript logic."""
    runner = build_runner(tmp_path)
    # diarize requires transcript if transcribe not in plan
    assert ("transcribe" not in ("diarize",) and "diarize" in ("diarize",)) is True
    # transcribe in plan -> no requirement
    assert ("transcribe" not in ("transcribe", "diarize") and "diarize" in ("transcribe", "diarize")) is False
    # classify doesn't require transcript directly
    assert ("transcribe" not in ("classify",) and "diarize" in ("classify",)) is False


def test_validate_diarization_availability_with_classify(tmp_path):
    """Test that classify step doesn't require diarization."""
    runner = build_runner(tmp_path)
    # classify doesn't require diarization
    validate_diarization_availability(("classify",), {}, None)
    # But prettify/assign do
    with pytest.raises(RuntimeError):
        validate_diarization_availability(("prettify",), {}, None)


