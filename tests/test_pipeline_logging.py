from __future__ import annotations

from pathlib import Path
import json
import logging

from mywhisper.checkpoints.models import PipelineCheckpoint
from mywhisper.models import DiarizedTurn, PipelineEvent, PodcastEpisode, TranscriptSegment
from mywhisper.myw.config import MywConfig
from mywhisper.myw.services.pipeline import (
    PipelineContext,
    PipelineRunner,
    QueueItem,
    _serialize_dataclass,
    _stringify_data,
)


class DummyQueue:
    def request_shutdown(self) -> None:
        pass

    def next_item(self):
        return None

    def release_current(self) -> None:
        pass

    def set_status(self, *_args, **_kwargs) -> None:
        pass

    def should_stop(self) -> bool:
        return False


class DummyCheckpoints:
    def get_step(self, *_args, **_kwargs):
        return None

    def get_episode(self, *_args, **_kwargs):
        return []

    def delete_episode(self, *_args, **_kwargs):
        pass


class InMemoryCheckpointStore(DummyCheckpoints):
    def __init__(self, rows: dict[tuple[str, str], PipelineCheckpoint] | None = None) -> None:
        self._rows = rows or {}

    def get_step(self, episode_id: str, step: str):
        return self._rows.get((episode_id, step))

    def get_episode(self, episode_id: str):
        return [checkpoint for (eid, _), checkpoint in self._rows.items() if eid == episode_id]

    def upsert(self, checkpoint: PipelineCheckpoint) -> None:
        self._rows[(checkpoint.episode_id, checkpoint.step)] = checkpoint

    def delete_episode(self, episode_id: str) -> None:
        for key in [key for key in self._rows if key[0] == episode_id]:
            del self._rows[key]


def build_runner(tmp_path: Path, *, queue: DummyQueue | None = None, checkpoints: DummyCheckpoints | None = None) -> PipelineRunner:
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
    return PipelineRunner(
        config=config,
        catalog=object(),
        queue=queue or DummyQueue(),
        checkpoints=checkpoints or DummyCheckpoints(),
        callback=None,
    )


def test_stringify_and_serialize_handles_collections(tmp_path):
    path_value = tmp_path / "file.txt"
    data = {
        "path": path_value,
        "list": [path_value],
        "set": {path_value},
    }
    result = _stringify_data(data)
    assert result["path"] == str(path_value)
    assert result["list"][0] == str(path_value)
    assert result["set"][0] == str(path_value)

    runner = build_runner(tmp_path)
    serialized = _serialize_dataclass(runner.config)
    assert serialized["data_dir"] == str(tmp_path)
    assert serialized["ollama_model"] == "llama3"


def test_summary_helpers_and_logging(tmp_path, caplog):
    runner = build_runner(tmp_path)
    episode = PodcastEpisode(
        episode_id="ep-log",
        show_title="Show",
        episode_title="Episode",
        source_path=tmp_path / "audio.wav",
    )
    context = PipelineContext(
        episode=episode,
        resume=False,
        completed_steps={},
        step_plan=("transcribe",),
    )
    segments = [
        TranscriptSegment(
            start=0.0,
            end=1.0,
            text="Hi",
            speaker_id="S0",
            speaker_name="Host",
        ),
        TranscriptSegment(
            start=2.0,
            end=3.5,
            text="Hello",
            speaker_id="S1",
            speaker_name="UNKNOWN",
        ),
    ]
    transcript_summary = runner._transcript_summary(segments)
    assert transcript_summary["segments"] == 2
    assert transcript_summary["speaker_ids"] == 2
    assert transcript_summary["duration_sec"] == 3.5

    diarization_summary = runner._diarization_summary(
        [{"speaker": "S0"}],
        artefact_path="/tmp/turns.rttm",
    )
    assert diarization_summary["turns"] == 1
    assert diarization_summary["artefact_path"] == "/tmp/turns.rttm"

    file_summary = runner._diarization_summary("/tmp/turns.rttm")
    assert file_summary["turns"] is None
    assert file_summary["artefact_path"] == "/tmp/turns.rttm"

    assignment_summary = runner._assignment_summary(segments)
    assert assignment_summary["segments"] == 2
    assert assignment_summary["named_segments"] == 1
    assert assignment_summary["unknown_segments"] == 1

    caplog.set_level(logging.INFO)
    runner._log_step_start(context, "transcribe", {"mode": "execute"})
    runner._log_step_end(context, "transcribe", {"segments": 2})
    start_logs = [record.message for record in caplog.records if "start" in record.message]
    assert any("step=transcribe" in message and "start" in message for message in start_logs)


def test_pipeline_process_episode_runs_full_plan(monkeypatch, tmp_path):
    queue = DummyQueue()
    checkpoints = InMemoryCheckpointStore()
    runner = build_runner(tmp_path, queue=queue, checkpoints=checkpoints)
    episode = PodcastEpisode(
        episode_id="ep-full",
        show_title="Full",
        episode_title="Plan",
        source_path=tmp_path / "audio.wav",
    )
    context = PipelineContext(
        episode=episode,
        resume=False,
        completed_steps={},
        step_plan=("transcribe", "diarize", "prettify", "thematize", "assign"),
    )

    transcript_segments = [
        TranscriptSegment(0.0, 1.0, "Hello", speaker_id="S0"),
        TranscriptSegment(1.0, 2.0, "Hi", speaker_id="S1"),
    ]

    assigned_segments = [
        TranscriptSegment(0.0, 1.0, "Hello", speaker_id="S0", speaker_name="Host"),
        TranscriptSegment(1.0, 2.0, "Hi", speaker_id="S1", speaker_name="Guest"),
    ]

    def make_event(step: str, message: str, status: str) -> PipelineEvent:
        return PipelineEvent(
            stage="progress",
            step_name=step,
            episode_id=episode.episode_id,
            message=message,
            payload={"step": message},
            checkpoint={"status": status},
        )

    class StubTranscriber:
        def transcribe(self, yield_progress=True):
            def generator():
                yield make_event("transcribe", "started", "started")
                return transcript_segments

            return generator()

    class StubDiarizationPipeline:
        def run(self):
            return [DiarizedTurn(start=0.0, end=1.0, speaker_id="S0")]

    assignment_path = tmp_path / "assigned.json"
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text("[]")
    readable_path = tmp_path / "readable.txt"
    readable_path.write_text("Host (S0): Hello")
    condensed_path = tmp_path / "condensed.json"
    condensed_path.write_text('[{"start":0.0,"end":1.0,"speaker_id":"S0","speaker_name":"Host","text":"Hello"}]')
    themes_path = tmp_path / "with_themes.json"
    themes_path.write_text("[]")

    class StubAssigner:
        def __init__(self):
            self._last_assignment_path = assignment_path

        def assign_names(self, segments, metadata=None, yield_progress=True):
            def generator():
                yield PipelineEvent(
                    stage="progress",
                    step_name="assign",
                    episode_id=episode.episode_id,
                    message="started",
                    payload={"step": "completed"},
                    checkpoint={
                        "status": "completed",
                        "assignment_path": str(assignment_path),
                    },
                )
                return assigned_segments

            return generator()
        def assign_from_readable(self, readable_path: Path, metadata=None, yield_progress: bool = True):
            # Mirror assign_names behavior for this stub, ignoring readable input
            return self.assign_names([], metadata=metadata, yield_progress=yield_progress)

    class StubPrettifier:
        def __init__(self, *args, **kwargs):
            pass

        def prettify(self, assignment_path: Path | None = None, yield_progress: bool = True):
            def generator():
                yield PipelineEvent(
                    stage="prettify",
                    step_name="prettify",
                    episode_id=episode.episode_id,
                    message="prettifying",
                    payload={"step": "completed"},
                    checkpoint={
                        "status": "completed",
                        "readable_path": str(readable_path),
                        "condensed_path": str(condensed_path),
                    },
                    artefact_paths={"readable": readable_path, "condensed": condensed_path},
                )
                return readable_path

            return generator()

    class StubThematizer:
        def __init__(self, *args, **kwargs):
            pass

        def thematize(self, condensed_path: Path | None = None, yield_progress: bool = True):
            def generator():
                yield PipelineEvent(
                    stage="thematize",
                    step_name="thematize",
                    episode_id=episode.episode_id,
                    message="thematizing",
                    payload={"step": "completed"},
                    checkpoint={
                        "status": "completed",
                        "themes_path": str(themes_path),
                        "condensed_path": str(condensed_path),
                    },
                    artefact_paths={"with_themes": themes_path},
                )
                return themes_path

            return generator()

    captured_events: list[PipelineEvent] = []

    class StubAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def process(self, event: PipelineEvent):
            captured_events.append(event)

    monkeypatch.setattr(
        "mywhisper.myw.services.pipeline.PodcastTranscriber.from_config",
        classmethod(lambda cls, episode, config: StubTranscriber()),
    )
    monkeypatch.setattr(
        "mywhisper.myw.services.pipeline.DiarizationPipeline.from_config",
        classmethod(lambda cls, episode, config: StubDiarizationPipeline()),
    )
    monkeypatch.setattr(
        "mywhisper.myw.services.pipeline.TranscriptAssigner.from_config",
        classmethod(lambda cls, episode, config: StubAssigner()),
    )
    monkeypatch.setattr(
        "mywhisper.myw.services.pipeline.TranscriptPrettifier",
        StubPrettifier,
    )
    monkeypatch.setattr(
        "mywhisper.myw.services.pipeline.EpisodeThematizer",
        StubThematizer,
    )
    monkeypatch.setattr("mywhisper.myw.services.pipeline.PipelineEventAdapter", StubAdapter)

    runner._process_episode(context)

    assert any(event.step_name == "transcribe" for event in captured_events)
    assert any(event.step_name == "assign" for event in captured_events)
    assert any(event.step_name == "prettify" for event in captured_events)
    assert any(event.step_name == "thematize" for event in captured_events)


def test_pipeline_process_episode_uses_cached_checkpoints(monkeypatch, tmp_path):
    transcript_path = tmp_path / "cached_transcript.json"
    transcript_payload = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Cached hello",
            "speaker_id": "S0",
            "speaker_name": "UNKNOWN",
        }
    ]
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(transcript_payload))

    rttm_path = tmp_path / "cached.rttm"
    rttm_path.write_text("dummy")

    episode = PodcastEpisode(
        episode_id="ep-cache",
        show_title="Cache",
        episode_title="Warm",
        source_path=tmp_path / "audio.wav",
    )

    checkpoints = InMemoryCheckpointStore(
        {
            (episode.episode_id, "transcribe"): PipelineCheckpoint(
                episode_id=episode.episode_id,
                step="transcribe",
                status="completed",
                stage="persisted",
                message="done",
                details={"transcript_path": str(transcript_path)},
            ),
            (episode.episode_id, "diarize"): PipelineCheckpoint(
                episode_id=episode.episode_id,
                step="diarize",
                status="completed",
                stage="persisted",
                message="done",
                details={"rttm_path": str(rttm_path)},
            ),
        }
    )

    runner = build_runner(tmp_path, queue=DummyQueue(), checkpoints=checkpoints)
    context = PipelineContext(
        episode=episode,
        resume=True,
        completed_steps={"transcribe": "completed", "diarize": "completed"},
        step_plan=("transcribe", "diarize", "prettify", "assign"),
    )

    class StubAssigner:
        def assign_names(self, segments, metadata=None, yield_progress=True):
            def generator():
                yield PipelineEvent(
                    stage="start",
                    step_name="assign",
                    episode_id=episode.episode_id,
                    message="assigning",
                    checkpoint={"status": "completed"},
                )
                return segments

            return generator()
        def assign_from_readable(self, readable_path: Path, metadata=None, yield_progress: bool = True):
            return self.assign_names([], metadata=metadata, yield_progress=yield_progress)

    class StubPrettifier:
        def __init__(self, *args, **kwargs):
            pass
        def prettify(self, assignment_path: Path | None = None, yield_progress: bool = True):
            def generator():
                rp = Path(str(assignment_path).replace("_with_names.json", "_readable.txt"))
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text("Readable content", encoding="utf-8")
                yield PipelineEvent(
                    stage="prettify",
                    step_name="prettify",
                    episode_id=episode.episode_id,
                    message="prettified",
                    payload={"step": "completed"},
                    checkpoint={"status": "completed", "readable_path": str(rp)},
                    artefact_paths={"readable": rp},
                )
                return rp

            return generator()

    monkeypatch.setattr(
        "mywhisper.myw.services.pipeline.TranscriptAssigner.from_config",
        classmethod(lambda cls, episode, config: StubAssigner()),
    )
    monkeypatch.setattr(
        "mywhisper.myw.services.pipeline.TranscriptPrettifier",
        StubPrettifier,
    )

    runner._process_episode(context)


def test_pipeline_runner_run_loop(monkeypatch, tmp_path):
    processed: list[PipelineContext] = []

    def fake_process(self, context: PipelineContext):
        processed.append(context)

    monkeypatch.setattr(PipelineRunner, "_process_episode", fake_process)

    first_episode = PodcastEpisode(
        episode_id="ep-run",
        show_title="Show",
        episode_title="Episode",
        source_path=tmp_path / "audio.wav",
    )

    class StubCatalog:
        def __init__(self):
            self._episodes = {"ep-run": first_episode, "ep-dup": first_episode}

        def get_episode(self, episode_id: str):
            return self._episodes.get(episode_id)

    class StubQueue:
        def __init__(self):
            self.items = [
                QueueItem("ep-run", steps=("assign", "assign", "diarize", "invalid")),
                QueueItem("missing"),
                None,
            ]
            self.status_updates: list[tuple[str, str, str]] = []
            self.released = 0

        def next_item(self):
            return self.items.pop(0)

        def release_current(self):
            self.released += 1

        def set_status(self, episode_id, status, remarks):
            self.status_updates.append((episode_id, status, remarks))

        def should_stop(self):
            return False

        def request_shutdown(self):
            pass

    queue = StubQueue()
    runner = build_runner(tmp_path, queue=queue, checkpoints=InMemoryCheckpointStore())
    runner.catalog = StubCatalog()

    runner._running.set()
    runner._run()

    assert len(processed) == 1
    assert processed[0].step_plan == ("assign", "diarize")
    assert queue.status_updates[-1][0] == "ep-run"
    assert queue.status_updates[-1][1] == "Completed"
    assert queue.released == 2


def test_pipeline_runner_clears_checkpoints_for_fresh_rerun(monkeypatch, tmp_path):
    captured: list[PipelineContext] = []

    def fake_process(self, context: PipelineContext):
        captured.append(context)

    monkeypatch.setattr(PipelineRunner, "_process_episode", fake_process)

    episode = PodcastEpisode(
        episode_id="ep-rerun",
        show_title="Show",
        episode_title="Episode",
        source_path=tmp_path / "audio.wav",
    )

    class StubCatalog:
        def get_episode(self, episode_id: str):
            return episode if episode_id == episode.episode_id else None

    class StubQueue:
        def __init__(self):
            self.items = [QueueItem(episode.episode_id), None]

        def next_item(self):
            return self.items.pop(0)

        def release_current(self):
            pass

        def set_status(self, *_args, **_kwargs):
            pass

        def should_stop(self):
            return False

        def request_shutdown(self):
            pass

    store = InMemoryCheckpointStore(
        {
            (episode.episode_id, "transcribe"): PipelineCheckpoint(
                episode_id=episode.episode_id,
                step="transcribe",
                status="completed",
                stage="persisted",
                message="done",
            ),
            (episode.episode_id, "assign"): PipelineCheckpoint(
                episode_id=episode.episode_id,
                step="assign",
                status="started",
                stage="progress",
                message="working",
            ),
        }
    )

    runner = build_runner(tmp_path, queue=StubQueue(), checkpoints=store)
    runner.catalog = StubCatalog()

    runner._running.set()
    runner._run()
    runner._running.clear()

    assert captured, "Pipeline runner did not process the episode"
    assert captured[0].step_plan[0] == "transcribe"
    assert captured[0].completed_steps == {}
    assert store.get_episode(episode.episode_id) == []


def test_read_transcript_supports_legacy_speaker_field(tmp_path):
    runner = build_runner(tmp_path)
    transcript_path = tmp_path / "legacy.json"
    payload = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "speaker": "S0",
        }
    ]
    transcript_path.write_text(json.dumps(payload))

    segments = runner._read_transcript(transcript_path)
    assert segments
    assert segments[0].speaker_id == "S0"


def test_apply_diarization_labels_assigns_ids(tmp_path):
    runner = build_runner(tmp_path)
    segments = [
        TranscriptSegment(0.0, 1.0, "intro"),
        TranscriptSegment(1.0, 2.0, "guest"),
    ]
    turns = [
        DiarizedTurn(start=0.0, end=1.2, speaker_id="SPK0"),
        DiarizedTurn(start=1.2, end=2.5, speaker_id="SPK1"),
    ]

    labelled = runner._apply_diarization_labels(segments, turns)
    assert labelled[0].speaker_id == "SPK0"
    assert labelled[1].speaker_id == "SPK1"

