from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ...assign import AssignmentConfig, TranscriptAssigner
from ...checkpoints import PipelineEventAdapter, CheckpointStore
from ...diarize import DiarizationConfig, DiarizationPipeline
from ...models import DiarizedTurn, PipelineEvent, PodcastEpisode, TranscriptSegment
from ...podcasts import PodcastCatalog
from ...transcribe import PodcastTranscriber, TranscriptionConfig
from ..config import MywConfig
from ..messages import PipelineCompleted, PipelineEventPayload, PipelineProgress, PipelineStopped
from ..models import PipelineStatus
from .queue import QueueController, QueueItem

LOGGER = logging.getLogger("mywhisper.myw.pipeline")


def _stringify_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_stringify_data(val) for val in value)
    if isinstance(value, set):
        return sorted(_stringify_data(val) for val in value)
    return value


def _serialize_dataclass(instance: Any) -> Dict[str, Any]:
    try:
        return {field.name: _stringify_data(getattr(instance, field.name)) for field in fields(instance)}
    except TypeError:
        return _stringify_data(instance)

ProgressCallback = Callable[[PipelineProgress | PipelineStopped | PipelineCompleted], None]

STEP_ORDER = ("transcribe", "diarize", "assign")


class PipelineInterrupted(Exception):
    """Raised when a stop is requested."""


@dataclass(slots=True)
class PipelineContext:
    episode: PodcastEpisode
    resume: bool
    completed_steps: Dict[str, str]
    step_plan: tuple[str, ...]


class PipelineRunner:
    """
    Drive mywhisper pipelines sequentially on a background thread.
    """

    def __init__(
        self,
        config: MywConfig,
        catalog: PodcastCatalog,
        queue: QueueController,
        checkpoints: CheckpointStore,
        callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.queue = queue
        self.checkpoints = checkpoints
        self.callback = callback
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = threading.Event()
        LOGGER.info("PipelineRunner configured with: %s", _serialize_dataclass(config))

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread.start()

    def shutdown(self) -> None:
        self._running.clear()
        self.queue.request_shutdown()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    # Internal execution helpers
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        while self._running.is_set():
            item = self.queue.next_item()
            if item is None:
                break

            episode = self.catalog.get_episode(item.episode_id)
            if not episode:
                LOGGER.warning("Episode %s not found in catalog", item.episode_id)
                self.queue.release_current()
                continue

            context = PipelineContext(
                episode=episode,
                resume=item.resume,
                completed_steps=self._completed_steps(episode.episode_id),
                step_plan=self._resolve_step_plan(item),
            )
            LOGGER.info(
                "Executing plan %s for episode %s (resume=%s)",
                context.step_plan,
                episode.episode_id,
                context.resume,
            )

            try:
                self._process_episode(context)
            except PipelineInterrupted:
                LOGGER.info("Pipeline interrupted for episode %s", episode.episode_id)
                self.queue.set_status(episode.episode_id, "Stopped", "Paused by user")
                self._emit_stop(episode.episode_id, "Pipeline paused")
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Pipeline failed for %s: %s", episode.episode_id, exc)
                self.queue.set_status(episode.episode_id, "Stopped", f"Error: {exc}")
                self._emit_stop(episode.episode_id, f"Error: {exc}")
            else:
                remarks = self._completion_message(context.step_plan)
                LOGGER.info(
                    "Completed plan %s for episode %s with remarks: %s",
                    context.step_plan,
                    context.episode.episode_id,
                    remarks,
                )
                self.queue.set_status(context.episode.episode_id, "Completed", remarks)
                self._emit_complete(context.episode.episode_id, remarks)
            finally:
                self.queue.release_current()

    def _process_episode(self, context: PipelineContext) -> None:
        adapter = PipelineEventAdapter(self.checkpoints, episode_id=context.episode.episode_id)

        plan = context.step_plan
        transcript_segments: Optional[Sequence[TranscriptSegment]] = None
        if "transcribe" in plan:
            if "transcribe" not in context.completed_steps:
                transcript_segments = self._run_transcription(context, adapter)
            else:
                self._log_step_start(
                    context,
                    "transcribe",
                    {
                        "mode": "load_checkpoint",
                        "checkpoint_status": context.completed_steps["transcribe"],
                    },
                )
                load_start = perf_counter()
                transcript_segments = self._load_transcript_segments(context)
                load_elapsed = perf_counter() - load_start
                self._log_step_end(
                    context,
                    "transcribe",
                    {
                        **self._transcript_summary(transcript_segments),
                        "source": "checkpoint",
                        "elapsed": round(load_elapsed, 2),
                    },
                )
                if transcript_segments is None:
                    LOGGER.warning(
                        "Transcript checkpoint missing for %s; rerunning transcription",
                        context.episode.episode_id,
                    )
                    transcript_segments = self._run_transcription(context, adapter)
        elif self._plan_requires_transcript(plan):
            self._log_step_start(
                context,
                "transcribe",
                {"mode": "load_checkpoint", "reason": "downstream_step_requires_transcript"},
            )
            load_start = perf_counter()
            transcript_segments = self._load_transcript_segments(context)
            load_elapsed = perf_counter() - load_start
            self._log_step_end(
                context,
                "transcribe",
                {
                    **self._transcript_summary(transcript_segments),
                    "source": "checkpoint",
                    "elapsed": round(load_elapsed, 2),
                },
            )
        self._validate_transcript_availability(plan, transcript_segments)
        self._check_stop()

        diarization_results = None
        if "diarize" in plan:
            if "diarize" not in context.completed_steps:
                diarization_results = self._run_diarization(context, transcript_segments, adapter)
            else:
                self._log_step_start(
                    context,
                    "diarize",
                    {
                        "mode": "load_checkpoint",
                        "checkpoint_status": context.completed_steps["diarize"],
                    },
                )
                load_start = perf_counter()
                diarization_results = self._load_diarization_results(context)
                load_elapsed = perf_counter() - load_start
                self._log_step_end(
                    context,
                    "diarize",
                    {
                        **self._diarization_summary(diarization_results),
                        "source": "checkpoint",
                        "elapsed": round(load_elapsed, 2),
                    },
                )
                if diarization_results is None:
                    LOGGER.warning(
                        "Diarization checkpoint missing for %s; rerunning diarization",
                        context.episode.episode_id,
                    )
                    diarization_results = self._run_diarization(context, transcript_segments, adapter)
        elif "assign" in plan:
            self._log_step_start(
                context,
                "diarize",
                {"mode": "load_checkpoint", "reason": "assignment_requires_diarization"},
            )
            load_start = perf_counter()
            diarization_results = self._load_diarization_results(context)
            load_elapsed = perf_counter() - load_start
            self._log_step_end(
                context,
                "diarize",
                {
                    **self._diarization_summary(diarization_results),
                    "source": "checkpoint",
                    "elapsed": round(load_elapsed, 2),
                },
            )
        self._validate_diarization_availability(plan, context.completed_steps, diarization_results)
        self._check_stop()

        if (
            "assign" in plan
            and "assign" not in context.completed_steps
            and transcript_segments is not None
        ):
            diarized_turns = self._ensure_diarized_turns(diarization_results)
            if diarized_turns:
                transcript_segments = self._apply_diarization_labels(transcript_segments, diarized_turns)
            else:
                LOGGER.warning(
                    "No diarization turns available for %s; speaker IDs will remain unset.",
                    context.episode.episode_id,
                )
            self._run_assignment(context, transcript_segments, adapter)
        self._check_stop()

    def _run_transcription(
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
    ) -> Sequence[TranscriptSegment]:
        if not self.config.whisper_model:
            raise RuntimeError("MYW_WHISPER_MODEL must be configured for transcription.")

        config = TranscriptionConfig(
            model_path=Path(self.config.whisper_model),
            data_root=self.config.data_dir,
            device=self.config.device,
        )
        self._log_step_start(
            context,
            "transcribe",
            {
                "mode": "execute",
                "config": _serialize_dataclass(config),
                "resume": context.resume,
            },
        )
        start_time = perf_counter()
        transcriber = PodcastTranscriber.from_config(context.episode, config)
        events = transcriber.transcribe(yield_progress=True)
        segments = self._consume_events(context, adapter, "transcribe", events, context.step_plan)
        elapsed = perf_counter() - start_time
        self._log_step_end(
            context,
            "transcribe",
            {
                **self._transcript_summary(segments),
                "source": "fresh",
                "elapsed": round(elapsed, 2),
            },
        )
        return segments

    def _load_transcript_segments(
        self,
        context: PipelineContext,
    ) -> Optional[Sequence[TranscriptSegment]]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "transcribe")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("transcript_path") or checkpoint.payload.get("path")
            LOGGER.info(f"Loading cached transcript from {path}")
            if path:
                transcript_path = Path(path)
                try:
                    return self._read_transcript(transcript_path)
                except Exception:
                    LOGGER.warning("Failed to load cached transcript at %s", transcript_path)
        return None

    def _run_diarization(
        self,
        context: PipelineContext,
        transcript_segments: Optional[Sequence[TranscriptSegment]],
        adapter: PipelineEventAdapter,
    ):
        config = DiarizationConfig(
            data_root=self.config.data_dir,
            hf_token=self.config.hf_token,
            device=self.config.device,
        )
        self._log_step_start(
            context,
            "diarize",
            {
                "mode": "execute",
                "config": _serialize_dataclass(config),
                "transcript_available": bool(transcript_segments),
            },
        )
        start_time = perf_counter()
        pipeline = DiarizationPipeline.from_config(context.episode, config)
        paths = config.artefact_paths(context.episode, context.episode.episode_key)

        start_event = PipelineEvent(
            stage="start",
            step_name="diarize",
            episode_id=context.episode.episode_id,
            message=f"Starting diarization for {context.episode.episode_title}",
            payload={
                "episode_key": context.episode.episode_key,
                "step": "started",
            },
            artefact_paths={"rttm": paths["rttm_path"]},
            checkpoint={
                "status": "started",
                "step": "diarize",
                "episode_key": context.episode.episode_key,
            },
        )
        self._handle_event(context, adapter, "diarize", start_event, context.step_plan)

        turns = pipeline.run()
        elapsed = perf_counter() - start_time

        completed_event = PipelineEvent(
            stage="persisted",
            step_name="diarize",
            episode_id=context.episode.episode_id,
            message="Persisted diarization RTTM",
            payload={
                "path": str(paths["rttm_path"]),
                "turns": len(turns),
                "step": "completed",
            },
            artefact_paths={"rttm": paths["rttm_path"]},
            checkpoint={
                "status": "completed",
                "step": "diarize",
                "rttm_path": str(paths["rttm_path"]),
                "turns": len(turns),
            },
        )
        self._handle_event(context, adapter, "diarize", completed_event, context.step_plan)
        self._log_step_end(
            context,
            "diarize",
            {
                **self._diarization_summary(turns, artefact_path=str(paths["rttm_path"])),
                "source": "fresh",
                "elapsed": round(elapsed, 2),
            },
        )
        return turns

    def _load_diarization_results(self, context: PipelineContext):
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "diarize")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("rttm_path")
            if path:
                return Path(path)
        return None

    def _run_assignment(
        self,
        context: PipelineContext,
        segments: Sequence[TranscriptSegment],
        adapter: PipelineEventAdapter,
    ) -> Sequence[TranscriptSegment]:
        config = AssignmentConfig(
            data_root=self.config.data_dir,
            ollama_model=self.config.ollama_model,
            spacy_model=self.config.spacy_model,
        )
        self._log_step_start(
            context,
            "assign",
            {
                "mode": "execute",
                "config": _serialize_dataclass(config),
                "segment_count": len(segments),
            },
        )
        start_time = perf_counter()
        assigner = TranscriptAssigner.from_config(context.episode, config)
        events = assigner.assign_names(segments, metadata=context.episode.metadata, yield_progress=True)
        assignment = self._consume_events(context, adapter, "assign", events, context.step_plan)
        elapsed = perf_counter() - start_time
        self._log_step_end(
            context,
            "assign",
            {
                **self._assignment_summary(assignment),
                "elapsed": round(elapsed, 2),
                "source": "fresh",
            },
        )
        return assignment

    def _consume_events(
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        step: str,
        events: Iterable[PipelineEvent],
        step_plan: Sequence[str],
    ):
        iterator = iter(events)
        result = None
        while True:
            try:
                event = next(iterator)
            except StopIteration as stop:
                result = stop.value
                break
            self._handle_event(context, adapter, step, event, step_plan)
        return result

    def _handle_event(
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        step: str,
        event: PipelineEvent,
        step_plan: Sequence[str],
    ) -> None:
        if event.transient:
            self._check_stop()
            return

        adapter.process(event)
        self.queue.set_status(
            context.episode.episode_id,
            "In progress",
            event.message,
        )
        status = PipelineStatus(
            active=True,
            episode_id=context.episode.episode_id,
            step=step,
            progress=self._progress_for(step, event, step_plan),
            message=event.message,
        )
        self._emit_progress(context.episode.episode_id, status, event.message)
        self._check_stop()

    def _log_step_start(self, context: PipelineContext, step: str, inputs: Dict[str, Any]) -> None:
        LOGGER.info(
            "Episode %s | step=%s | start | inputs=%s",
            context.episode.episode_id,
            step,
            _stringify_data(inputs),
        )

    def _log_step_end(self, context: PipelineContext, step: str, outputs: Dict[str, Any]) -> None:
        LOGGER.info(
            "Episode %s | step=%s | complete | outputs=%s",
            context.episode.episode_id,
            step,
            _stringify_data(outputs),
        )

    def _transcript_summary(
        self,
        segments: Optional[Sequence[TranscriptSegment]],
    ) -> Dict[str, Any]:
        if not segments:
            return {"segments": 0, "duration_sec": 0.0, "speaker_ids": 0}
        starts = [seg.start for seg in segments]
        ends = [seg.end for seg in segments]
        duration = max(ends, default=0.0) - min(starts, default=0.0)
        unique_speakers = len({seg.speaker_id for seg in segments if seg.speaker_id})
        return {
            "segments": len(segments),
            "duration_sec": round(duration, 2),
            "speaker_ids": unique_speakers,
        }

    def _diarization_summary(
        self,
        diarization_results,
        artefact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if diarization_results is None:
            return {"turns": 0, "artefact_path": artefact_path}
        if isinstance(diarization_results, (str, Path)):
            return {"turns": None, "artefact_path": str(diarization_results)}
        turns_count = len(diarization_results) if hasattr(diarization_results, "__len__") else None
        return {
            "turns": turns_count,
            "artefact_path": artefact_path,
        }

    def _assignment_summary(
        self,
        segments: Optional[Sequence[TranscriptSegment]],
    ) -> Dict[str, Any]:
        if not segments:
            return {"segments": 0, "named_segments": 0, "unknown_segments": 0}
        named_segments = sum(
            1 for seg in segments if seg.speaker_name and seg.speaker_name.strip().upper() != "UNKNOWN"
        )
        unknown_segments = len(segments) - named_segments
        return {
            "segments": len(segments),
            "named_segments": named_segments,
            "unknown_segments": unknown_segments,
        }

    def _completed_steps(self, episode_id: str) -> Dict[str, str]:
        checkpoints = self.checkpoints.get_episode(episode_id)
        return {cp.step: cp.status for cp in checkpoints if cp.status == "completed"}

    def _resolve_step_plan(self, item: QueueItem) -> tuple[str, ...]:
        raw_plan = item.steps or STEP_ORDER
        normalized: list[str] = []
        seen: set[str] = set()
        for step in raw_plan:
            if step not in STEP_ORDER or step in seen:
                continue
            normalized.append(step)
            seen.add(step)
        if not normalized:
            LOGGER.debug("Queue item %s provided empty step plan; defaulting to full pipeline", item.episode_id)
            return STEP_ORDER
        return tuple(normalized)

    def _plan_requires_transcript(self, plan: Sequence[str]) -> bool:
        return "transcribe" not in plan and any(step in ("diarize", "assign") for step in plan)

    def _validate_transcript_availability(
        self,
        plan: Sequence[str],
        segments: Optional[Sequence[TranscriptSegment]],
    ) -> None:
        if self._plan_requires_transcript(plan) and segments is None:
            raise RuntimeError(
                "Transcript segments are required for the selected steps. Run transcription first or include it in the plan."
            )

    def _validate_diarization_availability(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        diarization_results,
    ) -> None:
        if "assign" not in plan:
            return
        if "diarize" in plan or "diarize" in completed_steps:
            return
        if diarization_results is None:
            raise RuntimeError(
                "Diarization results are required for assignment. Run diarization first or include it in the plan."
            )

    def _completion_message(self, plan: Sequence[str]) -> str:
        normalized = list(plan) or list(STEP_ORDER)
        if tuple(normalized) == STEP_ORDER:
            return "Pipeline completed"
        if len(normalized) == 1:
            step = normalized[0]
            mapping = {
                "transcribe": "Transcription complete",
                "diarize": "Diarization complete",
                "assign": "Assignment complete",
            }
            return mapping.get(step, f"{step.title()} complete")
        pretty = ", ".join(step.title() for step in normalized)
        return f"{pretty} complete"

    def _check_stop(self) -> None:
        if self.queue.should_stop():
            raise PipelineInterrupted()

    def _progress_for(self, step: str, event: PipelineEvent, step_plan: Sequence[str]) -> float:
        if not step_plan:
            return 0.0
        try:
            index = step_plan.index(step)
        except ValueError:
            return 0.0
        total = len(step_plan)
        completed = index / total
        if event.checkpoint.get("status") == "completed":
            completed = (index + 1) / total
        return min(max(completed, 0.0), 1.0)

    def _emit_progress(self, episode_id: str, status: PipelineStatus, remarks: str) -> None:
        if not self.callback:
            return
        payload = PipelineEventPayload(
            episode_id=episode_id,
            status=status,
            remarks=remarks,
        )
        self.callback(PipelineProgress(payload))

    def _emit_stop(self, episode_id: str, remarks: str) -> None:
        if not self.callback:
            return
        status = PipelineStatus(active=False, episode_id=episode_id, step="Stopped", message=remarks)
        payload = PipelineEventPayload(episode_id=episode_id, status=status, remarks=remarks)
        self.callback(PipelineStopped(payload))

    def _emit_complete(self, episode_id: str, remarks: str) -> None:
        if not self.callback:
            return
        status = PipelineStatus(active=False, episode_id=episode_id, step="Completed", message=remarks, progress=1.0)
        payload = PipelineEventPayload(episode_id=episode_id, status=status, remarks=remarks)
        self.callback(PipelineCompleted(payload))

    def _read_transcript(self, path: Path) -> Optional[Sequence[TranscriptSegment]]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        segments: list[TranscriptSegment] = []
        for item in data:
            segments.append(
                TranscriptSegment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item.get("text", "")),
                    speaker_id=item.get("speaker_id") or item.get("speaker"),
                    speaker_name=item.get("speaker_name"),
                    confidence=item.get("confidence"),
                    justification=item.get("justification"),
                    metadata=item.get("metadata", {}),
                )
            )
        return segments

    def _ensure_diarized_turns(self, diarization_results) -> List[DiarizedTurn]:
        if diarization_results is None:
            return []

        if isinstance(diarization_results, list):
            turns: List[DiarizedTurn] = []
            for item in diarization_results:
                if isinstance(item, DiarizedTurn):
                    turns.append(item)
                elif isinstance(item, dict):
                    try:
                        start = float(item["start"])
                        end = float(item["end"])
                        speaker = str(item.get("speaker") or item.get("speaker_id") or "")
                    except (KeyError, TypeError, ValueError):
                        continue
                    turns.append(DiarizedTurn(start=start, end=end, speaker_id=speaker or "UNKNOWN"))
            turns.sort(key=lambda turn: (turn.start, turn.end))
            return turns

        if isinstance(diarization_results, (str, Path)):
            return self._read_rttm_turns(Path(diarization_results))

        return []

    def _read_rttm_turns(self, path: Path) -> List[DiarizedTurn]:
        if not path.exists():
            LOGGER.warning("RTTM file %s not found; cannot load diarization turns.", path)
            return []

        turns: List[DiarizedTurn] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                    continue
                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                except ValueError:
                    continue
                speaker = parts[7] if len(parts) > 7 else ""
                turns.append(
                    DiarizedTurn(
                        start=start,
                        end=start + duration,
                        speaker_id=str(speaker or f"speaker_{len(turns)}"),
                    )
                )
        turns.sort(key=lambda turn: (turn.start, turn.end))
        return turns

    def _apply_diarization_labels(
        self,
        segments: Sequence[TranscriptSegment],
        turns: Sequence[DiarizedTurn],
    ) -> List[TranscriptSegment]:
        if not segments:
            return []
        if not turns:
            return list(segments)

        sorted_turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
        updated_segments: List[TranscriptSegment] = []
        leading_index = 0
        total_turns = len(sorted_turns)

        for seg in segments:
            if seg.speaker_id:
                updated_segments.append(seg)
                continue

            start = seg.start
            end = seg.end
            best_id: Optional[str] = None
            best_overlap = 0.0

            idx = leading_index
            while idx < total_turns and sorted_turns[idx].end <= start:
                idx += 1
            leading_index = idx

            scan = idx
            while scan < total_turns:
                turn = sorted_turns[scan]
                if turn.start >= end:
                    break
                overlap_start = max(start, turn.start)
                overlap_end = min(end, turn.end)
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_id = turn.speaker_id
                scan += 1

            if best_id:
                updated_segments.append(
                    TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker_id=best_id,
                        speaker_name=seg.speaker_name,
                        confidence=seg.confidence,
                        justification=seg.justification,
                        metadata=dict(seg.metadata),
                    )
                )
            else:
                updated_segments.append(seg)

        return updated_segments

