from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ...assign import AssignmentConfig, TranscriptAssigner
from ...checkpoints import PipelineEventAdapter, CheckpointStore, PipelineCheckpoint
from ...classify import ClassifyConfig, EpisodeClassifier
from ...diarize import DiarizationConfig, DiarizationPipeline
from ...models import DiarizedTurn, PipelineEvent, PodcastEpisode, TranscriptSegment
from ...podcasts import PodcastCatalog
from ...prettify import PrettifyConfig, TranscriptPrettifier
from ...thematize import EpisodeThematizer, ThematizeConfig
from ...transcribe import PodcastTranscriber, TranscriptionConfig
from ...vocative import EpisodeVocativeDetector, VocativeConfig
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

STEP_ORDER = ("transcribe", "diarize", "prettify", "thematize", "classify", "vocative", "assign")


class PipelineInterrupted(Exception):
    """Raised when a stop is requested."""


@dataclass(slots=True)
class PipelineContext:
    episode: PodcastEpisode
    resume: bool
    completed_steps: Dict[str, str]
    step_plan: tuple[str, ...]
    pipeline_id: Optional[str] = None


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

    def start(self) -> None:  # pragma: no cover - threading integration
        if self._running.is_set():
            return
        self._running.set()
        self._thread.start()

    def shutdown(self) -> None:  # pragma: no cover - threading integration
        self._running.clear()
        self.queue.request_shutdown()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    # Internal execution helpers
    # ------------------------------------------------------------------ #

    def _run(self) -> None:  # pragma: no cover - complex integration path
        while self._running.is_set():
            item = self.queue.next_item()
            if item is None:
                break

            episode = self.catalog.get_episode(item.episode_id)
            if not episode:
                LOGGER.warning("Episode %s not found in catalog", item.episode_id)
                self.queue.release_current()
                continue

            step_plan = self._resolve_step_plan(item)
            checkpoints = list(self.checkpoints.get_episode(episode.episode_id))
            if self._should_reset_checkpoints(item, step_plan, checkpoints):
                self.checkpoints.delete_episode(episode.episode_id)
                LOGGER.info(
                    "Cleared %d checkpoints for episode %s before rerun",
                    len(checkpoints),
                    episode.episode_id,
                )
                checkpoints = []

            # Mark status: in_progress
            try:
                if item.pipeline_id:
                    self.checkpoints.set_pipeline_status(
                        episode_id=episode.episode_id,
                        pipeline_id=item.pipeline_id,
                        status="in_progress",
                        current_step=None,
                        last_completed_step=None,
                        progress=0.0,
                        remarks="Starting pipeline",
                    )
            except Exception:
                LOGGER.debug("Pipeline status update failed (startup) for %s", episode.episode_id)

            context = PipelineContext(
                episode=episode,
                resume=item.resume,
                completed_steps=self._completed_steps(episode.episode_id, checkpoints),
                step_plan=step_plan,
                pipeline_id=item.pipeline_id,
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

    def _process_episode(self, context: PipelineContext) -> None:  # pragma: no cover - complex integration path
        adapter = PipelineEventAdapter(
            self.checkpoints,
            episode_id=context.episode.episode_id,
            pipeline_id=context.pipeline_id,
        )

        plan = context.step_plan
        transcript_segments: Optional[Sequence[TranscriptSegment]] = None
        if "transcribe" in plan:
            # When resume=False, re-run even if step is completed
            should_run = "transcribe" not in context.completed_steps or not context.resume
            if should_run:
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
            # When resume=False, re-run even if step is completed
            should_run = "diarize" not in context.completed_steps or not context.resume
            if should_run:
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
        elif any(step in plan for step in ("prettify", "assign")):
            self._log_step_start(
                context,
                "diarize",
                {"mode": "load_checkpoint", "reason": "downstream_step_requires_diarization"},
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

        assignment_path: Optional[Path] = None
        readable_path: Optional[Path] = None
        condensed_path: Optional[Path] = None

        # Prettify (now before assign): requires diarization; formats readable from diarized segments/placeholders
        if "prettify" in plan:
            # When resume=False, re-run even if step is completed
            should_run = "prettify" not in context.completed_steps or not context.resume
            if should_run:
                if transcript_segments is None:
                    raise RuntimeError("Transcript segments are required to run prettify.")
                diarized_turns = self._ensure_diarized_turns(diarization_results)
                if diarized_turns:
                    transcript_segments = self._apply_diarization_labels(transcript_segments, diarized_turns)
                else:
                    LOGGER.warning(
                        "No diarization turns available for %s; speaker IDs will remain unset.",
                        context.episode.episode_id,
                    )
                # Ensure a placeholder 'assignment-like' JSON exists so prettifier can operate deterministically
                placeholder_assignment = self._ensure_placeholder_assignment(context, transcript_segments or [])
                readable_path = self._run_prettify(context, adapter, placeholder_assignment)
                condensed_path = self._load_condensed_path(context)
            else:
                readable_path = self._load_readable_path(context)
                condensed_path = self._load_condensed_path(context)
                if not readable_path or not readable_path.exists():
                    LOGGER.warning(
                        "Readable transcript missing for %s; regenerating prettify output.",
                        context.episode.episode_id,
                    )
                    if transcript_segments is None:
                        raise RuntimeError("Transcript segments are required to run prettify.")
                    diarized_turns = self._ensure_diarized_turns(diarization_results)
                    if diarized_turns:
                        transcript_segments = self._apply_diarization_labels(transcript_segments, diarized_turns)
                    placeholder_assignment = self._ensure_placeholder_assignment(context, transcript_segments or [])
                    readable_path = self._run_prettify(context, adapter, placeholder_assignment)
                    condensed_path = self._load_condensed_path(context)
        elif "thematize" in plan or "assign" in plan:
            readable_path = self._load_readable_path(context)
            condensed_path = self._load_condensed_path(context)

        # Assign (now after prettify): requires a readable transcript
        if "assign" in plan:
            # When resume=False, re-run even if step is completed
            should_run = "assign" not in context.completed_steps or not context.resume
            if should_run:
                if not readable_path:
                    readable_path = self._load_readable_path(context)
                if not readable_path or not readable_path.exists():
                    raise RuntimeError("Readable transcript is required for assignment. Run prettify first.")
                # Perform name inference based on readable transcript and update artefacts
                assigned_segments, derived_assignment_path = self._run_assignment_from_readable(
                    context, adapter, readable_path
                )
                if assigned_segments is not None:
                    transcript_segments = assigned_segments
                assignment_path = derived_assignment_path or self._load_assignment_path(context)
            else:
                assignment_path = self._load_assignment_path(context)

        # Validate artefacts for thematization
        self._validate_condensed_availability(plan, condensed_path)
        self._check_stop()

        themes_path: Optional[Path] = None
        if "thematize" in plan:
            # When resume=False, re-run even if step is completed
            should_run = "thematize" not in context.completed_steps or not context.resume
            if should_run:
                if condensed_path is None:
                    raise RuntimeError("Condensed transcript required for thematization.")
                themes_path = self._run_thematize(context, adapter, condensed_path)
            else:
                themes_path = self._load_themes_path(context)
                if not themes_path or not themes_path.exists():
                    LOGGER.warning(
                        "Themes artefact missing for %s; regenerating thematization output.",
                        context.episode.episode_id,
                    )
                    if condensed_path is None:
                        raise RuntimeError("Condensed transcript required for thematization.")
                    themes_path = self._run_thematize(context, adapter, condensed_path)
        elif "classify" in plan:
            themes_path = self._load_themes_path(context)
        self._check_stop()

        # Validate themes artefact for classification
        self._validate_themes_availability(plan, themes_path)
        self._check_stop()

        if "classify" in plan:
            # When resume=False, re-run even if step is completed
            should_run = "classify" not in context.completed_steps or not context.resume
            if should_run:
                if themes_path is None:
                    raise RuntimeError("Thematized transcript required for classification.")
                self._run_classify(context, adapter, themes_path)
            else:
                classified_path = self._load_classified_path(context)
                if not classified_path or not classified_path.exists():
                    LOGGER.warning(
                        "Classified artefact missing for %s; regenerating classification output.",
                        context.episode.episode_id,
                    )
                    if themes_path is None:
                        raise RuntimeError("Thematized transcript required for classification.")
                    self._run_classify(context, adapter, themes_path)
        self._check_stop()

        # Validate classified artefact for vocative detection
        self._validate_classified_availability(plan, context.completed_steps)
        self._check_stop()

        vocative_path: Optional[Path] = None
        if "vocative" in plan:
            # When resume=False, re-run even if step is completed
            should_run = "vocative" not in context.completed_steps or not context.resume
            if should_run:
                classified_path = self._load_classified_path(context)
                if classified_path is None:
                    raise RuntimeError("Classified transcript required for vocative detection.")
                vocative_path = self._run_vocative(context, adapter, classified_path)
            else:
                vocative_path = self._load_vocative_path(context)
                if not vocative_path or not vocative_path.exists():
                    LOGGER.warning(
                        "Vocative artefact missing for %s; regenerating vocative detection output.",
                        context.episode.episode_id,
                    )
                    classified_path = self._load_classified_path(context)
                    if classified_path is None:
                        raise RuntimeError("Classified transcript required for vocative detection.")
                    vocative_path = self._run_vocative(context, adapter, classified_path)
        elif "assign" in plan:
            vocative_path = self._load_vocative_path(context)
        self._check_stop()

        # Validate vocative artefact for assignment (if needed)
        self._validate_vocative_availability(plan, vocative_path)
        self._check_stop()

    def _run_transcription(  # pragma: no cover - complex integration path
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

    def _run_diarization(  # pragma: no cover - complex integration path
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

    def _load_assignment_path(self, context: PipelineContext) -> Optional[Path]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "assign")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("assignment_path") or checkpoint.payload.get("path")
            if path:
                resolved = Path(path)
                if resolved.exists():
                    return resolved
        return None

    def _load_readable_path(self, context: PipelineContext) -> Optional[Path]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "prettify")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("readable_path") or checkpoint.payload.get("path")
            if path:
                resolved = Path(path)
                if resolved.exists():
                    return resolved
        return None

    def _load_condensed_path(self, context: PipelineContext) -> Optional[Path]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "prettify")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("condensed_path") or checkpoint.payload.get("condensed_path")
            if path:
                resolved = Path(path)
                if resolved.exists():
                    return resolved
        # Fallback to any condensed path captured from recent events
        fallback = getattr(self, "_last_condensed_path", None)
        if isinstance(fallback, Path) and fallback.exists():
            return fallback
        return None

    def _load_themes_path(self, context: PipelineContext) -> Optional[Path]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "thematize")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("themes_path") or checkpoint.payload.get("path")
            if path:
                resolved = Path(path)
                if resolved.exists():
                    return resolved
        return None

    def _run_assignment(  # pragma: no cover - complex integration path
        self,
        context: PipelineContext,
        segments: Sequence[TranscriptSegment],
        adapter: PipelineEventAdapter,
    ) -> tuple[Sequence[TranscriptSegment], Optional[Path]]:
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
        assignment_path = getattr(assigner, "_last_assignment_path", None)
        if assignment_path:
            assignment_path = Path(assignment_path)
        return assignment, assignment_path

    def _run_assignment_from_readable(  # pragma: no cover - complex integration path
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        readable_path: Path,
    ) -> tuple[Sequence[TranscriptSegment], Optional[Path]]:
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
                "readable_path": str(readable_path),
            },
        )
        start_time = perf_counter()
        assigner = TranscriptAssigner.from_config(context.episode, config)
        events = assigner.assign_from_readable(readable_path, metadata=context.episode.metadata, yield_progress=True)
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
        assignment_path = getattr(assigner, "_last_assignment_path", None)
        if assignment_path:
            assignment_path = Path(assignment_path)
        return assignment, assignment_path

    def _run_prettify(  # pragma: no cover - complex integration path
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        assignment_path: Path,
    ) -> Path:
        config = PrettifyConfig(data_root=self.config.data_dir)
        self._log_step_start(
            context,
            "prettify",
            {
                "mode": "execute",
                "assignment_path": str(assignment_path),
            },
        )
        start_time = perf_counter()
        prettifier = TranscriptPrettifier(
            podcast=context.episode,
            config=config,
            catalog=self.catalog,
        )
        events = prettifier.prettify(assignment_path=assignment_path, yield_progress=True)
        readable_path = self._consume_events(context, adapter, "prettify", events, context.step_plan)
        elapsed = perf_counter() - start_time
        self._log_step_end(
            context,
            "prettify",
            {
                "path": str(readable_path),
                "elapsed": round(elapsed, 2),
                "source": "fresh",
            },
        )
        return readable_path

    def _run_thematize(  # pragma: no cover - complex integration path
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        condensed_path: Path,
    ) -> Path:
        config = ThematizeConfig(
            data_root=self.config.data_dir,
            llm_model=self.config.ollama_model,
        )
        self._log_step_start(
            context,
            "thematize",
            {
                "mode": "execute",
                "condensed_path": str(condensed_path),
            },
        )
        start_time = perf_counter()
        thematizer = EpisodeThematizer(
            podcast=context.episode,
            config=config,
            catalog=self.catalog,
        )
        events = thematizer.thematize(condensed_path=condensed_path, yield_progress=True)
        themes_path = self._consume_events(context, adapter, "thematize", events, context.step_plan)
        elapsed = perf_counter() - start_time
        self._log_step_end(
            context,
            "thematize",
            {
                "path": str(themes_path),
                "elapsed": round(elapsed, 2),
                "source": "fresh",
            },
        )
        return themes_path

    def _run_classify(  # pragma: no cover - complex integration path
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        themes_path: Path,
    ) -> Path:
        config = ClassifyConfig(data_root=self.config.data_dir)
        self._log_step_start(
            context,
            "classify",
            {
                "mode": "execute",
                "themes_path": str(themes_path),
            },
        )
        start_time = perf_counter()
        classifier = EpisodeClassifier(
            podcast=context.episode,
            config=config,
            catalog=self.catalog,
        )
        events = classifier.classify(themes_path=themes_path, yield_progress=True)
        classified_path = self._consume_events(context, adapter, "classify", events, context.step_plan)
        elapsed = perf_counter() - start_time
        self._log_step_end(
            context,
            "classify",
            {
                "path": str(classified_path),
                "elapsed": round(elapsed, 2),
                "source": "fresh",
            },
        )
        return classified_path

    def _load_classified_path(self, context: PipelineContext) -> Optional[Path]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "classify")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("classified_path") or checkpoint.payload.get("path")
            if path:
                resolved = Path(path)
                if resolved.exists():
                    return resolved
        return None

    def _run_vocative(  # pragma: no cover - complex integration path
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        classified_path: Path,
    ) -> Path:
        config = VocativeConfig(
            data_root=self.config.data_dir,
            spacy_model=self.config.spacy_model,
            llm_model=self.config.ollama_model,
        )
        self._log_step_start(
            context,
            "vocative",
            {
                "mode": "execute",
                "classified_path": str(classified_path),
            },
        )
        start_time = perf_counter()
        detector = EpisodeVocativeDetector(
            podcast=context.episode,
            config=config,
            catalog=self.catalog,
        )
        events = detector.detect_vocatives(classified_path=classified_path, yield_progress=True)
        vocative_path = self._consume_events(context, adapter, "vocative", events, context.step_plan)
        elapsed = perf_counter() - start_time
        self._log_step_end(
            context,
            "vocative",
            {
                "path": str(vocative_path),
                "elapsed": round(elapsed, 2),
                "source": "fresh",
            },
        )
        return vocative_path

    def _load_vocative_path(self, context: PipelineContext) -> Optional[Path]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "vocative")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("vocative_path") or checkpoint.payload.get("path")
            if path:
                resolved = Path(path)
                if resolved.exists():
                    return resolved
        return None

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
        # Capture condensed path from prettify events even if checkpoints are not persisted (useful in tests)
        if step == "prettify":
            try:
                condensed_val = event.checkpoint.get("condensed_path")
            except Exception:
                condensed_val = None
            if not condensed_val:
                try:
                    condensed_val = (event.artefact_paths or {}).get("condensed")
                except Exception:
                    condensed_val = None
            if condensed_val:
                try:
                    self._last_condensed_path = Path(condensed_val)
                except Exception:
                    self._last_condensed_path = None
        if event.transient:
            self._check_stop()
            return

        checkpoint = adapter.process(event)
        self.queue.set_status(
            context.episode.episode_id,
            "In progress",
            event.message,
        )
        # Update pipeline status row
        try:
            if checkpoint.pipeline_id:
                last_completed = checkpoint.step if checkpoint.status == "completed" else None
                self.checkpoints.set_pipeline_status(
                    episode_id=context.episode.episode_id,
                    pipeline_id=checkpoint.pipeline_id,
                    status="in_progress",
                    current_step=step,
                    last_completed_step=last_completed,
                    progress=self._progress_for(step, event, step_plan),
                    remarks=event.message,
                )
        except Exception:
            LOGGER.debug("Pipeline status update failed for %s step %s", context.episode.episode_id, step)
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
        # Record intended current step early so that failures before first event still reflect the step
        try:
            if context.pipeline_id:
                self.checkpoints.set_pipeline_status(
                    episode_id=context.episode.episode_id,
                    pipeline_id=context.pipeline_id,
                    status="in_progress",
                    current_step=step,
                    remarks=f"Starting {step}",
                )
        except Exception:
            LOGGER.debug("Pipeline status early-step update failed for %s step %s", context.episode.episode_id, step)

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

    def _completed_steps(
        self,
        episode_id: str,
        checkpoints: Optional[Iterable[PipelineCheckpoint]] = None,
    ) -> Dict[str, str]:
        checkpoint_rows = checkpoints if checkpoints is not None else self.checkpoints.get_episode(episode_id)
        return {cp.step: cp.status for cp in checkpoint_rows if cp.status == "completed"}

    def _should_reset_checkpoints(
        self,
        item: QueueItem,
        plan: Sequence[str],
        checkpoints: Sequence[PipelineCheckpoint],
    ) -> bool:
        if item.resume:
            return False
        if not plan:
            return False
        if plan[0] != STEP_ORDER[0]:
            return False
        return bool(checkpoints)

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
        return "transcribe" not in plan and any(step in ("diarize",) for step in plan)

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
        if not any(step in plan for step in ("prettify", "assign")):
            return
        # Allow if diarization will run in this plan or is already completed
        if "diarize" in plan or "diarize" in completed_steps:
            return
        if diarization_results is None:
            raise RuntimeError(
                "Diarization results are required for prettify/assign. Include diarization before these steps."
            )

    def _validate_assignment_availability(
        self,
        plan: Sequence[str],
        readable_path: Optional[Path],
    ) -> None:
        if "assign" not in plan:
            return
        if readable_path is None or not readable_path.exists():
            raise RuntimeError(
                "Readable transcript artefact is required. Run prettify before assignment or include it in the plan."
            )

    def _validate_condensed_availability(
        self,
        plan: Sequence[str],
        condensed_path: Optional[Path],
    ) -> None:
        if "thematize" not in plan:
            return
        if condensed_path is None or not condensed_path.exists():
            raise RuntimeError(
                "Condensed transcript artefact is required for thematization. Run prettify first or include it in the plan."
            )

    def _validate_themes_availability(
        self,
        plan: Sequence[str],
        themes_path: Optional[Path],
    ) -> None:
        if "classify" not in plan:
            return
        if themes_path is None or not themes_path.exists():
            raise RuntimeError(
                "Thematized transcript artefact is required for classification. Run thematize first or include it in the plan."
            )

    def _validate_classified_availability(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
    ) -> None:
        if "vocative" not in plan:
            return
        # Allow if classify will run in this plan or is already completed
        if "classify" in plan or "classify" in completed_steps:
            return
        # Check if classified path exists from checkpoint
        # This is a soft check - actual validation happens in _run_vocative
        pass

    def _validate_vocative_availability(
        self,
        plan: Sequence[str],
        vocative_path: Optional[Path],
    ) -> None:
        # Vocative is optional for assign step, so we don't require it
        # This validation can be used in the future if needed
        pass

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
                "prettify": "Prettify complete",
                "thematize": "Thematization complete",
                "classify": "Classification complete",
                "vocative": "Vocative detection complete",
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
        try:
            # Mark stopped in status table
            # We do not know pipeline_id here; best-effort read last known
            status_row = self.checkpoints.get_pipeline_status(episode_id)
            if status_row:
                self.checkpoints.set_pipeline_status(
                    episode_id=episode_id,
                    pipeline_id=status_row["pipeline_id"],
                    status="stopped",
                    # Preserve the last known step to reflect where it failed/paused
                    current_step=status_row.get("current_step"),
                    remarks=remarks,
                )
        except Exception:
            LOGGER.debug("Pipeline status stop update failed for %s", episode_id)
        if not self.callback:
            return
        status = PipelineStatus(active=False, episode_id=episode_id, step="Stopped", message=remarks)
        payload = PipelineEventPayload(episode_id=episode_id, status=status, remarks=remarks)
        self.callback(PipelineStopped(payload))

    def _emit_complete(self, episode_id: str, remarks: str) -> None:
        try:
            status_row = self.checkpoints.get_pipeline_status(episode_id)
            if status_row:
                self.checkpoints.set_pipeline_status(
                    episode_id=episode_id,
                    pipeline_id=status_row["pipeline_id"],
                    status="completed",
                    current_step="Completed",
                    last_completed_step="assign",
                    progress=1.0,
                    remarks=remarks,
                )
        except Exception:
            LOGGER.debug("Pipeline status completion update failed for %s", episode_id)
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

    def _ensure_placeholder_assignment(
        self,
        context: PipelineContext,
        segments: Sequence[TranscriptSegment],
    ) -> Path:
        """
        Persist a placeholder 'assigned transcript' JSON that contains diarized segments
        with speaker_id placeholders and speaker_name mirroring the speaker_id. This allows
        the existing prettifier to operate deterministically before real name assignment.
        """
        cfg = PrettifyConfig(data_root=self.config.data_dir)
        assignment_path = cfg.assignment_path(context.episode, context.episode.episode_key).resolve()
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        for seg in segments:
            records.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text,
                    "speaker_id": seg.speaker_id or "UNKNOWN",
                    "speaker_name": seg.speaker_id or "UNKNOWN",
                }
            )
        assignment_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return assignment_path

