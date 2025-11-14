from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

from ...assign import AssignmentConfig, TranscriptAssigner
from ...checkpoints import PipelineEventAdapter, CheckpointStore
from ...diarize import DiarizationConfig, DiarizationPipeline
from ...models import PipelineEvent, PodcastEpisode, TranscriptSegment
from ...podcasts import PodcastCatalog
from ...transcribe import PodcastTranscriber, TranscriptionConfig
from ..config import MywConfig
from ..messages import PipelineCompleted, PipelineEventPayload, PipelineProgress, PipelineStopped
from ..models import PipelineStatus
from .queue import QueueController, QueueItem

LOGGER = logging.getLogger("mywhisper.myw.pipeline")

ProgressCallback = Callable[[PipelineProgress | PipelineStopped | PipelineCompleted], None]

STEP_ORDER = ("transcribe", "diarize", "assign")


class PipelineInterrupted(Exception):
    """Raised when a stop is requested."""


@dataclass(slots=True)
class PipelineContext:
    episode: PodcastEpisode
    resume: bool
    completed_steps: Dict[str, str]


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
                self.queue.set_status(context.episode.episode_id, "Completed", "Pipeline finished")
                self._emit_complete(context.episode.episode_id, "Pipeline completed")
            finally:
                self.queue.release_current()

    def _process_episode(self, context: PipelineContext) -> None:
        adapter = PipelineEventAdapter(self.checkpoints, episode_id=context.episode.episode_id)

        transcript_segments: Optional[Sequence[TranscriptSegment]] = None
        if "transcribe" not in context.completed_steps:
            transcript_segments = self._run_transcription(context, adapter)
        else:
            transcript_segments = self._load_transcript_segments(context)
        self._check_stop()
        if transcript_segments is None:
            raise RuntimeError("Transcript segments unavailable; cannot continue pipeline.")

        diarization_results = None
        if "diarize" not in context.completed_steps:
            diarization_results = self._run_diarization(context, transcript_segments, adapter)
        else:
            diarization_results = self._load_diarization_results(context)
        self._check_stop()

        if "assign" not in context.completed_steps and transcript_segments is not None:
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
        transcriber = PodcastTranscriber.from_config(context.episode, config)
        events = transcriber.transcribe(yield_progress=True)
        return self._consume_events(context, adapter, "transcribe", events)

    def _load_transcript_segments(
        self,
        context: PipelineContext,
    ) -> Optional[Sequence[TranscriptSegment]]:
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "transcribe")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("transcript_path") or checkpoint.payload.get("path")
            if path:
                transcript_path = Path(path)
                try:
                    return self._read_transcript(transcript_path)
                except Exception:
                    LOGGER.debug("Failed to load cached transcript at %s", transcript_path)
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
        )
        pipeline = DiarizationPipeline.from_config(context.episode, config)
        events = pipeline.run(transcript_segments, yield_progress=True)
        return self._consume_events(context, adapter, "diarize", events)

    def _load_diarization_results(self, context: PipelineContext):
        checkpoint = self.checkpoints.get_step(context.episode.episode_id, "diarize")
        if checkpoint and checkpoint.status == "completed":
            path = checkpoint.details.get("rttm_path")
            if path:
                return path
        return None

    def _run_assignment(
        self,
        context: PipelineContext,
        segments: Sequence[TranscriptSegment],
        adapter: PipelineEventAdapter,
    ) -> None:
        config = AssignmentConfig(
            data_root=self.config.data_dir,
            ollama_model=self.config.ollama_model,
            spacy_model=self.config.spacy_model,
        )
        assigner = TranscriptAssigner.from_config(context.episode, config)
        events = assigner.assign_names(segments, metadata=context.episode.metadata, yield_progress=True)
        self._consume_events(context, adapter, "assign", events)

    def _consume_events(
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        step: str,
        events: Iterable[PipelineEvent],
    ):
        iterator = iter(events)
        result = None
        while True:
            try:
                event = next(iterator)
            except StopIteration as stop:
                result = stop.value
                break
            checkpoint = adapter.process(event)
            self.queue.set_status(
                context.episode.episode_id,
                "In progress",
                event.message,
            )
            status = PipelineStatus(
                active=True,
                episode_id=context.episode.episode_id,
                step=step,
                progress=self._progress_for(event),
                message=event.message,
            )
            self._emit_progress(context.episode.episode_id, status, event.message)
            self._check_stop()
        return result

    def _completed_steps(self, episode_id: str) -> Dict[str, str]:
        checkpoints = self.checkpoints.get_episode(episode_id)
        return {cp.step: cp.status for cp in checkpoints if cp.status == "completed"}

    def _check_stop(self) -> None:
        if self.queue.should_stop():
            raise PipelineInterrupted()

    def _progress_for(self, event: PipelineEvent) -> float:
        step = event.step_name or event.stage
        try:
            index = STEP_ORDER.index(step)
        except ValueError:
            index = 0
        completed = index / len(STEP_ORDER)
        if event.checkpoint.get("status") == "completed":
            completed = (index + 1) / len(STEP_ORDER)
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
                    speaker_id=item.get("speaker_id"),
                    speaker_name=item.get("speaker_name"),
                    confidence=item.get("confidence"),
                    justification=item.get("justification"),
                    metadata=item.get("metadata", {}),
                )
            )
        return segments

