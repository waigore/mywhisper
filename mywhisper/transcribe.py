"""
Transcription pipeline implementation for mywhisper.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from queue import SimpleQueue
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Sequence

import logging

from .config import ensure_data_subdir, resolve_data_root
from .models import AudioChunk, PipelineEvent, PodcastEpisode, TranscriptSegment

import torch
from pywhispercpp.model import Model
import torchaudio
import soundfile as sf

LOGGER = logging.getLogger("mywhisper.transcribe")

WHISPER_TIME_FACTOR = 100.0


@dataclass(slots=True)
class TranscriptionConfig:
    """
    Configuration for podcast transcription.
    """

    model_path: Path
    language: str = "en"
    target_sample_rate: int = 16000
    chunk_duration: Optional[float] = None
    chunk_overlap: float = 0.0
    output_dir: Path = field(default_factory=lambda: ensure_data_subdir("transcripts"))
    extract_dir: Path = field(default_factory=lambda: ensure_data_subdir("audio_chunks"))
    device: Optional[str] = None
    data_root: Path = field(default_factory=resolve_data_root)

    def transcript_path(
        self,
        podcast: PodcastEpisode,
        episode_key: Optional[str] = None,
    ) -> Path:
        key = episode_key or podcast.episode_key
        slug = podcast.artefact_slug()
        target_dir = ensure_data_subdir(f"transcripts/{slug}", self.data_root)
        return target_dir / f"{key}_whisper.json"

    def chunk_dir(
        self,
        podcast: PodcastEpisode,
        episode_key: Optional[str] = None,
    ) -> Path:
        key = episode_key or podcast.episode_key
        slug = podcast.artefact_slug()
        return ensure_data_subdir(f"audio_chunks/{slug}/{key}", self.data_root)


class WhisperModelFactory:
    """
    Factory responsible for creating Whisper model instances.
    """

    @staticmethod
    def create(config: TranscriptionConfig) -> Model:
        if Model is object:
            raise RuntimeError("pywhispercpp is not available in this environment.")
        LOGGER.debug("Loading Whisper model from %s", config.model_path)
        return Model(str(config.model_path), language=config.language, print_realtime=False)


class AudioChunker:
    """
    Create audio chunks suitable for Whisper transcription.
    """

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config

    @property
    def chunk_duration(self) -> Optional[float]:
        return self.config.chunk_duration

    @property
    def chunk_overlap(self) -> float:
        return self.config.chunk_overlap

    @property
    def target_sample_rate(self) -> int:
        return self.config.target_sample_rate

    def iterate_chunks(
        self,
        podcast: PodcastEpisode,
        episode_key: str,
    ) -> Generator[AudioChunk, None, None]:
        """
        Yield audio chunks for the podcast.
        """

        waveform, sample_rate = torchaudio.load(str(podcast.source_path))
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0).contiguous()

        if sample_rate != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            waveform = resample(waveform)
            sample_rate = self.target_sample_rate

        total_samples = waveform.shape[-1]
        total_duration = total_samples / sample_rate

        if not self.chunk_duration:
            yield AudioChunk(
                path=podcast.source_path,
                global_start=0.0,
                global_end=total_duration,
                tensor=waveform,
                sample_rate=sample_rate,
                artefact_key=episode_key,
            )
            return

        chunk_dir = self.config.chunk_dir(podcast, episode_key)
        chunk_duration = max(self.chunk_duration, 0.1)
        chunk_samples = int(chunk_duration * sample_rate)
        if chunk_samples <= 0:
            raise ValueError("chunk_duration must produce at least one sample.")
        overlap_samples = int(self.chunk_overlap * sample_rate)
        overlap_samples = min(overlap_samples, chunk_samples - 1)
        step = chunk_samples - overlap_samples

        index = 0
        for start in range(0, total_samples, step):
            end = min(start + chunk_samples, total_samples)
            chunk_tensor = waveform[start:end]
            start_sec = start / sample_rate
            end_sec = end / sample_rate
            chunk_path = chunk_dir / f"{podcast.artefact_slug()}__{episode_key}__chunk_{index:03d}.wav"

            chunk_np = chunk_tensor.cpu().numpy().astype("float32", copy=False)
            sf.write(str(chunk_path), chunk_np, sample_rate)

            yield AudioChunk(
                path=chunk_path,
                global_start=start_sec,
                global_end=end_sec,
                tensor=chunk_tensor,
                sample_rate=sample_rate,
                artefact_key=episode_key,
            )
            index += 1


class PodcastTranscriber:
    """
    Pipeline class orchestrating podcast transcription via Whisper.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: TranscriptionConfig,
        model: Model,
        chunker: AudioChunker,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config
        self.model = model
        self.chunker = chunker
        self.logger = logger or LOGGER.getChild(podcast.episode_id)

    @classmethod
    def from_config(
        cls,
        podcast: PodcastEpisode,
        config: TranscriptionConfig,
    ) -> "PodcastTranscriber":
        model = WhisperModelFactory.create(config)
        chunker = AudioChunker(config)
        return cls(podcast, config, model, chunker)

    def transcribe(
        self,
        yield_progress: bool = False,
    ) -> List[TranscriptSegment] | Generator[PipelineEvent, None, List[TranscriptSegment]]:
        """
        Execute the transcription pipeline.
        """

        pipeline = self._transcription_pipeline()
        if yield_progress:
            return pipeline

        try:
            while True:
                next(pipeline)
        except StopIteration as stop:
            return stop.value

    def load_cached_segments(
        self,
        episode_key: Optional[str] = None,
        *,
        artefact_key: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        """
        Load a previously persisted transcript if available.
        """

        key = episode_key or artefact_key
        if not key:
            key = getattr(self, "_last_episode_key", None) or getattr(self, "_last_artefact_key", None)
        if not key:
            key = self.podcast.episode_key
        if not key:
            raise ValueError("No episode key specified, and no cached transcription available.")

        path = self.config.transcript_path(self.podcast, key)
        if not path.exists():
            raise FileNotFoundError(f"Transcript not found at {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        segments: List[TranscriptSegment] = []
        for item in data:
            segments.append(
                TranscriptSegment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item["text"]),
                    speaker_id=item.get("speaker_id"),
                    speaker_name=item.get("speaker_name"),
                    confidence=item.get("confidence"),
                    justification=item.get("justification"),
                    metadata=item.get("metadata", {}),
                )
            )
        return segments

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _transcription_pipeline(
        self,
    ) -> Generator[PipelineEvent, None, List[TranscriptSegment]]:
        episode_key = self.podcast.episode_key
        transcript_path = self.config.transcript_path(self.podcast, episode_key)
        start_time = time.perf_counter()
        segments: List[TranscriptSegment] = []

        yield PipelineEvent(
            stage="start",
            step_name="transcribe",
            episode_id=self.podcast.episode_id,
            message=f"Starting transcription for {self.podcast.episode_title}",
            payload={
                "episode_id": self.podcast.episode_id,
                "source": str(self.podcast.source_path),
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "step": "started",
            },
            checkpoint={
                "status": "started",
                "step": "transcribe",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
            artefact_paths={"transcript": transcript_path},
            elapsed=0.0,
        )

        for idx, chunk in enumerate(self.chunker.iterate_chunks(self.podcast, episode_key)):
            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="chunk_prepared",
                step_name="transcribe",
                episode_id=self.podcast.episode_id,
                message=f"Prepared chunk {idx}",
                payload={
                    "chunk_index": idx,
                    "chunk_path": str(chunk.path),
                    "start": chunk.global_start,
                    "end": chunk.global_end,
                    "step": "chunk_prepared",
                },
                artefact_paths={"chunk": chunk.path},
                checkpoint={
                    "status": "chunk_prepared",
                    "step": "transcribe",
                    "chunk_index": idx,
                    "artefact_key": episode_key,
                    "episode_key": episode_key,
                },
                elapsed=elapsed,
            )

            chunk_segments = yield from self._transcribe_chunk(chunk, idx)
            segments.extend(chunk_segments)

            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="chunk_transcribed",
                step_name="transcribe",
                episode_id=self.podcast.episode_id,
                message=f"Transcribed chunk {idx}",
                payload={
                    "chunk_index": idx,
                    "segment_count": len(chunk_segments),
                    "step": "chunk_transcribed",
                },
                checkpoint={
                    "status": "chunk_completed",
                    "step": "transcribe",
                    "chunk_index": idx,
                    "artefact_key": episode_key,
                    "episode_key": episode_key,
                },
                elapsed=elapsed,
            )

        self._persist_transcript(transcript_path, segments)
        self._last_transcript_path = transcript_path
        self._last_episode_key = episode_key
        self._last_artefact_key = episode_key

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="persisted",
            step_name="transcribe",
            episode_id=self.podcast.episode_id,
            message="Persisted transcript",
            payload={
                "path": str(transcript_path),
                "segment_count": len(segments),
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "step": "completed",
            },
            artefact_paths={"transcript": transcript_path},
            checkpoint={
                "status": "completed",
                "step": "transcribe",
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "transcript_path": str(transcript_path),
                "segment_count": len(segments),
            },
            elapsed=elapsed,
        )

        return segments

    def _transcribe_chunk(
        self,
        chunk: AudioChunk,
        chunk_index: int,
    ) -> Generator[PipelineEvent, None, List[TranscriptSegment]]:
        if not hasattr(self.model, "transcribe"):
            raise RuntimeError("Whisper model does not implement `transcribe`.")

        if chunk.tensor is None:
            waveform, sample_rate = torchaudio.load(str(chunk.path))
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            chunk.tensor = waveform
            chunk.sample_rate = sample_rate

        tensor = chunk.tensor
        if tensor.dim() > 1:
            tensor = tensor.squeeze(0)
        audio_np = tensor.detach().cpu().numpy().astype("float32", copy=False)

        segment_queue: SimpleQueue = SimpleQueue()
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def enqueue_segment(segment: Any) -> None:
            segment_queue.put(segment)

        def worker() -> None:
            try:
                result_holder["segments"] = self.model.transcribe(
                    audio_np,
                    language=self.config.language,
                    new_segment_callback=enqueue_segment,
                )
            except BaseException as exc:  # pragma: no cover - defensive
                error_holder["error"] = exc
            finally:
                segment_queue.put(None)

        worker_thread = threading.Thread(
            target=worker,
            name=f"whisper-segment-{chunk_index}",
            daemon=True,
        )
        worker_thread.start()

        segment_counter = 0
        while True:
            segment = segment_queue.get()
            if segment is None:
                break

            text = str(getattr(segment, "text", "")).strip()
            preview = text if len(text) <= 80 else f"{text[:77]}..."
            start_seconds = (float(getattr(segment, "t0", 0.0)) / WHISPER_TIME_FACTOR) + chunk.global_start
            end_seconds = (float(getattr(segment, "t1", 0.0)) / WHISPER_TIME_FACTOR) + chunk.global_start
            end_seconds = max(start_seconds, end_seconds)

            yield PipelineEvent(
                stage="segment_detected",
                step_name="transcribe",
                episode_id=self.podcast.episode_id,
                message=f"Chunk {chunk_index} segment {segment_counter + 1}: {preview or '[silence]'}",
                payload={
                    "chunk_index": chunk_index,
                    "segment_index": segment_counter,
                    "start": start_seconds,
                    "end": end_seconds,
                    "text": text,
                    "step": "segment_detected",
                },
                checkpoint={
                    "status": "in_progress",
                    "step": "transcribe",
                    "chunk_index": chunk_index,
                    "segment_index": segment_counter,
                },
                transient=True,
            )
            segment_counter += 1

        worker_thread.join()
        if "error" in error_holder:
            raise error_holder["error"]

        whisper_segments = result_holder.get("segments") or []
        normalized: List[TranscriptSegment] = []
        for seg in whisper_segments:
            start_seconds = (seg.t0 / WHISPER_TIME_FACTOR) + chunk.global_start
            end_seconds = (seg.t1 / WHISPER_TIME_FACTOR) + chunk.global_start
            end_seconds = max(start_seconds, end_seconds)
            normalized.append(
                TranscriptSegment(
                    start=start_seconds,
                    end=end_seconds,
                    text=str(seg.text).strip(),
                )
            )
        return normalized

    def _persist_transcript(self, path: Path, segments: Sequence[TranscriptSegment]) -> None:
        records = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "confidence": seg.confidence,
                "justification": seg.justification,
                "metadata": seg.metadata,
            }
            for seg in segments
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)



