"""
Diarization pipeline for mywhisper.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

import logging

from .config import ensure_data_subdir, resolve_data_root
from .models import (
    AudioChunk,
    DiarizedTurn,
    PipelineEvent,
    PodcastEpisode,
    TranscriptSegment,
)

import torch
import torchaudio
import soundfile as sf
import numpy as np
import joblib
from pyannote.audio import Pipeline as PyAnnotePipeline
from pyannote.audio import Inference
from pyannote.core import Annotation, Segment
from sklearn.cluster import AgglomerativeClustering

LOGGER = logging.getLogger("mywhisper.diarize")


@dataclass(slots=True)
class DiarizationConfig:
    """
    Configuration for diarization pipelines.
    """

    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    chunk_minutes: float = 10.0
    overlap_seconds: float = 2.0
    embedding_window: str = "whole"
    output_dir: Path = field(default_factory=lambda: ensure_data_subdir("transcripts"))
    chunk_dir: Path = field(default_factory=lambda: ensure_data_subdir("audio_chunks"))
    data_root: Path = field(default_factory=resolve_data_root)
    device: Optional[str] = None

    def artefact_paths(
        self,
        podcast: PodcastEpisode,
        episode_key: Optional[str] = None,
    ) -> dict[str, Path]:
        key = episode_key or podcast.episode_key
        slug = podcast.artefact_slug()
        return {
            "chunk_dir": ensure_data_subdir(
                f"audio_chunks/{slug}/{key}", self.data_root
            ),
            "cluster_path": ensure_data_subdir(
                f"transcripts/{slug}", self.data_root
            )
            / f"{key}_clusters.pkl",
            "rttm_path": ensure_data_subdir(
                f"transcripts/rttm", self.data_root
            )
            / f"{slug}_{key}.rttm",
        }


class ChunkScheduler:
    """
    Create generator of audio chunks for diarization.
    """

    def __init__(
        self,
        config: DiarizationConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or LOGGER

    def schedule(self, podcast: PodcastEpisode, episode_key: str) -> Generator[AudioChunk, None, None]:
        chunk_seconds = max(self.config.chunk_minutes * 60.0, 1.0)
        overlap_seconds = max(self.config.overlap_seconds, 0.0)

        paths = self.config.artefact_paths(podcast, episode_key)
        chunk_dir = paths["chunk_dir"]

        waveform, sample_rate = torchaudio.load(str(podcast.source_path))
        channels, total_samples = waveform.shape

        chunk_samples = int(chunk_seconds * sample_rate)
        if chunk_samples <= 0:
            raise ValueError("chunk_minutes must resolve to a positive chunk duration.")

        overlap_samples = int(overlap_seconds * sample_rate)
        overlap_samples = min(overlap_samples, chunk_samples - 1)
        step = chunk_samples - overlap_samples
        if step <= 0:
            raise ValueError("overlap_seconds must be smaller than chunk duration.")

        index = 0
        for start in range(0, total_samples, step):
            end = min(start + chunk_samples, total_samples)
            chunk_tensor = waveform[:, start:end]

            global_start = start / sample_rate
            global_end = end / sample_rate

            chunk_path = chunk_dir / f"{podcast.artefact_slug()}__{episode_key}__chunk_{index:03d}.wav"

            chunk_np = chunk_tensor.transpose(0, 1).cpu().numpy()
            sf.write(str(chunk_path), chunk_np, sample_rate)

            yield AudioChunk(
                path=chunk_path,
                global_start=global_start,
                global_end=global_end,
                tensor=chunk_tensor,
                sample_rate=sample_rate,
                artefact_key=episode_key,
            )
            index += 1


class PyAnnotePipelineFactory:
    """
    Factory to create PyAnnote diarization components.
    """

    @staticmethod
    def create_pipeline(config: DiarizationConfig) -> PyAnnotePipeline:
        pipeline = PyAnnotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.hf_token,
        )
        if config.device:
            pipeline.to(config.device)
        return pipeline

    @staticmethod
    def create_embedding_inference(config: DiarizationConfig) -> Inference:
        return Inference(
            "pyannote/embedding",
            device=config.device,
            use_auth_token=config.hf_token,
            window=config.embedding_window,
        )


class SpeakerClusterer:
    """
    Manage clustering for diarized speaker embeddings.
    """

    def __init__(self, cluster_dir: Path) -> None:
        self.cluster_dir = cluster_dir
        self.cluster_path: Optional[Path] = None
        self.clusterer: Optional[AgglomerativeClustering] = None
        self.centroids: Optional["np.ndarray"] = None  # type: ignore[name-defined]
        self.num_speakers: Optional[int] = None
        self.reference_embeddings: Optional["np.ndarray"] = None  # type: ignore[name-defined]
        self.reference_labels: Optional["np.ndarray"] = None  # type: ignore[name-defined]

    def prepare(self, cluster_path: Path) -> None:
        self.cluster_path = cluster_path

    def fit_reference(self, embeddings: "np.ndarray", num_speakers: int) -> None:  # type: ignore[name-defined]
        if embeddings.size == 0:
            raise ValueError("No embeddings provided for reference clustering.")

        clusterer = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric="cosine",
            linkage="average",
        )
        clusterer.fit(embeddings)
        labels = clusterer.labels_
        unique_labels = np.unique(labels)
        centroids = np.stack(
            [embeddings[labels == label].mean(axis=0) for label in unique_labels]
        )

        self.clusterer = clusterer
        self.centroids = centroids
        self.reference_embeddings = embeddings
        self.reference_labels = labels
        self.num_speakers = num_speakers

    def assign(self, embeddings: "np.ndarray") -> List[int]:  # type: ignore[override, name-defined]
        if self.centroids is None:
            raise RuntimeError("Clusterer not initialised.")
        if embeddings.size == 0:
            return []

        emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cent_norm = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        emb_unit = embeddings / np.clip(emb_norm, a_min=1e-12, a_max=None)
        cent_unit = self.centroids / np.clip(cent_norm, a_min=1e-12, a_max=None)
        sims = emb_unit @ cent_unit.T
        return sims.argmax(axis=1).tolist()

    def save(self) -> None:
        if self.cluster_path is None:
            return
        payload = {
            "centroids": self.centroids,
            "num_speakers": self.num_speakers,
            "reference_embeddings": self.reference_embeddings,
            "reference_labels": self.reference_labels,
        }
        joblib.dump(payload, self.cluster_path)

    def load(self) -> None:
        if self.cluster_path is None:
            return
        if not self.cluster_path.exists():
            return
        data = joblib.load(self.cluster_path)
        self.centroids = data.get("centroids")
        self.reference_embeddings = data.get("reference_embeddings")
        self.reference_labels = data.get("reference_labels")
        self.num_speakers = data.get("num_speakers")


class DiarizationPipeline:
    """
    Orchestrates diarization workflow.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: DiarizationConfig,
        scheduler: ChunkScheduler,
        pipeline: PyAnnotePipeline,
        embedding_inference: Inference,
        clusterer: SpeakerClusterer,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config
        self.scheduler = scheduler
        self.pipeline = pipeline
        self.embedding_inference = embedding_inference
        self.clusterer = clusterer
        self.logger = logger or LOGGER.getChild(podcast.episode_id)

    @classmethod
    def from_config(cls, podcast: PodcastEpisode, config: DiarizationConfig) -> "DiarizationPipeline":
        scheduler = ChunkScheduler(config)
        pipeline = PyAnnotePipelineFactory.create_pipeline(config)
        embedding_inference = PyAnnotePipelineFactory.create_embedding_inference(config)
        cluster_dir = ensure_data_subdir(f"transcripts/{podcast.artefact_slug()}", config.data_root)
        clusterer = SpeakerClusterer(cluster_dir)
        return cls(
            podcast=podcast,
            config=config,
            scheduler=scheduler,
            pipeline=pipeline,
            embedding_inference=embedding_inference,
            clusterer=clusterer,
        )

    def run(  # pragma: no cover - depends on full pyannote pipeline
        self,
        transcript: Optional[Sequence[TranscriptSegment]] = None,
        yield_progress: bool = False,
    ) -> List[DiarizedTurn] | Generator[PipelineEvent, None, List[DiarizedTurn]]:
        pipeline = self._run_pipeline(transcript)
        if yield_progress:
            return pipeline

        try:
            while True:
                next(pipeline)
        except StopIteration as stop:
            return stop.value

    def write_json_transcript(
        self,
        transcript: Sequence[TranscriptSegment],
        target_path: Path,
    ) -> None:
        records = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "confidence": seg.confidence,
                "metadata": seg.metadata,
            }
            for seg in transcript
        ]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_pipeline(  # pragma: no cover - heavy external dependency orchestration
        self,
        transcript: Optional[Sequence[TranscriptSegment]],
    ) -> Generator[PipelineEvent, None, List[DiarizedTurn]]:
        episode_key = self.podcast.episode_key
        paths = self.config.artefact_paths(self.podcast, episode_key)
        self.clusterer.prepare(paths["cluster_path"])
        self.clusterer.load()

        start_time = time.perf_counter()
        yield PipelineEvent(
            stage="start",
            step_name="diarize",
            episode_id=self.podcast.episode_id,
            message=f"Starting diarization for {self.podcast.episode_title}",
            payload={
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "step": "started",
            },
            artefact_paths={
                "chunk_dir": paths["chunk_dir"],
                "cluster": paths["cluster_path"],
                "rttm": paths["rttm_path"],
            },
            checkpoint={
                "status": "started",
                "step": "diarize",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
            elapsed=0.0,
        )

        global_annotation = Annotation()
        diarized_turns: List[DiarizedTurn] = []
        target_num_speakers = self.clusterer.num_speakers or self.config.num_speakers

        for idx, chunk in enumerate(self.scheduler.schedule(self.podcast, episode_key)):
            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="chunk_ready",
                step_name="diarize",
                episode_id=self.podcast.episode_id,
                message=f"Scheduled chunk {idx}",
                payload={
                    "chunk_index": idx,
                    "chunk_path": str(chunk.path),
                    "start": chunk.global_start,
                    "end": chunk.global_end,
                    "step": "chunk_ready",
                },
                artefact_paths={"chunk": chunk.path},
                checkpoint={
                    "status": "chunk_ready",
                    "step": "diarize",
                    "chunk_index": idx,
                    "artefact_key": episode_key,
                    "episode_key": episode_key,
                },
                elapsed=elapsed,
            )

            annotation = self._run_diarization(chunk, target_num_speakers)

            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="annotation_ready",
                step_name="diarize",
                episode_id=self.podcast.episode_id,
                message=f"Diarized chunk {idx}",
                payload={
                    "chunk_index": idx,
                    "segment_count": len(list(annotation.itertracks())),
                    "step": "annotation_ready",
                },
                checkpoint={
                    "status": "annotation_ready",
                    "step": "diarize",
                    "chunk_index": idx,
                    "artefact_key": episode_key,
                    "episode_key": episode_key,
                },
                elapsed=elapsed,
            )

            embeddings, segments = self._extract_embeddings(chunk, annotation)
            if not segments:
                continue

            if self.clusterer.centroids is None:
                computed_speakers = target_num_speakers or self._count_speakers(annotation)
                if not computed_speakers:
                    raise ValueError("Unable to determine number of speakers for clustering.")
                self.clusterer.fit_reference(embeddings, computed_speakers)
                target_num_speakers = computed_speakers
                self.clusterer.save()

            assignments = self.clusterer.assign(embeddings)

            turns = self._commit_segments(global_annotation, chunk, segments, assignments)
            diarized_turns.extend(turns)

            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="cluster_assigned",
                step_name="diarize",
                episode_id=self.podcast.episode_id,
                message=f"Assigned speakers for chunk {idx}",
                payload={
                    "chunk_index": idx,
                    "turns": len(turns),
                    "step": "cluster_assigned",
                },
                checkpoint={
                    "status": "cluster_assigned",
                    "step": "diarize",
                    "chunk_index": idx,
                    "artefact_key": episode_key,
                    "episode_key": episode_key,
                },
                elapsed=elapsed,
            )

        self._write_rttm(paths["rttm_path"], global_annotation)
        self._last_rttm_path = paths["rttm_path"]
        self._last_episode_key = episode_key
        self._last_artefact_key = episode_key

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="persisted",
            step_name="diarize",
            episode_id=self.podcast.episode_id,
            message="Persisted diarization RTTM",
            payload={
                "path": str(paths["rttm_path"]),
                "turns": len(diarized_turns),
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "step": "completed",
            },
            artefact_paths={
                "rttm": paths["rttm_path"],
                "cluster_dir": paths["cluster_path"],
            },
            checkpoint={
                "status": "completed",
                "step": "diarize",
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "rttm_path": str(paths["rttm_path"]),
                "turns": len(diarized_turns),
            },
            elapsed=elapsed,
        )

        return sorted(diarized_turns, key=lambda turn: (turn.start, turn.end))

    def _run_diarization(  # pragma: no cover - delegates to pyannote pipeline
        self,
        chunk: AudioChunk,
        num_speakers: Optional[int],
    ) -> Annotation:
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        return self.pipeline(str(chunk.path), **kwargs)

    def _extract_embeddings(  # pragma: no cover - relies on pyannote embedding inference
        self,
        chunk: AudioChunk,
        annotation: Annotation,
    ) -> Tuple["np.ndarray", List[Segment]]:  # type: ignore[name-defined]
        min_duration = getattr(self.embedding_inference, "duration", None) or 2.0
        file_duration = chunk.duration()
        embeddings: List["np.ndarray"] = []  # type: ignore[name-defined]
        segments: List[Segment] = []

        for segment, _, _ in annotation.itertracks(yield_label=True):
            padded = self._ensure_min_duration(segment, min_duration, file_duration)
            emb = self.embedding_inference.crop(str(chunk.path), padded)
            if isinstance(emb, tuple):
                emb = emb[0]
            arr = np.asarray(emb).squeeze()
            if arr.ndim == 0:
                continue
            embeddings.append(arr)
            segments.append(segment)

        if not embeddings:
            return np.empty((0, 0)), []
        return np.stack(embeddings), segments

    @staticmethod
    def _ensure_min_duration(segment: Segment, min_duration: float, file_duration: float) -> Segment:
        if segment.duration >= min_duration:
            return segment
        center = segment.middle
        half = min_duration / 2.0
        start = max(0.0, center - half)
        end = min(file_duration, center + half)
        if end - start < min_duration:
            if start == 0.0:
                end = min(file_duration, min_duration)
            elif end == file_duration:
                start = max(0.0, file_duration - min_duration)
        return Segment(start, end)

    def _count_speakers(self, annotation: Annotation) -> int:
        labels = {label for _, _, label in annotation.itertracks(yield_label=True)}
        return len(labels)

    def _commit_segments(
        self,
        global_annotation: Annotation,
        chunk: AudioChunk,
        segments: Sequence[Segment],
        assignments: Sequence[int],
    ) -> List[DiarizedTurn]:  # pragma: no cover - integration with pyannote Annotation
        turns: List[DiarizedTurn] = []
        for seg, speaker_idx in zip(segments, assignments):
            speaker_label = f"SPEAKER_{speaker_idx:02d}"
            global_segment = Segment(
                chunk.global_start + seg.start,
                chunk.global_start + seg.end,
            )
            global_annotation[global_segment] = speaker_label
            turns.append(
                DiarizedTurn(
                    start=global_segment.start,
                    end=global_segment.end,
                    speaker_id=speaker_label,
                )
            )
        return turns

    def _write_rttm(self, path: Path, annotation: Annotation) -> None:  # pragma: no cover
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            annotation.write_rttm(handle)



