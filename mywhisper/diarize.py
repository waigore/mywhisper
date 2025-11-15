"""
Diarization pipeline for mywhisper.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torchaudio
from pyannote.audio import Pipeline as PyAnnotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation
from tqdm.auto import tqdm

from .config import ensure_data_subdir, resolve_data_root
from .models import DiarizedTurn, PodcastEpisode, TranscriptSegment

LOGGER = logging.getLogger("mywhisper.diarize")

ProgressHookFactory = Callable[[], ProgressHook]


@dataclass(slots=True)
class DiarizationConfig:
    """
    Configuration for diarization pipelines.
    """

    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    target_sample_rate: int = 16000
    output_dir: Path = field(default_factory=lambda: ensure_data_subdir("transcripts"))
    rttm_dir: Path = field(default_factory=lambda: ensure_data_subdir("transcripts/rttm"))
    data_root: Path = field(default_factory=resolve_data_root)
    device: Optional[torch.device | str] = None
    progress_hook_factory: Optional[ProgressHookFactory] = None

    def artefact_paths(
        self,
        podcast: PodcastEpisode,
        episode_key: Optional[str] = None,
    ) -> dict[str, Path]:
        key = episode_key or podcast.episode_key
        slug = podcast.artefact_slug()
        transcript_dir = ensure_data_subdir(f"transcripts/{slug}", self.data_root)
        rttm_dir = ensure_data_subdir("transcripts/rttm", self.data_root)
        return {
            "transcript_dir": transcript_dir,
            "json_path": transcript_dir / f"{key}_diarization.json",
            "rttm_path": rttm_dir / f"{slug}_{key}.rttm",
        }


class WaveformLoader:
    """
    Normalize podcast audio into the waveform dict expected by PyAnnote.
    """

    def __init__(self, target_sample_rate: int = 16000) -> None:
        self.target_sample_rate = target_sample_rate
        self.logger = LOGGER.getChild("waveform_loader")

    def load(self, path: Path) -> Dict[str, torch.Tensor | int]:
        self.logger.info(
            "Loading waveform for diarization (target_sample_rate=%s, path=%s)",
            self.target_sample_rate,
            str(path),
        )
        waveform, sample_rate = torchaudio.load(str(path))
        self.logger.debug(
            "Loaded waveform",
            extra={"path": str(path), "sample_rate": sample_rate, "channels": waveform.shape[0]},
        )

        if sample_rate != self.target_sample_rate:
            self.logger.info(
                "Resampling waveform for diarization from %s Hz to %s Hz",
                sample_rate,
                self.target_sample_rate,
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            self.logger.info(
                "Downmixing waveform to mono for diarization (channels=%s)",
                waveform.shape[0],
            )
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "waveform": waveform.contiguous(),
            "sample_rate": self.target_sample_rate,
        }


class TqdmProgressHook(ProgressHook):
    """
    Mirror PyAnnote pipeline progress through tqdm progress bars.
    """

    def __init__(self) -> None:
        super().__init__(hidden=True)
        self._bars: Dict[str, "tqdm"] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()
        return False

    def __call__(
        self,
        step_name,
        step_artifact,
        file=None,
        total=None,
        completed=None,
    ):
        total = total or completed or 1
        completed = completed or total

        bar = self._bars.get(step_name)
        if bar is None:
            bar = tqdm(total=total, desc=f"{step_name}", unit="step", leave=True)
            self._bars[step_name] = bar
        else:
            if bar.total != total:
                bar.total = total

        delta = completed - bar.n
        if delta > 0:
            bar.update(delta)

        if completed >= total:
            bar.close()
            self._bars.pop(step_name, None)


class PyAnnotePipelineFactory:
    """
    Factory to create PyAnnote diarization components.
    """

    @staticmethod
    def create_pipeline(config: DiarizationConfig) -> PyAnnotePipeline:
        target_device_str: Optional[str] = None
        resolved_device: Optional[torch.device] = None
        if config.device:
            if isinstance(config.device, torch.device):
                resolved_device = config.device
                target_device_str = str(config.device)
            else:
                target_device_str = str(config.device)
                try:
                    resolved_device = torch.device(config.device)
                except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Invalid device %s for diarization: %s", config.device, exc)
                    resolved_device = None
        else:
            target_device_str = "cpu"

        LOGGER.info(
            "Loading PyAnnote pipeline model=%s device=%s",
            "pyannote/speaker-diarization-community-1",
            target_device_str,
        )
        pipeline = PyAnnotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=config.hf_token,
        )
        if resolved_device is not None:
            pipeline.to(resolved_device)
            LOGGER.info("Moved PyAnnote pipeline to device=%s", resolved_device)
        return pipeline


class DiarizationPipeline:
    """
    Orchestrates diarization workflow.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: DiarizationConfig,
        pipeline: PyAnnotePipeline,
        waveform_loader: WaveformLoader,
        progress_hook_factory: Optional[ProgressHookFactory] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config
        self.pipeline = pipeline
        self.waveform_loader = waveform_loader
        self.progress_hook_factory = progress_hook_factory or TqdmProgressHook
        self.logger = logger or LOGGER.getChild(podcast.episode_id)

    @classmethod
    def from_config(cls, podcast: PodcastEpisode, config: DiarizationConfig) -> "DiarizationPipeline":
        pipeline = PyAnnotePipelineFactory.create_pipeline(config)
        waveform_loader = WaveformLoader(config.target_sample_rate)
        progress_hook_factory = config.progress_hook_factory or TqdmProgressHook
        return cls(
            podcast=podcast,
            config=config,
            pipeline=pipeline,
            waveform_loader=waveform_loader,
            progress_hook_factory=progress_hook_factory,
        )

    def run(self) -> List[DiarizedTurn]:  # pragma: no cover - depends on pyannote model
        """
        Execute PyAnnote diarization on the full audio file.
        """

        episode_key = self.podcast.episode_key
        paths = self.config.artefact_paths(self.podcast, episode_key)
        payload = self.waveform_loader.load(self.podcast.source_path)

        pipeline_kwargs = {}
        if self.config.num_speakers:
            pipeline_kwargs["num_speakers"] = self.config.num_speakers
        self.logger.info(
            "Starting diarization run episode=%s source=%s device=%s num_speakers=%s",
            self.podcast.episode_id,
            str(self.podcast.source_path),
            self.config.device or "cpu",
            pipeline_kwargs.get("num_speakers"),
        )
        annotation = self._diarize(payload, pipeline_kwargs)
        self._write_rttm(paths["rttm_path"], annotation)
        self.logger.info(
            "Diarization completed episode=%s rttm=%s turns=%s",
            self.podcast.episode_id,
            str(paths["rttm_path"]),
            len(list(annotation.itertracks())),
        )
        return self._annotation_to_turns(annotation)

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
                "speaker_id": seg.speaker_id,
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

    def _diarize(
        self,
        payload: Dict[str, torch.Tensor | int],
        pipeline_kwargs: Dict[str, object],
    ) -> Annotation:
        hook = self._build_progress_hook()
        context = hook if hasattr(hook, "__enter__") else contextlib.nullcontext(hook)
        with context as active_hook:
            diarization = self.pipeline(
                payload,
                hook=active_hook,
                **pipeline_kwargs,
            )
        return diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization

    def _build_progress_hook(self) -> ProgressHook:
        factory = self.progress_hook_factory or TqdmProgressHook
        hook = factory()
        if not isinstance(hook, ProgressHook):
            raise TypeError("progress_hook_factory must return a ProgressHook instance.")
        return hook

    def _annotation_to_turns(self, annotation: Annotation) -> List[DiarizedTurn]:
        turns: List[DiarizedTurn] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            turns.append(
                DiarizedTurn(
                    start=segment.start,
                    end=segment.end,
                    speaker_id=str(label),
                )
            )
        return sorted(turns, key=lambda turn: (turn.start, turn.end))

    def _write_rttm(self, path: Path, annotation: Annotation) -> None:  # pragma: no cover
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            annotation.write_rttm(handle)