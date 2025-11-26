from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence

from .llm_client import LLMClient, OllamaClient
from .config import ensure_episode_subdir, resolve_data_root
from .logging_utils import LoggingBase
from .models import PipelineEvent, PodcastEpisode
from .podcasts import PodcastCatalog

LOGGER = logging.getLogger("mywhisper.thematize")

DEFAULT_PROMPT_TEMPLATE = """You are an analyst who summarizes podcast transcripts into thematic sections.
Return a JSON array. Each element must include:
- "theme": short title (<=6 words)
- "summary": 2-3 sentences describing the discussion
- Optional "highlights": array of concise bullet strings

Focus on the provided transcript chunk only.

Podcast: {show_title} – {episode_title}
Transcript chunk:
{chunk}
"""


@dataclass(slots=True)
class ThematizeConfig:
    """
    Configuration for LLM-driven thematization.
    """

    data_root: Path = field(default_factory=resolve_data_root)
    max_tokens_per_chunk: int = 2000
    chunk_overlap_ratio: float = 0.15
    llm_model: str = "llama3"
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    fallback_theme: str = "Episode Overview"
    output_subdir: str = "transcripts"

    def condensed_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_condensed.json"

    def themes_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_with_themes.json"


class EpisodeThematizer(LoggingBase):
    """
    Convert readable transcripts into structured theme sections via LLM prompts.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: Optional[ThematizeConfig] = None,
        catalog: Optional[PodcastCatalog] = None,
        client: Optional[LLMClient] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config or ThematizeConfig()
        self.catalog = catalog
        self.client = client or OllamaClient(self.config.llm_model)
        base_logger = logger or LOGGER
        self.logger = base_logger.getChild(podcast.episode_id)
        self._last_readable_path: Optional[Path] = None
        self._last_themes_path: Optional[Path] = None

    def thematize(
        self,
        *,
        condensed_path: Optional[Path] = None,
        yield_progress: bool = False,
    ) -> Path | Generator[PipelineEvent, None, Path]:
        pipeline = self._pipeline(condensed_path=condensed_path)
        if yield_progress:
            return pipeline

        try:
            while True:
                next(pipeline)
        except StopIteration as stop:
            return stop.value

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _pipeline(self, condensed_path: Optional[Path]) -> Generator[PipelineEvent, None, Path]:
        episode_key = self.podcast.episode_key
        resolved_condensed = condensed_path or self.config.condensed_path(self.podcast, episode_key)
        resolved_condensed = resolved_condensed.resolve()
        themes_path = self.config.themes_path(self.podcast, episode_key).resolve()

        if not resolved_condensed.exists():
            raise FileNotFoundError(f"Condensed transcript not found at {resolved_condensed}")

        start_time = time.perf_counter()
        self.logger.info(
            "Thematization start | episode=%s | condensed=%s | output=%s",
            self.podcast.episode_id,
            resolved_condensed,
            themes_path,
        )

        yield PipelineEvent(
            stage="thematize",
            step_name="thematize",
            episode_id=self.podcast.episode_id,
            message="Loading condensed transcript",
            payload={
                "episode_key": episode_key,
                "condensed_path": str(resolved_condensed),
                "step": "load",
            },
            artefact_paths={"condensed": resolved_condensed},
            checkpoint={
                "status": "started",
                "step": "thematize",
                "condensed_path": str(resolved_condensed),
                "episode_key": episode_key,
            },
        )

        # Load condensed records and process per segment (no overlap)
        records = json.loads(resolved_condensed.read_text(encoding="utf-8"))
        segments: List[dict] = [rec for rec in records if isinstance(rec, dict)]
        segment_count = len(segments)

        enriched: List[dict] = []
        failure_count = 0

        for index, seg in enumerate(segments):
            try:
                theme, summary = self._invoke_llm_for_segment(seg, index)
                enriched.append(
                    {
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "speaker_id": str(seg.get("speaker_id") or "UNKNOWN"),
                        "speaker_name": str(seg.get("speaker_name") or seg.get("speaker_id") or "UNKNOWN"),
                        "text": str(seg.get("text") or "").strip(),
                        "theme": theme,
                        "summary": summary,
                    }
                )
            except Exception as exc:  # pragma: no cover - fallback path exercised separately
                failure_reason = str(exc)
                failure_count += 1
                self.logger.warning("LLM generation failed on segment %d: %s", index, exc)
                # Use fallback for this segment but preserve transcript data
                enriched.append(self._create_fallback_segment(seg, index, failure_reason))

            yield PipelineEvent(
                stage="thematize",
                step_name="thematize",
                episode_id=self.podcast.episode_id,
                message=f"Processed segment {index + 1}/{segment_count}",
                payload={
                    "segment_index": index,
                    "step": "segment_completed",
                },
                checkpoint={
                    "status": f"segment_{index + 1}",
                    "step": "thematize",
                    "segment_index": index,
                },
            )

        if not enriched:
            # Only use full fallback if no segments were processed at all
            enriched = self._fallback_segments(None)
        elif failure_count > 0:
            self.logger.warning(
                "Completed thematization with %d fallback segments out of %d total",
                failure_count,
                len(enriched),
            )

        themes_path.parent.mkdir(parents=True, exist_ok=True)
        themes_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(
            "Persisted themes | path=%s | segments=%d",
            themes_path,
            len(enriched),
        )

        artefact_key = f"{episode_key}_with_themes"
        if self.catalog:
            self.catalog.record_artefact(
                episode_id=self.podcast.episode_id,
                kind="with_themes",
                path=themes_path,
                artefact_key=artefact_key,
            )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="thematize",
            step_name="thematize",
            episode_id=self.podcast.episode_id,
            message="Persisted thematic summary",
            payload={
                "path": str(themes_path),
                "segments": len(enriched),
                "step": "completed",
            },
            artefact_paths={"with_themes": themes_path},
            checkpoint={
                "status": "completed",
                "step": "thematize",
                "themes_path": str(themes_path),
                "segments": len(enriched),
                "condensed_path": str(resolved_condensed),
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )

        self._last_themes_path = themes_path
        return themes_path

    def _chunk_transcript(self, text: str) -> List[str]:  # pragma: no cover - legacy method
        if not text:
            return []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        max_tokens = max(1, self.config.max_tokens_per_chunk)
        overlap_ratio = min(max(self.config.chunk_overlap_ratio, 0.0), 0.9)

        chunks: List[str] = []
        buffer: List[str] = []
        token_counts: List[int] = []
        token_total = 0

        for line in lines:
            tokens = self._estimate_tokens(line)
            if buffer and (token_total + tokens) > max_tokens:
                chunks.append("\n".join(buffer))
                overlap = int(round(len(buffer) * overlap_ratio))
                if overlap > 0:
                    buffer = buffer[-overlap:]
                    token_counts = token_counts[-overlap:]
                    token_total = sum(token_counts)
                else:
                    buffer = []
                    token_counts = []
                    token_total = 0

            buffer.append(line)
            token_counts.append(tokens)
            token_total += tokens

        if buffer:
            chunks.append("\n".join(buffer))
        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:  # pragma: no cover - legacy method
        words = text.split()
        return max(1, int(len(words) * 1.3) or len(text) // 4 or 1)

    def _invoke_llm_for_segment(self, segment: dict, segment_index: int) -> tuple[str, str]:
        segment_text = str(segment.get("text") or "").strip()
        speaker = str(segment.get("speaker_name") or segment.get("speaker_id") or "UNKNOWN")
        prompt = (
            "You are summarizing a single speaker segment from a podcast.\n"
            "Return ONLY valid JSON: {\"theme\": string (<=6 words), \"summary\": string (<=50 words)}.\n"
            f"Podcast: {self.podcast.show_title} – {self.podcast.episode_title}\n"
            f"Speaker: {speaker}\n"
            f"Segment:\n{segment_text}\n"
        )
        response = self.client.generate(prompt)
        data = json.loads(response.strip())
        if isinstance(data, list):
            data = data[0] if data else {}
        theme = str(data.get("theme") or f"Segment {segment_index + 1}").strip()
        summary = str(data.get("summary") or "").strip()
        # Enforce rough 50-word cap
        words = summary.split()
        if len(words) > 50:
            summary = " ".join(words[:50])
        return theme, summary

    def _parse_sections(self, raw_output: str, chunk_index: int) -> List[dict]:  # pragma: no cover - legacy method
        payload = raw_output.strip()
        data = json.loads(payload)
        if isinstance(data, dict):
            items = data.get("sections") or data.get("themes") or []
        else:
            items = data

        sections: List[dict] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            theme = str(item.get("theme") or f"Chunk {chunk_index + 1} Section {idx + 1}").strip()
            summary = str(
                item.get("summary")
                or item.get("description")
                or item.get("details")
                or ""
            ).strip()
            highlights_field = item.get("highlights")
            highlights: List[str] = []
            if isinstance(highlights_field, Sequence) and not isinstance(highlights_field, (str, bytes)):
                for highlight in highlights_field:
                    text = str(highlight).strip()
                    if text:
                        highlights.append(text)
            sections.append(
                {
                    "theme": theme or f"Chunk {chunk_index + 1}",
                    "summary": summary or "Summary unavailable.",
                    "highlights": highlights,
                    "chunk_index": chunk_index,
                }
            )
        return sections

    def _create_fallback_segment(self, segment: dict, segment_index: int, reason: Optional[str]) -> dict:
        """Create a fallback segment preserving the original transcript data."""
        summary = "Theme generation unavailable for this segment."
        if reason:
            summary = f"{summary} LLM error: {reason}"
        return {
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "speaker_id": str(segment.get("speaker_id") or "UNKNOWN"),
            "speaker_name": str(segment.get("speaker_name") or segment.get("speaker_id") or "UNKNOWN"),
            "text": str(segment.get("text") or "").strip(),
            "theme": self.config.fallback_theme,
            "summary": summary,
        }

    def _fallback_segments(self, reason: Optional[str]) -> List[dict]:
        """Create a single fallback segment when no transcript data is available."""
        summary = "Transcript unavailable."
        if reason:
            summary = f"{summary} LLM fallback reason: {reason}"
        return [{"start": 0.0, "end": 0.0, "speaker_id": "UNKNOWN", "speaker_name": "UNKNOWN", "text": "", "theme": self.config.fallback_theme, "summary": summary}]

    # Legacy helpers left in place for compatibility; not used in segment-based thematization


