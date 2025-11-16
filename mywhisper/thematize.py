from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence

from .assign import LLMClient, OllamaClient
from .config import ensure_episode_subdir, resolve_data_root
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

    def readable_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_readable.txt"

    def themes_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_themes.json"


class EpisodeThematizer:
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
        readable_path: Optional[Path] = None,
        yield_progress: bool = False,
    ) -> Path | Generator[PipelineEvent, None, Path]:
        pipeline = self._pipeline(readable_path=readable_path)
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

    def _pipeline(self, readable_path: Optional[Path]) -> Generator[PipelineEvent, None, Path]:
        episode_key = self.podcast.episode_key
        resolved_readable = readable_path or self.config.readable_path(self.podcast, episode_key)
        resolved_readable = resolved_readable.resolve()
        themes_path = self.config.themes_path(self.podcast, episode_key).resolve()

        if not resolved_readable.exists():
            raise FileNotFoundError(f"Readable transcript not found at {resolved_readable}")

        start_time = time.perf_counter()
        self.logger.info(
            "Thematization start | episode=%s | readable=%s | output=%s",
            self.podcast.episode_id,
            resolved_readable,
            themes_path,
        )

        yield PipelineEvent(
            stage="thematize",
            step_name="thematize",
            episode_id=self.podcast.episode_id,
            message="Loading readable transcript",
            payload={
                "episode_key": episode_key,
                "readable_path": str(resolved_readable),
                "step": "load",
            },
            artefact_paths={"readable": resolved_readable},
            checkpoint={
                "status": "started",
                "step": "thematize",
                "readable_path": str(resolved_readable),
                "episode_key": episode_key,
            },
        )

        readable_text = resolved_readable.read_text(encoding="utf-8").strip()
        chunks = self._chunk_transcript(readable_text)
        chunk_count = max(1, len(chunks))

        sections: List[dict] = []
        failure_reason: Optional[str] = None

        for index, chunk_text in enumerate(chunks):
            try:
                chunk_sections = self._invoke_llm(chunk_text, index)
                sections.extend(chunk_sections)
            except Exception as exc:  # pragma: no cover - fallback path exercised separately
                failure_reason = str(exc)
                self.logger.warning("LLM generation failed on chunk %d: %s", index, exc)
                sections = []
                break

            yield PipelineEvent(
                stage="thematize",
                step_name="thematize",
                episode_id=self.podcast.episode_id,
                message=f"Processed chunk {index + 1}/{chunk_count}",
                payload={
                    "chunk_index": index,
                    "sections": len(chunk_sections),
                    "step": "chunk_completed",
                },
                checkpoint={
                    "status": f"chunk_{index + 1}",
                    "step": "thematize",
                    "chunk_index": index,
                },
            )

        if not sections:
            sections = self._fallback_sections(readable_text, failure_reason)

        merged = self._merge_adjacent_sections(sections)
        themes_path.parent.mkdir(parents=True, exist_ok=True)
        themes_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(
            "Persisted themes | path=%s | sections=%d",
            themes_path,
            len(merged),
        )

        artefact_key = f"{episode_key}_themes"
        if self.catalog:
            self.catalog.record_artefact(
                episode_id=self.podcast.episode_id,
                kind="themes",
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
                "sections": len(merged),
                "step": "completed",
            },
            artefact_paths={"themes": themes_path},
            checkpoint={
                "status": "completed",
                "step": "thematize",
                "themes_path": str(themes_path),
                "sections": len(merged),
                "readable_path": str(resolved_readable),
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )

        self._last_readable_path = resolved_readable
        self._last_themes_path = themes_path
        return themes_path

    def _chunk_transcript(self, text: str) -> List[str]:
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
    def _estimate_tokens(text: str) -> int:
        words = text.split()
        return max(1, int(len(words) * 1.3) or len(text) // 4 or 1)

    def _invoke_llm(self, chunk_text: str, chunk_index: int) -> List[dict]:
        prompt = self.config.prompt_template.format(
            show_title=self.podcast.show_title,
            episode_title=self.podcast.episode_title,
            chunk=chunk_text,
        )
        response = self.client.generate(prompt)
        return self._parse_sections(response, chunk_index)

    def _parse_sections(self, raw_output: str, chunk_index: int) -> List[dict]:
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

    def _fallback_sections(self, transcript_text: str, reason: Optional[str]) -> List[dict]:
        summary = textwrap.shorten(
            transcript_text or "Transcript unavailable.",
            width=420,
            placeholder="…",
        )
        if reason:
            summary = f"{summary}\n\nLLM fallback reason: {reason}"
        return [
            {
                "theme": self.config.fallback_theme,
                "summary": summary,
                "highlights": [],
                "chunk_index": None,
            }
        ]

    def _merge_adjacent_sections(self, sections: Iterable[dict]) -> List[dict]:
        merged: List[dict] = []
        for section in sections:
            theme = (section.get("theme") or "").strip()
            summary = (section.get("summary") or "").strip()
            highlights = section.get("highlights") or []

            if merged and theme and theme.lower() == merged[-1]["theme"].lower():
                merged[-1]["summary"] = (merged[-1]["summary"] + " " + summary).strip()
                merged[-1]["highlights"].extend(highlights)
            else:
                merged.append(
                    {
                        "theme": theme or self.config.fallback_theme,
                        "summary": summary or "Summary unavailable.",
                        "highlights": list(highlights),
                    }
                )
        return merged


