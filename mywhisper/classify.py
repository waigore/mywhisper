from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Sequence

from .config import ensure_episode_subdir, resolve_data_root
from .models import PipelineEvent, PodcastEpisode
from .podcasts import PodcastCatalog

LOGGER = logging.getLogger("mywhisper.classify")

DEFAULT_CANDIDATE_LABELS = [
    "podcast advertisement or sponsorship",
    "promo or call-to-action",
    "episode intro or outro filler",
    "main editorial content",
]


@dataclass(slots=True)
class ClassifyConfig:
    """
    Configuration for zero-shot classification of podcast segments.
    """

    data_root: Path = field(default_factory=resolve_data_root)
    model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    candidate_labels: List[str] = field(default_factory=lambda: DEFAULT_CANDIDATE_LABELS.copy())
    classification_threshold: float = 0.75
    max_words_per_chunk: int = 300
    output_subdir: str = "transcripts"

    def themes_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        """Locate the thematized JSON input file."""
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_with_themes.json"

    def classified_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        """Locate the classified JSON output file."""
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_classified.json"


class EpisodeClassifier:
    """
    Classify podcast segments to identify non-editorial content using zero-shot classification.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: Optional[ClassifyConfig] = None,
        catalog: Optional[PodcastCatalog] = None,
        classifier: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config or ClassifyConfig()
        self.catalog = catalog
        self._classifier = classifier
        base_logger = logger or LOGGER
        self.logger = base_logger.getChild(podcast.episode_id)
        self._last_classified_path: Optional[Path] = None

    def classify(
        self,
        *,
        themes_path: Optional[Path] = None,
        yield_progress: bool = False,
    ) -> Path | Generator[PipelineEvent, None, Path]:
        pipeline = self._pipeline(themes_path=themes_path)
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

    def _pipeline(self, themes_path: Optional[Path]) -> Generator[PipelineEvent, None, Path]:
        episode_key = self.podcast.episode_key
        resolved_themes = themes_path or self.config.themes_path(self.podcast, episode_key)
        resolved_themes = resolved_themes.resolve()
        classified_path = self.config.classified_path(self.podcast, episode_key).resolve()

        if not resolved_themes.exists():
            raise FileNotFoundError(f"Thematized transcript not found at {resolved_themes}")

        start_time = time.perf_counter()
        self.logger.info(
            "Classification start | episode=%s | themes=%s | output=%s",
            self.podcast.episode_id,
            resolved_themes,
            classified_path,
        )

        yield PipelineEvent(
            stage="classify",
            step_name="classify",
            episode_id=self.podcast.episode_id,
            message="Loading thematized transcript",
            payload={
                "episode_key": episode_key,
                "themes_path": str(resolved_themes),
                "step": "load",
            },
            artefact_paths={"themes": resolved_themes},
            checkpoint={
                "status": "started",
                "step": "classify",
                "themes_path": str(resolved_themes),
                "episode_key": episode_key,
            },
        )

        # Load thematized records
        records = json.loads(resolved_themes.read_text(encoding="utf-8"))
        segments: List[dict] = [rec for rec in records if isinstance(rec, dict)]
        segment_count = len(segments)

        enriched: List[dict] = []

        for index, seg in enumerate(segments):
            segment_text = str(seg.get("text") or "").strip()
            if not segment_text:
                # Empty segment, add with empty classifications
                enriched.append(
                    {
                        **seg,
                        "classifications": [],
                    }
                )
                continue

            # Classify segment (with chunking if needed)
            classifications = self._classify_segment(segment_text)

            enriched.append(
                {
                    **seg,
                    "classifications": classifications,
                }
            )

            yield PipelineEvent(
                stage="classify",
                step_name="classify",
                episode_id=self.podcast.episode_id,
                message=f"Processed segment {index + 1}/{segment_count}",
                payload={
                    "segment_index": index,
                    "step": "segment_completed",
                },
                checkpoint={
                    "status": f"segment_{index + 1}",
                    "step": "classify",
                    "segment_index": index,
                },
            )

        classified_path.parent.mkdir(parents=True, exist_ok=True)
        classified_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(
            "Persisted classifications | path=%s | segments=%d",
            classified_path,
            len(enriched),
        )

        artefact_key = f"{episode_key}_classified"
        if self.catalog:
            self.catalog.record_artefact(
                episode_id=self.podcast.episode_id,
                kind="classified",
                path=classified_path,
                artefact_key=artefact_key,
            )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="classify",
            step_name="classify",
            episode_id=self.podcast.episode_id,
            message="Persisted classifications",
            payload={
                "path": str(classified_path),
                "segments": len(enriched),
                "step": "completed",
            },
            artefact_paths={"classified": classified_path},
            checkpoint={
                "status": "completed",
                "step": "classify",
                "classified_path": str(classified_path),
                "segments": len(enriched),
                "themes_path": str(resolved_themes),
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )

        self._last_classified_path = classified_path
        return classified_path

    def _classify_segment(self, text: str) -> List[dict]:
        """
        Classify a segment's text. If text is > 300 words, split into chunks and classify each.
        Returns a list of distinct classifications.
        """
        word_count = len(text.split())
        if word_count <= self.config.max_words_per_chunk:
            # Single chunk classification
            result = self._classify_text(text)
            return [result] if result else []

        # Split into chunks and classify each
        chunks = self._split_into_chunks(text, self.config.max_words_per_chunk)
        chunk_results: List[dict] = []
        label_to_result: dict[str, dict] = {}

        for chunk in chunks:
            result = self._classify_text(chunk)
            if result:
                label = result["label"]
                # Keep the result with the highest score for each distinct label
                if label not in label_to_result or result["score"] > label_to_result[label]["score"]:
                    label_to_result[label] = result

        # Convert to list, preserving all distinct labels
        return list(label_to_result.values())

    def _split_into_chunks(self, text: str, max_words: int) -> List[str]:
        """
        Split text into sentence-based chunks of at most max_words each.
        """
        if not text:
            return []

        # Split into sentences (simple approach: split on sentence-ending punctuation)
        sentences = re.split(r"([.!?]+\s+)", text)
        # Rejoin sentences with their punctuation
        sentence_list: List[str] = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_list.append(sentences[i] + sentences[i + 1])
            else:
                sentence_list.append(sentences[i])

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_word_count = 0

        for sentence in sentence_list:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > max_words and current_chunk:
                # Finalize current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def _classify_text(self, text: str) -> Optional[dict]:
        """
        Run zero-shot classification on text and return simplified result.
        Returns dict with keys: label, score, is_non_editorial.
        """
        if not text or not text.strip():
            return None

        classifier = self._get_classifier()
        try:
            result = classifier(text, self.config.candidate_labels, multi_label=True)
        except Exception as exc:
            self.logger.warning("Classification failed for text (first 100 chars): %s", text[:100], exc_info=exc)
            return None

        if not result or "labels" not in result or "scores" not in result:
            return None

        labels = result["labels"]
        scores = result["scores"]

        if not labels or not scores or len(labels) == 0:
            return None

        # Get top label and score
        top_label = labels[0]
        top_score = float(scores[0])

        is_non_editorial = (
            top_score > self.config.classification_threshold
            and top_label != "main editorial content"
        )

        return {
            "label": top_label,
            "score": round(top_score, 4),
            "is_non_editorial": is_non_editorial,
        }

    def _get_classifier(self) -> Any:
        """Lazy-load the classifier pipeline."""
        if self._classifier is not None:
            return self._classifier

        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.config.model_name,
            )
            return self._classifier
        except ImportError:
            raise RuntimeError(
                "transformers package is required for classification. Install with: pip install transformers"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load classification model: {exc}") from exc

