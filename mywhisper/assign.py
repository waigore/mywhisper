"""
Speaker assignment pipeline for mywhisper using graph-based and contextual inference.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import networkx as nx
from networkx.algorithms.bipartite import maximum_matching

from .config import ensure_data_subdir, ensure_episode_subdir, resolve_data_root
from .logging_utils import LoggingBase
from .models import (
    InferenceResult,
    PipelineEvent,
    PodcastEpisode,
    SpeakerInference,
    TranscriptSegment,
)

LOGGER = logging.getLogger("mywhisper.assign")


@dataclass(slots=True)
class AssignmentConfig:
    """
    Configuration for speaker assignment pipeline.
    """

    output_dir: Path = field(default_factory=lambda: ensure_data_subdir("transcripts"))
    data_root: Path = field(default_factory=resolve_data_root)

    def inferred_names_path(
        self,
        podcast: PodcastEpisode,
        episode_key: Optional[str] = None,
    ) -> Path:
        key = episode_key or podcast.episode_key
        dir_path = ensure_episode_subdir(key, self.data_root, "transcripts")
        return dir_path / f"{key}_inferred_names.json"


class GraphBasedInference(LoggingBase):
    """
    Graph-based inference using bipartite matching to assign vocatives to speakers.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize GraphBasedInference with optional logger."""
        super().__init__()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = LOGGER.getChild("GraphBasedInference")

    def infer(
        self,
        segments: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, InferenceResult], Dict[str, List[str]]]:
        """
        Infer speaker names using bipartite graph matching.
        
        When a speaker uses a vocative, we create edges from OTHER speakers (next/previous)
        to that vocative, since the speaker using the vocative is addressing someone else.
        
        Args:
            segments: List of segment dictionaries with speaker_id, text, and addressed_person_candidates
            
        Returns:
            Tuple of (assignments dict mapping speaker_id to InferenceResult, sentences dict mapping speaker_id to list of sentences)
        """
        # Track which speakers use which vocatives (to exclude invalid assignments)
        # A speaker cannot be assigned a name they use as a vocative
        speaker_uses_vocative: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Collect edges: other_speaker -> vocative with weights
        # When SPEAKER_X uses "Name" as vocative, create edges from OTHER speakers to "Name"
        edges: Dict[Tuple[str, str], int] = defaultdict(int)
        vocative_sentences: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        for i, segment in enumerate(segments):
            speaker_id = segment.get("speaker_id")
            if not speaker_id:
                continue
            
            candidates = segment.get("addressed_person_candidates", [])
            for candidate in candidates:
                if candidate.get("classification") != "VOCATIVE":
                    continue
                
                vocative_name = candidate.get("name", "").strip()
                if not vocative_name:
                    continue
                
                sentence = candidate.get("sentence", "").strip()
                
                # Track that this speaker uses this vocative (invalid assignment)
                speaker_uses_vocative[(speaker_id, vocative_name)] += 1
                
                # Create edges from OTHER speakers to this vocative
                # Primary: next speaker (most likely to be the person addressed)
                if i + 1 < len(segments):
                    next_speaker = segments[i + 1].get("speaker_id")
                    if next_speaker and next_speaker != speaker_id:
                        edge_key = (next_speaker, vocative_name)
                        edges[edge_key] += 1
                        if sentence:
                            vocative_sentences[edge_key].append(sentence)
                
                # Secondary: previous speaker (less likely, but possible)
                if i > 0:
                    prev_speaker = segments[i - 1].get("speaker_id")
                    if prev_speaker and prev_speaker != speaker_id:
                        edge_key = (prev_speaker, vocative_name)
                        edges[edge_key] += 0.5  # Lower weight for previous speaker
                        if sentence:
                            vocative_sentences[edge_key].append(sentence)
        
        if not edges:
            self.logger.debug("No vocative edges found for graph inference")
            return {}, {}
        
        # Create bipartite graph
        G = nx.Graph()
        speakers = set(spk for spk, _ in edges.keys())
        vocatives = set(voc for _, voc in edges.keys())
        G.add_nodes_from(speakers, bipartite=0)
        G.add_nodes_from(vocatives, bipartite=1)
        
        # Add edges, but exclude any where the speaker uses the vocative themselves
        excluded_edges = 0
        for (spk, voc), weight in edges.items():
            # Skip if this speaker uses this vocative (invalid assignment)
            if (spk, voc) in speaker_uses_vocative:
                excluded_edges += 1
                self.logger.debug(
                    "Excluding edge (%s, %s): speaker uses this vocative",
                    spk,
                    voc,
                )
                continue
            G.add_edge(spk, voc, weight=weight)
        
        if excluded_edges > 0:
            self.logger.info(
                "Excluded %d invalid edges where speakers use vocatives themselves",
                excluded_edges,
            )
        
        if G.number_of_edges() == 0:
            self.logger.debug("No valid edges after filtering; cannot perform matching")
            return {}, {}
        
        # Use weighted greedy matching instead of unweighted maximum matching
        # This prioritizes higher-weight edges (next speaker > previous speaker)
        # For each vocative, find the speaker with the highest total weight
        vocative_to_speakers: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (spk, voc), weight in edges.items():
            # Skip if this speaker uses this vocative (invalid assignment)
            if (spk, voc) in speaker_uses_vocative:
                continue
            vocative_to_speakers[voc].append((spk, weight))
        
        # Sort vocatives by their maximum weight (prioritize strong matches)
        vocative_assignments: List[Tuple[str, str, float]] = []
        for voc, speaker_weights in vocative_to_speakers.items():
            # Find speaker with highest total weight for this vocative
            speaker_totals: Dict[str, float] = defaultdict(float)
            for spk, weight in speaker_weights:
                speaker_totals[spk] += weight
            
            if speaker_totals:
                best_speaker = max(speaker_totals, key=speaker_totals.get)
                best_weight = speaker_totals[best_speaker]
                vocative_assignments.append((voc, best_speaker, best_weight))
        
        # Sort by weight (descending) to process strongest matches first
        vocative_assignments.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy assignment: assign each vocative to best available speaker
        # (handle conflicts by keeping first assignment, which is highest weight)
        matching: Dict[str, str] = {}  # speaker -> vocative
        used_speakers: set[str] = set()
        used_vocatives: set[str] = set()
        
        for voc, spk, weight in vocative_assignments:
            # Skip if speaker or vocative already assigned
            if spk in used_speakers or voc in used_vocatives:
                continue
            
            # Double-check: ensure this speaker doesn't use this vocative
            if (spk, voc) in speaker_uses_vocative:
                self.logger.warning(
                    "Filtering invalid assignment: %s -> %s (speaker uses this vocative)",
                    spk,
                    voc,
                )
                continue
            
            matching[spk] = voc
            used_speakers.add(spk)
            used_vocatives.add(voc)
        
        # Build assignments with confidence and sentences
        assignments: Dict[str, InferenceResult] = {}
        speaker_sentences: Dict[str, List[str]] = defaultdict(list)
        
        for spk, voc in matching.items():
            # Calculate confidence: edge weight / total weights from that speaker
            total_weight = sum(G[spk][n].get("weight", 0) for n in G.neighbors(spk))
            edge_weight = G[spk][voc].get("weight", 0)
            conf = edge_weight / total_weight if total_weight > 0 else 0.0
            
            assignments[spk] = InferenceResult(name=voc, confidence=conf)
            
            # Collect sentences for this speaker-vocative pair
            edge_key = (spk, voc)
            if edge_key in vocative_sentences:
                speaker_sentences[spk].extend(vocative_sentences[edge_key])
        
        self.logger.info(
            "Graph-based inference: %d speakers matched to vocatives",
            len(assignments),
        )
        
        return assignments, dict(speaker_sentences)


class ContextualTurnInference(LoggingBase):
    """
    Contextual turn-taking inference using sequential patterns.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize ContextualTurnInference with optional logger."""
        super().__init__()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = LOGGER.getChild("ContextualTurnInference")

    def infer(
        self,
        segments: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, InferenceResult], Dict[str, List[str]]]:
        """
        Infer speaker names using contextual turn-taking patterns.
        
        Args:
            segments: List of segment dictionaries with speaker_id, text, and addressed_person_candidates
            
        Returns:
            Tuple of (assignments dict mapping speaker_id to InferenceResult, sentences dict mapping speaker_id to list of sentences)
        """
        # Score matrix: vocative -> speaker -> score
        scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        vocative_sentences: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        for i, segment in enumerate(segments):
            speaker_id = segment.get("speaker_id")
            if not speaker_id:
                continue
            
            candidates = segment.get("addressed_person_candidates", [])
            for candidate in candidates:
                if candidate.get("classification") != "VOCATIVE":
                    continue
                
                vocative_name = candidate.get("name", "").strip()
                if not vocative_name:
                    continue
                
                sentence = candidate.get("sentence", "").strip()
                
                # Boost score for next speaker (primary boost: 1.0)
                if i + 1 < len(segments):
                    next_speaker = segments[i + 1].get("speaker_id")
                    if next_speaker:
                        scores[vocative_name][next_speaker] += 1.0
                        vocative_sentences[(vocative_name, next_speaker)].append(sentence)
                
                # Lesser boost for previous speaker (secondary boost: 0.5)
                if i > 0:
                    prev_speaker = segments[i - 1].get("speaker_id")
                    if prev_speaker:
                        scores[vocative_name][prev_speaker] += 0.5
                        vocative_sentences[(vocative_name, prev_speaker)].append(sentence)
        
        if not scores:
            self.logger.debug("No vocative scores found for contextual inference")
            return {}, {}
        
        # For each vocative, normalize scores using softmax
        assignments: Dict[str, InferenceResult] = {}
        speaker_sentences: Dict[str, List[str]] = defaultdict(list)
        
        for voc, spk_scores in scores.items():
            if not spk_scores:
                continue
            
            # Apply softmax normalization
            exp_scores = {spk: math.exp(score) for spk, score in spk_scores.items()}
            softmax_total = sum(exp_scores.values())
            
            if softmax_total > 0:
                # Find best speaker with highest softmax probability
                best_spk = max(exp_scores, key=exp_scores.get)
                softmax_conf = exp_scores[best_spk] / softmax_total
                
                assignments[best_spk] = InferenceResult(name=voc, confidence=softmax_conf)
                
                # Collect sentences for this speaker-vocative pair
                edge_key = (voc, best_spk)
                if edge_key in vocative_sentences:
                    speaker_sentences[best_spk].extend(vocative_sentences[edge_key])
        
        self.logger.info(
            "Contextual inference: %d speakers matched to vocatives",
            len(assignments),
        )
        
        return assignments, dict(speaker_sentences)


class TranscriptAssigner(LoggingBase):
    """
    End-to-end pipeline for assigning speaker names using graph-based and contextual inference.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: AssignmentConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config
        self.logger = logger or LOGGER.getChild(podcast.episode_id)
        self._last_inferred_names_path: Optional[Path] = None
        self._last_episode_key: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        podcast: PodcastEpisode,
        config: AssignmentConfig,
    ) -> "TranscriptAssigner":
        """
        Create TranscriptAssigner instance from config.
        
        Args:
            podcast: The podcast episode to process
            config: Assignment configuration
            
        Returns:
            TranscriptAssigner instance
        """
        return cls(podcast=podcast, config=config)

    def infer_names(
        self,
        vocative_path: Optional[Path] = None,
        yield_progress: bool = False,
    ) -> Dict[str, Any] | Generator[PipelineEvent, None, Dict[str, Any]]:
        """
        Infer speaker names from vocative data using graph-based and contextual inference.
        
        Args:
            vocative_path: Path to vocative JSON file (if None, uses default path)
            yield_progress: If True, yields PipelineEvent objects during execution
            
        Returns:
            Dictionary with inferred_names_path and speakers list, or generator if yield_progress=True
        """
        pipeline = self._infer_pipeline(vocative_path=vocative_path)
        if yield_progress:
            return pipeline
        
        try:
            while True:
                next(pipeline)
        except StopIteration as stop:
            return stop.value

    def _infer_pipeline(
        self,
        vocative_path: Optional[Path],
    ) -> Generator[PipelineEvent, None, Dict[str, Any]]:
        episode_key = self.podcast.episode_key
        inferred_names_path = self.config.inferred_names_path(self.podcast, episode_key)
        resolved_vocative = vocative_path or self._get_default_vocative_path(episode_key)
        
        start_time = time.perf_counter()
        self.logger.info(
            "Speaker inference started | episode=%s | vocative_path=%s | output=%s",
            self.podcast.episode_id,
            resolved_vocative,
            inferred_names_path,
        )
        
        yield PipelineEvent(
            stage="start",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message=f"Inferring speaker names for {self.podcast.episode_title}",
            payload={
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "vocative_path": str(resolved_vocative) if resolved_vocative else None,
                "step": "started",
            },
            artefact_paths={"vocative": resolved_vocative} if resolved_vocative else {},
            checkpoint={
                "status": "started",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
            elapsed=0.0,
        )
        
        # Load vocative data
        if not resolved_vocative or not resolved_vocative.exists():
            self.logger.warning(
                "Vocative file not found at %s; cannot infer speaker names",
                resolved_vocative,
            )
            yield PipelineEvent(
                stage="error",
                step_name="assign",
                episode_id=self.podcast.episode_id,
                message="Vocative file not found",
                payload={
                    "error": "vocative_file_not_found",
                    "path": str(resolved_vocative) if resolved_vocative else None,
                },
                elapsed=time.perf_counter() - start_time,
            )
            return {
                "inferred_names_path": None,
                "speakers": [],
            }
        
        segments = self._load_vocative_segments(resolved_vocative)
        self.logger.info("Loaded %d segments from vocative file", len(segments))
        
        # Run graph-based inference
        graph_inference = GraphBasedInference(logger=self.logger)
        graph_assignments, graph_sentences = graph_inference.infer(segments)
        
        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="graph_inference",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message="Completed graph-based inference",
            payload={
                "assignments": len(graph_assignments),
                "step": "graph_inference",
            },
            checkpoint={
                "status": "graph_inference",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )
        
        # Run contextual inference
        context_inference = ContextualTurnInference(logger=self.logger)
        context_assignments, context_sentences = context_inference.infer(segments)
        
        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="context_inference",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message="Completed contextual inference",
            payload={
                "assignments": len(context_assignments),
                "step": "context_inference",
            },
            checkpoint={
                "status": "context_inference",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )
        
        # Combine results
        all_speakers = set()
        for seg in segments:
            speaker_id = seg.get("speaker_id")
            if speaker_id:
                all_speakers.add(speaker_id)
        
        speakers_list: List[Dict[str, Any]] = []
        for speaker_id in sorted(all_speakers):
            graph_result = graph_assignments.get(speaker_id)
            context_result = context_assignments.get(speaker_id)
            
            # Collect all sentences for this speaker
            sentences = set()
            if speaker_id in graph_sentences:
                sentences.update(graph_sentences[speaker_id])
            if speaker_id in context_sentences:
                sentences.update(context_sentences[speaker_id])
            
            speaker_data: Dict[str, Any] = {
                "speaker_id": speaker_id,
                "graph_inference": (
                    {"name": graph_result.name, "confidence": graph_result.confidence}
                    if graph_result
                    else None
                ),
                "context_inference": (
                    {"name": context_result.name, "confidence": context_result.confidence}
                    if context_result
                    else None
                ),
                "sentences": sorted(list(sentences)),
            }
            speakers_list.append(speaker_data)
        
        # Persist results
        output_data = {"speakers": speakers_list}
        inferred_names_path.parent.mkdir(parents=True, exist_ok=True)
        with inferred_names_path.open("w", encoding="utf-8") as handle:
            json.dump(output_data, handle, ensure_ascii=False, indent=2)
        
        self._last_inferred_names_path = inferred_names_path
        self._last_episode_key = episode_key
        
        self.logger.info(
            "Persisted inferred names | path=%s | speakers=%d",
            inferred_names_path,
            len(speakers_list),
        )
        
        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="persisted",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message="Persisted inferred speaker names",
            payload={
                "path": str(inferred_names_path),
                "speakers": len(speakers_list),
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "step": "completed",
            },
            artefact_paths={"inferred_names": inferred_names_path},
            checkpoint={
                "status": "completed",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "inferred_names_path": str(inferred_names_path),
                "speakers": len(speakers_list),
            },
            elapsed=elapsed,
        )
        
        return {
            "inferred_names_path": inferred_names_path,
            "speakers": speakers_list,
        }

    def get_inferred_names_path(self) -> Optional[Path]:
        """
        Get the inferred names path from the last execution.
        
        Returns:
            Path to the inferred names file, or None if not yet executed.
        """
        return self._last_inferred_names_path

    def get_outputs(self) -> Dict[str, Any]:
        """
        Get all outputs from assign execution.
        
        Returns:
            Dictionary with inferred_names_path key.
        """
        return {
            "inferred_names_path": self._last_inferred_names_path,
        }

    def _get_default_vocative_path(self, episode_key: str) -> Optional[Path]:
        """Get default vocative path based on episode key."""
        from .vocative import VocativeConfig
        
        config = VocativeConfig(data_root=self.config.data_root)
        vocative_path = config.vocative_path(self.podcast, episode_key)
        return vocative_path if vocative_path.exists() else None

    def _load_vocative_segments(self, vocative_path: Path) -> List[Dict[str, Any]]:
        """Load segments from vocative JSON file."""
        with vocative_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        
        if not isinstance(data, list):
            self.logger.warning("Vocative file does not contain a list; returning empty segments")
            return []
        
        return data
