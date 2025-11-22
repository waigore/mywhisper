from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ...assign import AssignmentConfig, TranscriptAssigner
from ...checkpoints import CheckpointStore, PipelineCheckpoint
from ...classify import ClassifyConfig, EpisodeClassifier
from ...diarize import DiarizationConfig, DiarizationPipeline
from ...models import DiarizedTurn, PipelineEvent, PodcastEpisode, TranscriptSegment
from ...podcasts import PodcastCatalog
from ...prettify import PrettifyConfig, TranscriptPrettifier
from ...thematize import ThematizeConfig, EpisodeThematizer
from ...transcribe import PodcastTranscriber, TranscriptionConfig
from ...vocative import EpisodeVocativeDetector, VocativeConfig
from ..config import MywConfig

LOGGER = logging.getLogger("mywhisper.myw.steps")


class Step(ABC):
    """
    Base abstract class for pipeline steps.
    
    Each step encapsulates its own configuration, executor creation, and execution logic.
    This allows pipeline.py to work with steps generically without importing step-specific classes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this step."""
        pass

    @abstractmethod
    def create_config(self, myw_config: MywConfig) -> Any:
        """Create step-specific configuration from MywConfig."""
        pass

    @abstractmethod
    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create and return the executor instance for this step."""
        pass

    @abstractmethod
    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """
        Execute the step using the provided executor.
        
        Args:
            executor: The executor instance created by create_executor
            **kwargs: Step-specific execution arguments
        
        Returns:
            Iterable of PipelineEvent objects
        """
        pass

    @abstractmethod
    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """
        Determine if this step should execute.
        
        Args:
            plan: The step plan being executed
            completed_steps: Dictionary of completed step names to their status
            resume: Whether this is a resume operation
        
        Returns:
            True if the step should execute, False otherwise
        """
        pass

    @abstractmethod
    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """
        Return the step names this step depends on.
        
        Args:
            plan: The step plan being executed
        
        Returns:
            Tuple of step names that must complete before this step
        """
        pass

    @abstractmethod
    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """
        Load required inputs from checkpoints for this step.
        
        Args:
            checkpoints: Checkpoint store to query
            episode_id: Episode identifier
            context: Pipeline context (may contain episode, etc.)
        
        Returns:
            Dictionary of dependency names to their loaded values
        """
        pass

    @abstractmethod
    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare execution inputs from dependencies and context.
        
        Args:
            context: Pipeline context
            dependencies: Loaded dependencies from load_dependencies
            **kwargs: Additional context
        
        Returns:
            Dictionary of input names to their prepared values for execute()
        """
        pass

    @abstractmethod
    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """
        Load this step's own artefact if it has been completed.
        
        Args:
            checkpoints: Checkpoint store to query
            episode_id: Episode identifier
        
        Returns:
            The loaded artefact, or None if not found
        """
        pass

    @abstractmethod
    def get_output_dependencies(self) -> Dict[str, str]:
        """
        Return the mapping from this step's output keys to dependency keys.
        
        This defines the contract for what this step produces and how it maps
        to dependency keys that downstream steps consume.
        
        Returns:
            Dictionary mapping step output keys to dependency keys.
            For example: {"transcribe": "transcript_segments"} means the step
            stores its result under "transcribe" key, which maps to "transcript_segments"
            dependency key for downstream steps.
        """
        pass

    @abstractmethod
    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """
        Extract all outputs from step execution result and executor.
        
        This method should return a dictionary with all output keys defined
        in get_output_dependencies(). The result parameter may be the primary
        return value from execute(), and executor may contain additional state
        needed to extract all outputs.
        
        Args:
            executor: The executor instance used for step execution
            result: The primary result from step execution (may be None for checkpoint loads)
        
        Returns:
            Dictionary mapping output keys to their values. Must include all
            keys from get_output_dependencies().
        """
        pass

    @abstractmethod
    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary information for this step's output.
        
        Args:
            result: The primary result from step execution
            step_outputs: Dictionary of all step outputs (keyed by output keys)
        
        Returns:
            Dictionary with summary information about the step's output
        """
        pass

    @abstractmethod
    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get validation keyword arguments for this step from step outputs.
        
        This method extracts the necessary validation arguments from the step outputs
        dictionary. These kwargs are then passed to validate_step_availability().
        
        Args:
            step_outputs: Dictionary of all step outputs (keyed by output keys)
            checkpoints: Optional checkpoint store for loading dependencies
            episode_id: Optional episode ID for loading dependencies
        
        Returns:
            Dictionary of validation keyword arguments
        """
        pass

    @abstractmethod
    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """
        Validate that this step can run given the plan and dependencies.
        
        Args:
            plan: The step plan being executed
            completed_steps: Dictionary of completed step names to their status
            dependencies: Dictionary of dependency keys to their loaded values
        
        Raises:
            RuntimeError: If validation fails
        """
        pass


@dataclass
class StepMetadata:
    """Metadata for a pipeline step including path keys, validators, and loaders."""

    name: str
    path_keys: tuple[str, ...]
    validator: Callable[..., None]
    artefact_loader: Optional[Callable[[Path], Any]] = None
    artefact_key: Optional[str] = None


@dataclass
class StepDefinition:
    """Definition of a pipeline step with its execution logic and dependencies."""

    name: str
    dependencies: tuple[str, ...]
    config_factory: Callable[[MywConfig], Any]
    executor: Callable[..., Any]
    loader: Callable[[CheckpointStore, str, str], Optional[Any]]
    validator: Callable[[Sequence[str], Dict[str, str], Any], None]
    path_keys: tuple[str, ...]
    preprocessor: Optional[Callable[..., Any]] = None
    postprocessor: Optional[Callable[..., Any]] = None


def load_artefact_path(
    checkpoints: CheckpointStore,
    episode_id: str,
    step: str,
    path_keys: Sequence[str],
) -> Optional[Path]:
    """
    Generic loader for artefact paths from checkpoints.
    
    Args:
        checkpoints: Checkpoint store to query
        episode_id: Episode identifier
        step: Step name
        path_keys: Keys to try in order (details first, then payload)
    
    Returns:
        Resolved Path if found and exists, None otherwise
    """
    checkpoint = checkpoints.get_step(episode_id, step)
    if checkpoint and checkpoint.status == "completed":
        for key in path_keys:
            path = checkpoint.details.get(key) or checkpoint.payload.get(key)
            if path:
                resolved = Path(path)
                if resolved.exists():
                    return resolved
    return None


def get_step_metadata(step_name: str) -> StepMetadata:
    """Retrieve metadata for a given step."""
    if step_name not in STEP_METADATA_REGISTRY:
        raise ValueError(f"Unknown step: {step_name}. Available steps: {list(STEP_METADATA_REGISTRY.keys())}")
    return STEP_METADATA_REGISTRY[step_name]


def load_step_path(
    step_name: str,
    checkpoints: CheckpointStore,
    episode_id: str,
) -> Optional[Path]:
    """
    Generic path loader using step metadata.
    
    Args:
        step_name: Name of the step
        checkpoints: Checkpoint store to query
        episode_id: Episode identifier
    
    Returns:
        Resolved Path if found and exists, None otherwise
    """
    metadata = get_step_metadata(step_name)
    return load_artefact_path(checkpoints, episode_id, step_name, metadata.path_keys)


def load_step_path_with_key(
    step_name: str,
    path_key: str,
    checkpoints: CheckpointStore,
    episode_id: str,
) -> Optional[Path]:
    """
    Load a specific path key for a step (e.g., condensed_path from prettify).
    
    Args:
        step_name: Name of the step
        path_key: Specific key to load
        checkpoints: Checkpoint store to query
        episode_id: Episode identifier
    
    Returns:
        Resolved Path if found and exists, None otherwise
    """
    return load_artefact_path(checkpoints, episode_id, step_name, (path_key,))


def load_step_artefact(
    step_name: str,
    checkpoints: CheckpointStore,
    episode_id: str,
) -> Optional[Any]:
    """
    Generic artefact loader using step metadata.
    Loads the path and optionally applies a custom loader function.
    
    Args:
        step_name: Name of the step
        checkpoints: Checkpoint store to query
        episode_id: Episode identifier
    
    Returns:
        Loaded artefact (type depends on step's artefact_loader) or Path if no loader is defined
    """
    path = load_step_path(step_name, checkpoints, episode_id)
    if path is None:
        return None
    
    metadata = get_step_metadata(step_name)
    if metadata.artefact_loader:
        try:
            return metadata.artefact_loader(path)
        except Exception as exc:
            LOGGER.warning("Failed to load artefact for step %s from %s: %s", step_name, path, exc)
            return None
    
    return path


def _try_get_executor_outputs(executor: Any, output_keys: tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """
    Try to extract outputs from executor using get_outputs() method.
    
    Args:
        executor: The executor instance
        output_keys: Tuple of output keys to extract
    
    Returns:
        Dictionary of outputs if successful, None otherwise
    """
    if executor is None:
        return None
    
    try:
        executor_dict = executor.get_outputs()
        if executor_dict:
            return {key: executor_dict.get(key) for key in output_keys}
    except AttributeError:
        pass
    
    return None


def _try_get_legacy_attribute(executor: Any, attribute_name: str) -> Optional[Any]:
    """
    Try to get a legacy attribute from executor.
    
    Args:
        executor: The executor instance
        attribute_name: Name of the attribute to get
    
    Returns:
        Attribute value if found, None otherwise
    """
    if executor is None:
        return None
    
    try:
        return getattr(executor, attribute_name, None)
    except AttributeError:
        return None


def extract_step_outputs(
    step_name: str,
    result: Any,
    step: Step,
    executor: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Extract all outputs from a step's result according to its output contract.
    
    This function uses the step's get_outputs() method to extract all outputs
    in a standardized way, then applies the output contract mapping.
    
    Args:
        step_name: Name of the step
        result: The primary result from step execution (may be None for checkpoint loads)
        step: The step instance (used to get outputs and output contract)
        executor: Optional executor instance (may be None for checkpoint loads)
    
    Returns:
        Dictionary mapping output keys to their values, including the step name
        and all outputs defined in the step's output contract.
    """
    # Get all outputs from the step
    outputs = step.get_outputs(executor, result)
    
    # Ensure step name is included in outputs
    if step_name not in outputs:
        outputs[step_name] = result
    
    # Apply output contract mapping - map outputs to dependency keys
    output_contract = step.get_output_dependencies()
    for output_key, dependency_key in output_contract.items():
        if output_key in outputs and outputs[output_key] is not None:
            outputs[dependency_key] = outputs[output_key]
    
    return outputs


def map_outputs_to_dependencies(
    step_name: str,
    step_outputs: Dict[str, Any],
    step: Step,
) -> Dict[str, Any]:
    """
    Map step outputs to dependency keys according to the step's output contract.
    
    This function uses the step's get_output_dependencies() contract to automatically
    map step outputs to the dependency keys that downstream steps expect.
    
    Args:
        step_name: Name of the step that just ran
        step_outputs: Dictionary of all step outputs (keyed by output keys)
        step: The step instance (used to get output contract)
    
    Returns:
        Dictionary mapping dependency keys to their values (to be merged into dependencies)
    """
    dependency_mapping: Dict[str, Any] = {}
    output_contract = step.get_output_dependencies()
    
    # Map each output key to its corresponding dependency key
    for output_key, dependency_key in output_contract.items():
        if output_key in step_outputs and step_outputs[output_key] is not None:
            dependency_mapping[dependency_key] = step_outputs[output_key]
    
    return dependency_mapping


def validate_step_availability(
    step_name: str,
    plan: Sequence[str],
    myw_config: MywConfig,
    completed_steps: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """
    Validate step availability by delegating to the step's validate() method.
    
    This function gets the step instance and delegates validation to the step itself,
    following the abstraction principle where each step knows how to validate itself.
    
    Args:
        step_name: Name of the step to validate
        plan: Current step plan
        myw_config: MywConfig instance needed to create step instance
        completed_steps: Dictionary of completed steps (optional)
        **kwargs: Additional arguments mapped to dependencies:
            - segments -> transcript_segments
            - diarization_results -> diarization_results
            - readable_path -> readable_path
            - condensed_path -> condensed_path
            - themes_path -> themes_path
            - vocative_path -> vocative_path
    
    Raises:
        RuntimeError: If validation fails
        ValueError: If step_name is unknown
    """
    # Handle special case: "condensed" is not a real step, but an artefact from prettify
    if step_name == "condensed":
        condensed_path = kwargs.get("condensed_path")
        validate_condensed_availability(plan, condensed_path)
        return
    
    # Get step instance and delegate validation
    try:
        step = get_step(step_name, myw_config)
    except ValueError:
        # Step not found - no validation needed
        return
    
    # Map kwargs to dependency keys expected by steps
    dependencies: Dict[str, Any] = {}
    if "segments" in kwargs:
        dependencies["transcript_segments"] = kwargs["segments"]
    if "diarization_results" in kwargs:
        dependencies["diarization_results"] = kwargs["diarization_results"]
    if "readable_path" in kwargs:
        dependencies["readable_path"] = kwargs["readable_path"]
    if "condensed_path" in kwargs:
        dependencies["condensed_path"] = kwargs["condensed_path"]
    if "themes_path" in kwargs:
        dependencies["themes_path"] = kwargs["themes_path"]
    if "vocative_path" in kwargs:
        dependencies["vocative_path"] = kwargs["vocative_path"]
    
    # Delegate to step's validate method
    step.validate(plan, completed_steps or {}, dependencies)


def read_transcript(path: Path) -> Optional[Sequence[TranscriptSegment]]:
    """Read transcript segments from a JSON file."""
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


def ensure_diarized_turns(diarization_results: Any) -> List[DiarizedTurn]:
    """Convert diarization results to a list of DiarizedTurn objects."""
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
        return read_rttm_turns(Path(diarization_results))

    return []


def read_rttm_turns(path: Path) -> List[DiarizedTurn]:
    """Read diarization turns from an RTTM file."""
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


def apply_diarization_labels(
    segments: Sequence[TranscriptSegment],
    turns: Sequence[DiarizedTurn],
) -> List[TranscriptSegment]:
    """Apply diarization speaker IDs to transcript segments based on time overlap."""
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


def ensure_placeholder_assignment(
    config: MywConfig,
    episode: PodcastEpisode,
    segments: Sequence[TranscriptSegment],
) -> Path:
    """
    Persist a placeholder 'assigned transcript' JSON that contains diarized segments
    with speaker_id placeholders and speaker_name mirroring the speaker_id.
    """
    cfg = PrettifyConfig(data_root=config.data_dir)
    assignment_path = cfg.assignment_path(episode, episode.episode_key).resolve()
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


# Validation functions
def validate_transcript_availability(
    plan: Sequence[str],
    segments: Optional[Sequence[TranscriptSegment]],
) -> None:
    """Validate that transcript segments are available when required."""
    requires_transcript = "transcribe" not in plan and any(step in ("diarize",) for step in plan)
    if requires_transcript and segments is None:
        raise RuntimeError(
            "Transcript segments are required for the selected steps. Run transcription first or include it in the plan."
        )


def validate_diarization_availability(
    plan: Sequence[str],
    completed_steps: Dict[str, str],
    diarization_results: Any,
) -> None:
    """Validate that diarization results are available when required."""
    if not any(step in plan for step in ("prettify", "assign")):
        return
    # Allow if diarization will run in this plan or is already completed
    if "diarize" in plan or "diarize" in completed_steps:
        return
    if diarization_results is None:
        raise RuntimeError(
            "Diarization results are required for prettify/assign. Include diarization before these steps."
        )


def validate_assignment_availability(
    plan: Sequence[str],
    readable_path: Optional[Path],
) -> None:
    """Validate that readable transcript is available for assignment."""
    if "assign" not in plan:
        return
    if readable_path is None or not readable_path.exists():
        raise RuntimeError(
            "Readable transcript artefact is required. Run prettify before assignment or include it in the plan."
        )


def validate_condensed_availability(
    plan: Sequence[str],
    condensed_path: Optional[Path],
) -> None:
    """Validate that condensed transcript is available for thematization."""
    if "thematize" not in plan:
        return
    if condensed_path is None or not condensed_path.exists():
        raise RuntimeError(
            "Condensed transcript artefact is required for thematization. Run prettify first or include it in the plan."
        )


def validate_themes_availability(
    plan: Sequence[str],
    themes_path: Optional[Path],
) -> None:
    """Validate that themes artefact is available for classification."""
    if "classify" not in plan:
        return
    if themes_path is None or not themes_path.exists():
        raise RuntimeError(
            "Thematized transcript artefact is required for classification. Run thematize first or include it in the plan."
        )


def validate_classified_availability(
    plan: Sequence[str],
    completed_steps: Dict[str, str],
) -> None:
    """Validate that classified artefact is available for vocative detection."""
    if "vocative" not in plan:
        return
    # Allow if classify will run in this plan or is already completed
    if "classify" in plan or "classify" in completed_steps:
        return
    # Soft check - actual validation happens in step execution


def validate_vocative_availability(
    plan: Sequence[str],
    vocative_path: Optional[Path],
) -> None:
    """Validate vocative artefact (optional for assign step)."""
    # Vocative is optional for assign step, so we don't require it
    pass


# Step metadata registry
STEP_METADATA_REGISTRY: Dict[str, StepMetadata] = {}


def _register_step_metadata(
    name: str,
    path_keys: tuple[str, ...],
    validator: Callable[..., None],
    artefact_loader: Optional[Callable[[Path], Any]] = None,
    artefact_key: Optional[str] = None,
) -> None:
    """Register step metadata in the registry."""
    STEP_METADATA_REGISTRY[name] = StepMetadata(
        name=name,
        path_keys=path_keys,
        validator=validator,
        artefact_loader=artefact_loader,
        artefact_key=artefact_key,
    )


# Register all steps with their metadata
_register_step_metadata(
    name="transcribe",
    path_keys=("transcript_path", "path"),
    validator=validate_transcript_availability,
    artefact_loader=read_transcript,
)

_register_step_metadata(
    name="diarize",
    path_keys=("rttm_path",),
    validator=validate_diarization_availability,
)

_register_step_metadata(
    name="prettify",
    path_keys=("readable_path", "path"),  # Primary path
    validator=lambda *args, **kwargs: None,  # No direct validation for prettify
)

# Prettify also produces condensed_path, but it's loaded separately
_register_step_metadata(
    name="prettify_condensed",
    path_keys=("condensed_path",),
    validator=lambda *args, **kwargs: None,
)

_register_step_metadata(
    name="assign",
    path_keys=("assignment_path", "path"),
    validator=validate_assignment_availability,
)

_register_step_metadata(
    name="thematize",
    path_keys=("themes_path", "path"),
    validator=validate_themes_availability,
)

_register_step_metadata(
    name="classify",
    path_keys=("classified_path", "path"),
    validator=validate_classified_availability,
)

_register_step_metadata(
    name="vocative",
    path_keys=("vocative_path", "path"),
    validator=validate_vocative_availability,
)

# Step registry (legacy - for backwards compatibility)
# Note: Executors are implemented as methods on PipelineRunner.
# The registry provides metadata and wiring information.
STEP_REGISTRY: Dict[str, StepDefinition] = {}

# Step order for canonical pipeline execution
STEP_ORDER = ("transcribe", "diarize", "prettify", "thematize", "classify", "vocative", "assign")


# Concrete step implementations
class TranscribeStep(Step):
    """Transcription step that uses Whisper to transcribe audio."""

    @property
    def name(self) -> str:
        return "transcribe"

    def create_config(self, myw_config: MywConfig) -> Any:
        """Create TranscriptionConfig from MywConfig."""
        if not myw_config.whisper_model:
            raise RuntimeError("MYW_WHISPER_MODEL must be configured for transcription.")
        return TranscriptionConfig(
            model_path=Path(myw_config.whisper_model),
            data_root=myw_config.data_dir,
            device=myw_config.device,
        )

    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create PodcastTranscriber instance."""
        config = self.create_config(self._myw_config)
        return PodcastTranscriber.from_config(episode, config)

    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """Execute transcription."""
        return executor.transcribe(yield_progress=True)

    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """Determine if transcription should run."""
        if self.name not in plan:
            return False
        # When resume=False, re-run even if step is completed
        return self.name not in completed_steps or not resume

    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """Transcribe has no dependencies."""
        return ()

    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Transcribe has no dependencies to load."""
        return {}

    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe requires no input preparation."""
        return {}

    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """Load transcript segments from checkpoint."""
        return load_step_artefact(self.name, checkpoints, episode_id)

    def get_output_dependencies(self) -> Dict[str, str]:
        """Transcribe outputs transcript segments."""
        return {"transcribe": "transcript_segments"}

    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """Extract all outputs from transcribe execution."""
        return {"transcribe": result}

    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary information for transcription output."""
        segments = result if isinstance(result, Sequence) else None
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

    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validation kwargs for transcribe step."""
        return {"segments": step_outputs.get("transcribe")}

    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate that transcription can run."""
        segments = dependencies.get("transcript_segments")
        validate_transcript_availability(plan, segments)

    def __init__(self, myw_config: MywConfig) -> None:
        self._myw_config = myw_config


class DiarizeStep(Step):
    """Diarization step that identifies speakers in audio."""

    @property
    def name(self) -> str:
        return "diarize"

    def create_config(self, myw_config: MywConfig) -> Any:
        """Create DiarizationConfig from MywConfig."""
        return DiarizationConfig(
            data_root=myw_config.data_dir,
            hf_token=myw_config.hf_token,
            device=myw_config.device,
        )

    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create DiarizationPipeline instance."""
        config = self.create_config(self._myw_config)
        return DiarizationPipeline.from_config(episode, config)

    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """
        Execute diarization.
        
        Note: Diarization has special event handling. This method yields
        a start event, then runs the pipeline, then yields a completed event.
        The executor's run() method returns turns directly.
        """
        episode = kwargs.get("episode")
        if episode is None:
            raise ValueError("episode is required for DiarizeStep.execute")
        
        config = self.create_config(self._myw_config)
        paths = config.artefact_paths(episode, episode.episode_key)

        start_event = PipelineEvent(
            stage="start",
            step_name="diarize",
            episode_id=episode.episode_id,
            message=f"Starting diarization for {episode.episode_title}",
            payload={
                "episode_key": episode.episode_key,
                "step": "started",
            },
            artefact_paths={"rttm": paths["rttm_path"]},
            checkpoint={
                "status": "started",
                "step": "diarize",
                "episode_key": episode.episode_key,
            },
        )
        yield start_event

        # Run the diarization pipeline
        turns = executor.run()

        completed_event = PipelineEvent(
            stage="persisted",
            step_name="diarize",
            episode_id=episode.episode_id,
            message="Persisted diarization RTTM",
            payload={
                "path": str(paths["rttm_path"]),
                "turns": len(turns) if turns else 0,
                "step": "completed",
            },
            artefact_paths={"rttm": paths["rttm_path"]},
            checkpoint={
                "status": "completed",
                "step": "diarize",
                "rttm_path": str(paths["rttm_path"]),
                "turns": len(turns) if turns else 0,
            },
        )
        yield completed_event

    def get_start_event(
        self, episode: PodcastEpisode, executor: Any
    ) -> tuple[PipelineEvent, Dict[str, Path]]:
        """Get the start event and artefact paths for diarization."""
        config = self.create_config(self._myw_config)
        paths = config.artefact_paths(episode, episode.episode_key)

        start_event = PipelineEvent(
            stage="start",
            step_name="diarize",
            episode_id=episode.episode_id,
            message=f"Starting diarization for {episode.episode_title}",
            payload={
                "episode_key": episode.episode_key,
                "step": "started",
            },
            artefact_paths={"rttm": paths["rttm_path"]},
            checkpoint={
                "status": "started",
                "step": "diarize",
                "episode_key": episode.episode_key,
            },
        )
        return start_event, paths

    def get_completed_event(
        self, episode: PodcastEpisode, paths: Dict[str, Path], turns: List[Any]
    ) -> PipelineEvent:
        """Get the completed event for diarization."""
        return PipelineEvent(
            stage="persisted",
            step_name="diarize",
            episode_id=episode.episode_id,
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

    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """Determine if diarization should run."""
        if self.name not in plan:
            return False
        # When resume=False, re-run even if step is completed
        return self.name not in completed_steps or not resume

    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """Diarize optionally depends on transcribe (for transcript segments)."""
        # Transcript segments are optional for diarization
        return ("transcribe",)

    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Load transcript segments if available."""
        segments = load_step_artefact("transcribe", checkpoints, episode_id)
        return {"transcript_segments": segments}

    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare inputs for diarize: include episode from context."""
        episode = getattr(context, "episode", None)
        if episode is None:
            raise RuntimeError("Episode is required in context for DiarizeStep.prepare_inputs")
        return {"episode": episode}

    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """Load diarization results (RTTM path) from checkpoint."""
        return load_step_path(self.name, checkpoints, episode_id)

    def get_output_dependencies(self) -> Dict[str, str]:
        """Diarize outputs diarization results (RTTM path)."""
        return {"diarize": "diarization_results"}

    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """Extract all outputs from diarize execution."""
        return {"diarize": result}

    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary information for diarization output."""
        diarization_results = result
        artefact_path = str(result) if isinstance(result, (str, Path)) else None
        if diarization_results is None:
            return {"turns": 0, "artefact_path": artefact_path}
        if isinstance(diarization_results, (str, Path)):
            return {"turns": None, "artefact_path": str(diarization_results)}
        turns_count = len(diarization_results) if isinstance(diarization_results, Sized) else None
        return {
            "turns": turns_count,
            "artefact_path": artefact_path,
        }

    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validation kwargs for diarize step."""
        return {"diarization_results": step_outputs.get("diarize")}

    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate that diarization can run."""
        diarization_results = dependencies.get("diarization_results")
        validate_diarization_availability(plan, completed_steps, diarization_results)

    def ensure_diarized_turns(self, diarization_results: Any) -> List[DiarizedTurn]:
        """Convert diarization results to a list of DiarizedTurn objects."""
        # Reference the function defined in this module
        return ensure_diarized_turns(diarization_results)

    def apply_diarization_labels(
        self, segments: Sequence[TranscriptSegment], turns: Sequence[DiarizedTurn]
    ) -> List[TranscriptSegment]:
        """Apply diarization speaker IDs to transcript segments based on time overlap."""
        # Reference the function defined in this module
        return apply_diarization_labels(segments, turns)

    def __init__(self, myw_config: MywConfig) -> None:
        self._myw_config = myw_config


class PrettifyStep(Step):
    """Prettify step that formats transcript into readable format."""

    @property
    def name(self) -> str:
        return "prettify"

    def create_config(self, myw_config: MywConfig) -> Any:
        """Create PrettifyConfig from MywConfig."""
        return PrettifyConfig(data_root=myw_config.data_dir)

    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create TranscriptPrettifier instance."""
        if catalog is None:
            raise ValueError("catalog is required for PrettifyStep")
        config = self.create_config(self._myw_config)
        return TranscriptPrettifier(podcast=episode, config=config, catalog=catalog)

    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """Execute prettification."""
        assignment_path = kwargs.get("assignment_path")
        if assignment_path is None:
            raise ValueError("assignment_path is required for PrettifyStep.execute")
        return executor.prettify(assignment_path=assignment_path, yield_progress=True)

    def ensure_placeholder_assignment(
        self, episode: PodcastEpisode, segments: Sequence[TranscriptSegment]
    ) -> Path:
        """Create a placeholder assignment file for prettify step."""
        # Reference the function defined in this module
        return ensure_placeholder_assignment(self._myw_config, episode, segments)

    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """Determine if prettify should run."""
        if self.name not in plan:
            return False
        # When resume=False, re-run even if step is completed
        return self.name not in completed_steps or not resume

    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """Prettify depends on transcribe and diarize."""
        return ("transcribe", "diarize")

    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Load transcript segments and diarization results."""
        segments = load_step_artefact("transcribe", checkpoints, episode_id)
        diarization_results = load_step_path("diarize", checkpoints, episode_id)
        return {
            "transcript_segments": segments,
            "diarization_results": diarization_results,
        }

    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for prettify: apply diarization labels and create placeholder assignment.
        
        Requires episode from context.
        """
        episode = getattr(context, "episode", None)
        if episode is None:
            raise RuntimeError("Episode is required in context for PrettifyStep.prepare_inputs")
        
        transcript_segments = dependencies.get("transcript_segments")
        if transcript_segments is None:
            raise RuntimeError("Transcript segments are required to run prettify.")
        
        diarization_results = dependencies.get("diarization_results")
        
        # Apply diarization labels if available
        if diarization_results:
            diarized_turns = ensure_diarized_turns(diarization_results)
            if diarized_turns:
                transcript_segments = apply_diarization_labels(transcript_segments, diarized_turns)
            else:
                LOGGER.warning(
                    "No diarization turns available for %s; speaker IDs will remain unset.",
                    episode.episode_id,
                )
        
        # Create placeholder assignment
        assignment_path = self.ensure_placeholder_assignment(episode, transcript_segments or [])
        
        return {"assignment_path": assignment_path}

    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """Load readable path from checkpoint. Returns a dict with readable_path and condensed_path."""
        readable_path = load_step_path(self.name, checkpoints, episode_id)
        condensed_path = load_step_path_with_key(self.name, "condensed_path", checkpoints, episode_id)
        if readable_path:
            return {"readable_path": readable_path, "condensed_path": condensed_path}
        return None

    def get_output_dependencies(self) -> Dict[str, str]:
        """Prettify outputs both readable_path and condensed_path."""
        return {
            "readable_path": "readable_path",
            "condensed_path": "condensed_path",
        }

    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """Extract all outputs from prettify execution."""
        # If result is already a dict (from checkpoint or fresh execution), use it
        if isinstance(result, dict):
            return {
                "readable_path": result.get("readable_path"),
                "condensed_path": result.get("condensed_path"),
            }
        
        # Try to get from executor state if available
        executor_outputs = _try_get_executor_outputs(executor, ("readable_path", "condensed_path"))
        if executor_outputs:
            return executor_outputs
        
        # Fallback: result might be just readable_path (legacy support)
        condensed = _try_get_legacy_attribute(executor, "_last_condensed_path")
        return {
            "readable_path": result,
            "condensed_path": condensed,
        }

    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary information for prettify output."""
        readable_path = step_outputs.get("readable_path")
        return {"path": str(readable_path) if readable_path else None}

    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validation kwargs for prettify step."""
        # Prettify doesn't need validation kwargs for itself
        return {}

    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate that prettify can run. Prettify has no specific validation requirements."""
        # Prettify step itself doesn't require validation, but it produces condensed_path
        # which is validated separately for thematize step
        pass

    def __init__(self, myw_config: MywConfig) -> None:
        self._myw_config = myw_config


class AssignStep(Step):
    """Assignment step that assigns speaker names to segments."""

    @property
    def name(self) -> str:
        return "assign"

    def create_config(self, myw_config: MywConfig) -> Any:
        """Create AssignmentConfig from MywConfig."""
        return AssignmentConfig(
            data_root=myw_config.data_dir,
            ollama_model=myw_config.ollama_model,
            spacy_model=myw_config.spacy_model,
        )

    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create TranscriptAssigner instance."""
        config = self.create_config(self._myw_config)
        return TranscriptAssigner.from_config(episode, config)

    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """Execute assignment."""
        segments = kwargs.get("segments")
        readable_path = kwargs.get("readable_path")
        metadata = kwargs.get("metadata", {})

        if readable_path:
            return executor.assign_from_readable(readable_path, metadata=metadata, yield_progress=True)
        elif segments:
            return executor.assign_names(segments, metadata=metadata, yield_progress=True)
        else:
            raise ValueError("Either segments or readable_path must be provided for AssignStep.execute")

    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """Determine if assign should run."""
        if self.name not in plan:
            return False
        # When resume=False, re-run even if step is completed
        return self.name not in completed_steps or not resume

    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """Assign depends on prettify (for readable transcript)."""
        return ("prettify",)

    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Load readable transcript path."""
        readable_path = load_step_path("prettify", checkpoints, episode_id)
        return {"readable_path": readable_path}

    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare inputs for assign: use readable_path and metadata."""
        readable_path = dependencies.get("readable_path")
        if not readable_path or not readable_path.exists():
            raise RuntimeError("Readable transcript is required for assignment. Run prettify first.")
        
        episode = getattr(context, "episode", None)
        metadata = episode.metadata if episode else {}
        
        return {"readable_path": readable_path, "metadata": metadata}

    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """Load assignment path and transcript segments from checkpoint."""
        assignment_path = load_step_path(self.name, checkpoints, episode_id)
        if assignment_path:
            # Try to load transcript segments from the assignment file
            segments = load_step_artefact("transcribe", checkpoints, episode_id)
            return {
                "transcript_segments": segments,
                "assignment_path": assignment_path,
            }
        return None

    def get_output_dependencies(self) -> Dict[str, str]:
        """Assign outputs both transcript_segments and assignment_path."""
        return {
            "transcript_segments": "transcript_segments",
            "assignment_path": "assignment_path",
        }

    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """Extract all outputs from assign execution."""
        # If result is already a dict (from checkpoint or fresh execution), use it
        if isinstance(result, dict):
            return {
                "transcript_segments": result.get("transcript_segments"),
                "assignment_path": result.get("assignment_path"),
            }
        
        # Try to get from executor state if available
        executor_outputs = _try_get_executor_outputs(executor, ("transcript_segments", "assignment_path"))
        if executor_outputs:
            return executor_outputs
        
        # Try get_assignment_path method if available
        if executor is not None:
            try:
                assignment_path = executor.get_assignment_path()
                return {
                    "transcript_segments": result,
                    "assignment_path": assignment_path,
                }
            except AttributeError:
                pass
        
        # Legacy fallback: result might be just transcript_segments
        assignment_path = _try_get_legacy_attribute(executor, "_last_assignment_path")
        return {
            "transcript_segments": result,
            "assignment_path": assignment_path,
        }

    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary information for assignment output."""
        segments = step_outputs.get("transcript_segments")
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

    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validation kwargs for assign step."""
        return {"readable_path": step_outputs.get("readable_path")}

    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate that assign can run."""
        readable_path = dependencies.get("readable_path")
        validate_assignment_availability(plan, readable_path)

    def __init__(self, myw_config: MywConfig) -> None:
        self._myw_config = myw_config


class ThematizeStep(Step):
    """Thematization step that identifies themes in transcript."""

    @property
    def name(self) -> str:
        return "thematize"

    def create_config(self, myw_config: MywConfig) -> Any:
        """Create ThematizeConfig from MywConfig."""
        return ThematizeConfig(data_root=myw_config.data_dir, llm_model=myw_config.ollama_model)

    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create EpisodeThematizer instance."""
        if catalog is None:
            raise ValueError("catalog is required for ThematizeStep")
        config = self.create_config(self._myw_config)
        return EpisodeThematizer(podcast=episode, config=config, catalog=catalog)

    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """Execute thematization."""
        condensed_path = kwargs.get("condensed_path")
        if condensed_path is None:
            raise ValueError("condensed_path is required for ThematizeStep.execute")
        return executor.thematize(condensed_path=condensed_path, yield_progress=True)

    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """Determine if thematize should run."""
        if self.name not in plan:
            return False
        # When resume=False, re-run even if step is completed
        return self.name not in completed_steps or not resume

    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """Thematize depends on prettify (for condensed transcript)."""
        return ("prettify",)

    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Load condensed transcript path."""
        condensed_path = load_step_path_with_key("prettify", "condensed_path", checkpoints, episode_id)
        return {"condensed_path": condensed_path}

    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare inputs for thematize: use condensed_path."""
        condensed_path = dependencies.get("condensed_path")
        if condensed_path is None:
            raise RuntimeError("Condensed transcript required for thematization.")
        return {"condensed_path": condensed_path}

    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """Load themes path from checkpoint."""
        return load_step_path(self.name, checkpoints, episode_id)

    def get_output_dependencies(self) -> Dict[str, str]:
        """Thematize outputs themes_path."""
        return {"thematize": "themes_path"}

    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """Extract all outputs from thematize execution."""
        return {"thematize": result}

    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary information for thematize output."""
        return {"path": str(result) if result else None}

    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validation kwargs for thematize step."""
        # Thematize needs condensed_path for validation, not themes_path (which is its output)
        return {"condensed_path": step_outputs.get("condensed_path")}

    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate that thematize can run."""
        condensed_path = dependencies.get("condensed_path")
        validate_condensed_availability(plan, condensed_path)

    def __init__(self, myw_config: MywConfig) -> None:
        self._myw_config = myw_config


class ClassifyStep(Step):
    """Classification step that classifies episode content."""

    @property
    def name(self) -> str:
        return "classify"

    def create_config(self, myw_config: MywConfig) -> Any:
        """Create ClassifyConfig from MywConfig."""
        return ClassifyConfig(data_root=myw_config.data_dir)

    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create EpisodeClassifier instance."""
        if catalog is None:
            raise ValueError("catalog is required for ClassifyStep")
        config = self.create_config(self._myw_config)
        return EpisodeClassifier(podcast=episode, config=config, catalog=catalog)

    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """Execute classification."""
        themes_path = kwargs.get("themes_path")
        if themes_path is None:
            raise ValueError("themes_path is required for ClassifyStep.execute")
        return executor.classify(themes_path=themes_path, yield_progress=True)

    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """Determine if classify should run."""
        if self.name not in plan:
            return False
        # When resume=False, re-run even if step is completed
        return self.name not in completed_steps or not resume

    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """Classify depends on thematize."""
        return ("thematize",)

    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Load themes path."""
        themes_path = load_step_path("thematize", checkpoints, episode_id)
        return {"themes_path": themes_path}

    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare inputs for classify: use themes_path."""
        themes_path = dependencies.get("themes_path")
        if themes_path is None:
            raise RuntimeError("Thematized transcript required for classification.")
        return {"themes_path": themes_path}

    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """Load classified path from checkpoint."""
        return load_step_path(self.name, checkpoints, episode_id)

    def get_output_dependencies(self) -> Dict[str, str]:
        """Classify outputs classified_path."""
        return {"classify": "classified_path"}

    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """Extract all outputs from classify execution."""
        return {"classify": result}

    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary information for classify output."""
        return {"path": str(result) if result else None}

    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validation kwargs for classify step."""
        # Classify doesn't need validation kwargs from its own outputs
        # It validates based on themes_path dependency, which needs to be loaded from checkpoints
        validation_kwargs = {}
        if checkpoints is not None and episode_id is not None:
            themes_path = load_step_path("thematize", checkpoints, episode_id)
            if themes_path:
                validation_kwargs["themes_path"] = themes_path
        return validation_kwargs

    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate that classify can run."""
        themes_path = dependencies.get("themes_path")
        validate_themes_availability(plan, themes_path)

    def __init__(self, myw_config: MywConfig) -> None:
        self._myw_config = myw_config


class VocativeStep(Step):
    """Vocative detection step that identifies vocative expressions."""

    @property
    def name(self) -> str:
        return "vocative"

    def create_config(self, myw_config: MywConfig) -> Any:
        """Create VocativeConfig from MywConfig."""
        return VocativeConfig(
            data_root=myw_config.data_dir,
            spacy_model=myw_config.spacy_model,
            llm_model=myw_config.ollama_model,
        )

    def create_executor(self, episode: PodcastEpisode, catalog: Optional[PodcastCatalog] = None) -> Any:
        """Create EpisodeVocativeDetector instance."""
        if catalog is None:
            raise ValueError("catalog is required for VocativeStep")
        config = self.create_config(self._myw_config)
        return EpisodeVocativeDetector(podcast=episode, config=config, catalog=catalog)

    def execute(self, executor: Any, **kwargs: Any) -> Iterable[PipelineEvent]:
        """Execute vocative detection."""
        classified_path = kwargs.get("classified_path")
        if classified_path is None:
            raise ValueError("classified_path is required for VocativeStep.execute")
        return executor.detect_vocatives(classified_path=classified_path, yield_progress=True)

    def should_run(self, plan: Sequence[str], completed_steps: Dict[str, str], resume: bool) -> bool:
        """Determine if vocative should run."""
        if self.name not in plan:
            return False
        # When resume=False, re-run even if step is completed
        return self.name not in completed_steps or not resume

    def get_dependencies(self, plan: Sequence[str]) -> tuple[str, ...]:
        """Vocative depends on classify."""
        return ("classify",)

    def load_dependencies(
        self,
        checkpoints: CheckpointStore,
        episode_id: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Load classified path."""
        classified_path = load_step_path("classify", checkpoints, episode_id)
        return {"classified_path": classified_path}

    def prepare_inputs(
        self,
        context: Any,
        dependencies: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare inputs for vocative: use classified_path."""
        classified_path = dependencies.get("classified_path")
        if classified_path is None:
            raise RuntimeError("Classified transcript required for vocative detection.")
        return {"classified_path": classified_path}

    def load_artefact(self, checkpoints: CheckpointStore, episode_id: str) -> Optional[Any]:
        """Load vocative path from checkpoint."""
        return load_step_path(self.name, checkpoints, episode_id)

    def get_output_dependencies(self) -> Dict[str, str]:
        """Vocative outputs vocative_path."""
        return {"vocative": "vocative_path"}

    def get_outputs(self, executor: Any, result: Any) -> Dict[str, Any]:
        """Extract all outputs from vocative execution."""
        return {"vocative": result}

    def get_summary(self, result: Any, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary information for vocative output."""
        return {"path": str(result) if result else None}

    def get_validation_kwargs(
        self,
        step_outputs: Dict[str, Any],
        checkpoints: Optional[Any] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get validation kwargs for vocative step."""
        return {"vocative_path": step_outputs.get("vocative")}

    def validate(
        self,
        plan: Sequence[str],
        completed_steps: Dict[str, str],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate that vocative can run."""
        # Vocative validation is soft - it just checks if classify is available
        validate_classified_availability(plan, completed_steps)

    def __init__(self, myw_config: MywConfig) -> None:
        self._myw_config = myw_config


def get_step(name: str, myw_config: MywConfig) -> Step:
    """Factory function to create a step instance by name."""
    step_map: Dict[str, type[Step]] = {
        "transcribe": TranscribeStep,
        "diarize": DiarizeStep,
        "prettify": PrettifyStep,
        "assign": AssignStep,
        "thematize": ThematizeStep,
        "classify": ClassifyStep,
        "vocative": VocativeStep,
    }
    step_class = step_map.get(name)
    if step_class is None:
        raise ValueError(f"Unknown step: {name}. Available steps: {list(step_map.keys())}")
    return step_class(myw_config)

