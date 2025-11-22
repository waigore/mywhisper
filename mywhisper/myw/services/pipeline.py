from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ...checkpoints import PipelineEventAdapter, CheckpointStore, PipelineCheckpoint
from ...models import DiarizedTurn, PipelineEvent, PodcastEpisode, TranscriptSegment
from ...podcasts import PodcastCatalog
from ..config import MywConfig
from ..messages import PipelineCompleted, PipelineEventPayload, PipelineProgress, PipelineStopped
from ..models import PipelineStatus
from .queue import QueueController, QueueItem
from .steps import (
    STEP_ORDER,
    extract_step_outputs,
    get_step,
    load_step_artefact,
    load_step_path,
    load_step_path_with_key,
    map_outputs_to_dependencies,
    validate_step_availability,
)

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


class PipelineInterrupted(Exception):
    """Raised when a stop is requested."""


@dataclass(slots=True)
class PipelineContext:
    episode: PodcastEpisode
    resume: bool
    completed_steps: Dict[str, str]
    step_plan: tuple[str, ...]
    pipeline_id: Optional[str] = None


@dataclass
class PipelineState:
    """
    State object to track step outputs during pipeline execution.
    
    This dataclass defines the standardized dependency keys that steps use
    to communicate with each other. Each field corresponds to a dependency
    key that downstream steps expect:
    
    - transcript_segments: Output from transcribe step
    - diarization_results: Output from diarize step (RTTM path)
    - readable_path: Output from prettify step
    - condensed_path: Output from prettify step
    - assignment_path: Output from assign step
    - themes_path: Output from thematize step
    - classified_path: Output from classify step
    - vocative_path: Output from vocative step
    
    The mapping from step outputs to these dependency keys is defined
    by each step's get_output_dependencies() method.
    """

    transcript_segments: Optional[Sequence[TranscriptSegment]] = None
    diarization_results: Optional[Any] = None
    assignment_path: Optional[Path] = None
    readable_path: Optional[Path] = None
    condensed_path: Optional[Path] = None
    themes_path: Optional[Path] = None
    classified_path: Optional[Path] = None
    vocative_path: Optional[Path] = None


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

    def _process_episode(self, context: PipelineContext) -> None:
        """
        Process an episode through the pipeline steps.
        
        This method orchestrates step execution generically, delegating all
        step-specific logic to the steps themselves.
        """
        adapter = PipelineEventAdapter(
            self.checkpoints,
            episode_id=context.episode.episode_id,
            pipeline_id=context.pipeline_id,
        )

        plan = context.step_plan
        # Track step outputs for dependency resolution
        step_outputs: Dict[str, Any] = {}

        # Iterate through steps in canonical order
        for step_name in STEP_ORDER:
            # Skip if step not in plan (unless needed by downstream steps)
            step = get_step(step_name, self.config)
            
            # Check if step should run
            should_run = step.should_run(plan, context.completed_steps, context.resume)
            
            # Load dependencies for this step
            dependencies = step.load_dependencies(self.checkpoints, context.episode.episode_id, context)
            
            # Merge step_outputs into dependencies using standardized contracts
            # Map outputs from all previously executed steps to their dependency keys
            for prev_step_name in STEP_ORDER:
                if prev_step_name in step_outputs and STEP_ORDER.index(prev_step_name) < STEP_ORDER.index(step_name):
                    prev_step = get_step(prev_step_name, self.config)
                    dependency_mapping = map_outputs_to_dependencies(prev_step_name, step_outputs, prev_step)
                    dependencies.update(dependency_mapping)
            
            # Check if step is needed (in plan or required by downstream steps)
            step_needed = step_name in plan
            if not step_needed:
                # Check if any downstream step needs this step's output
                downstream_steps = [s for s in STEP_ORDER if STEP_ORDER.index(s) > STEP_ORDER.index(step_name)]
                for downstream in downstream_steps:
                    if downstream in plan:
                        downstream_step = get_step(downstream, self.config)
                        if step_name in downstream_step.get_dependencies(plan):
                            step_needed = True
                            break
            
            # Guard clause: handle steps that aren't needed
            if not step_needed:
                # Step is needed by downstream but not in plan - just load artefact if available
                artefact = step.load_artefact(self.checkpoints, context.episode.episode_id)
                if artefact:
                    extracted = extract_step_outputs(
                        step_name,
                        artefact,
                        step,
                    )
                    step_outputs.update(extracted)
                # Continue to validation
                self._validate_and_check_stop(step_name, step_outputs, plan, context)
                continue
            
            # Step is needed - handle execution or checkpoint loading
            in_plan = step_name in plan
            
            # Guard clause: handle steps not in plan
            if not in_plan:
                # Step needed but not in plan - load from checkpoint
                artefact, extracted = self._load_step_from_checkpoint(context, step_name, step)
                if artefact:
                    step_outputs.update(extracted)
                # Continue to validation
                self._validate_and_check_stop(step_name, step_outputs, plan, context)
                continue
            
            # Step is in plan - prepare inputs and execute or load
            # Prepare inputs for execution
            try:
                inputs = step.prepare_inputs(context, dependencies, **step_outputs)
            except RuntimeError as e:
                raise
            
            # Guard clause: execute if should_run
            if should_run:
                # Execute the step
                extracted = self._execute_step(
                    context,
                    adapter,
                    step_name,
                    step,
                    inputs,
                    plan,
                    step_outputs,
                )
                step_outputs.update(extracted)
                # Continue to validation
                self._validate_and_check_stop(step_name, step_outputs, plan, context)
                continue
            
            # Not should_run - try to load from checkpoint
            artefact, extracted = self._load_step_from_checkpoint(context, step_name, step)
            
            # Guard clause: handle checkpoint missing
            if artefact is None:
                # Checkpoint missing, need to rerun
                LOGGER.warning(
                    "%s checkpoint missing for %s; rerunning step",
                    step_name,
                    context.episode.episode_id,
                )
                # Rerun the step (inputs already prepared above)
                extracted = self._execute_step(
                    context,
                    adapter,
                    step_name,
                    step,
                    inputs,
                    plan,
                    step_outputs,
                    extra_log_data={"reason": "checkpoint_missing"},
                )
                step_outputs.update(extracted)
            else:
                # Successfully loaded from checkpoint
                step_outputs.update(extracted)
            
            # Validate step availability
            self._validate_and_check_stop(step_name, step_outputs, plan, context)

    def _validate_and_check_stop(
        self,
        step_name: str,
        step_outputs: Dict[str, Any],
        plan: Sequence[str],
        context: PipelineContext,
    ) -> None:
        """Validate step and check for stop request."""
        validation_kwargs = self._get_validation_kwargs(step_name, step_outputs, context)
        validate_step_availability(step_name, plan, myw_config=self.config, completed_steps=context.completed_steps, **validation_kwargs)
        self._validate_downstream_artefacts(step_name, step_outputs, plan, context)
        self._check_stop()

    def _extract_and_log_outputs(
        self,
        context: PipelineContext,
        step_name: str,
        step: Any,
        result_or_artefact: Any,
        step_outputs: Dict[str, Any],
        source: str,
        elapsed: float,
        executor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Extract outputs from step result/artefact and log completion.
        
        Returns extracted outputs dict.
        """
        # Extract all outputs using standardized contract
        extracted = extract_step_outputs(
            step_name,
            result_or_artefact,
            step,
            executor=executor,
        )
        
        # Log completion
        summary = step.get_summary(result_or_artefact, step_outputs)
        self._log_step_end(
            context,
            step_name,
            {
                **summary,
                "source": source,
                "elapsed": round(elapsed, 2),
            },
        )
        
        return extracted

    def _execute_step(
        self,
        context: PipelineContext,
        adapter: PipelineEventAdapter,
        step_name: str,
        step: Any,
        inputs: Dict[str, Any],
        plan: Sequence[str],
        step_outputs: Dict[str, Any],
        mode: str = "execute",
        extra_log_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a step and return extracted outputs.
        
        Returns extracted outputs dict.
        """
        # Log step start
        log_data = {"mode": mode, "resume": context.resume}
        if extra_log_data:
            log_data.update(extra_log_data)
        self._log_step_start(context, step_name, log_data)
        
        start_time = perf_counter()
        
        # Create executor
        executor = step.create_executor(context.episode, catalog=self.catalog)
        
        # Execute the step with prepared inputs
        events = step.execute(executor, **inputs)
        
        # Consume events and get result
        result = self._consume_events(context, adapter, step_name, events, plan)
        elapsed = perf_counter() - start_time
        
        # Extract outputs and log completion
        extracted = self._extract_and_log_outputs(
            context,
            step_name,
            step,
            result,
            step_outputs,
            source="fresh",
            elapsed=elapsed,
            executor=executor,
        )
        
        return extracted

    def _load_step_from_checkpoint(
        self,
        context: PipelineContext,
        step_name: str,
        step: Any,
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Load step artefact from checkpoint and return (artefact, extracted_outputs).
        
        Returns tuple of (artefact, extracted_outputs). If artefact is None,
        extracted_outputs will be empty dict.
        """
        # Log checkpoint load start
        self._log_step_start(
            context,
            step_name,
            {
                "mode": "load_checkpoint",
                "checkpoint_status": context.completed_steps.get(step_name, "unknown"),
            },
        )
        
        load_start = perf_counter()
        artefact = step.load_artefact(self.checkpoints, context.episode.episode_id)
        load_elapsed = perf_counter() - load_start
        
        if artefact is None:
            return None, {}
        
        # Extract outputs from loaded artefact
        extracted = extract_step_outputs(
            step_name,
            artefact,
            step,
        )
        
        # Log completion
        summary = step.get_summary(artefact, {})
        self._log_step_end(
            context,
            step_name,
            {
                **summary,
                "source": "checkpoint",
                "elapsed": round(load_elapsed, 2),
            },
        )
        
        return artefact, extracted

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

    def _get_validation_kwargs(self, step_name: str, step_outputs: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        """
        Get validation keyword arguments for a step by delegating to the step itself.
        
        This method follows the abstraction principle - each step knows what validation
        kwargs it needs from its outputs.
        
        For steps that need dependencies for validation (like classify needing themes_path),
        we also load those dependencies from checkpoints.
        """
        # Handle special case: "condensed" is not a real step
        if step_name == "condensed":
            return {"condensed_path": step_outputs.get("condensed_path")}
        
        # Get step instance and delegate to its get_validation_kwargs method
        try:
            step = get_step(step_name, self.config)
            return step.get_validation_kwargs(
                step_outputs,
                checkpoints=self.checkpoints,
                episode_id=context.episode.episode_id,
            )
        except ValueError:
            # Step not found - return empty kwargs
            return {}

    def _validate_downstream_artefacts(
        self,
        step_name: str,
        step_outputs: Dict[str, Any],
        plan: Sequence[str],
        context: PipelineContext,
    ) -> None:
        """
        Validate artefacts produced by this step that are needed by downstream steps.
        
        This method checks if any outputs from the current step are needed by downstream
        steps in the plan, and validates those outputs accordingly. This handles cases
        like prettify producing condensed_path which is needed by thematize.
        """
        # Check downstream steps in the plan
        current_index = STEP_ORDER.index(step_name) if step_name in STEP_ORDER else -1
        downstream_steps = [
            s for s in STEP_ORDER
            if s in plan and STEP_ORDER.index(s) > current_index
        ]
        
        # Check if prettify produced condensed_path and thematize needs it
        if step_name == "prettify" and "thematize" in downstream_steps:
            condensed_path = step_outputs.get("condensed_path")
            validate_step_availability("condensed", plan, myw_config=self.config, condensed_path=condensed_path)

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

    # Validation is now handled generically via validate_step_availability

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
                    last_completed_step=STEP_ORDER[-1] if STEP_ORDER else None,
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

