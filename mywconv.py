from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence

from mywhisper.checkpoints import CheckpointStore
from mywhisper.myw.config import ConfigError, MywConfig, load_config
from mywhisper.myw.logging import setup_logging
from mywhisper.myw.messages import PipelineCompleted, PipelineProgress, PipelineStopped
from mywhisper.myw.models import EpisodeViewState, PipelineStatus
from mywhisper.myw.services.catalog import CatalogService
from mywhisper.myw.services.pipeline import PipelineRunner, STEP_ORDER
from mywhisper.myw.services.queue import QueueController
from mywhisper.podcasts import PodcastCatalog
from uuid import uuid4


class PipelineMonitor:
    """
    Receive pipeline events from the runner and display them on stdout.
    """

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._done = threading.Event()
        self._success = False
        self._final_status: Optional[PipelineStatus] = None
        self._final_remarks: str = ""

    def __call__(
        self,
        message: PipelineProgress | PipelineStopped | PipelineCompleted,
    ) -> None:
        payload = message.payload
        status = payload.status

        if isinstance(message, PipelineProgress):
            self._print_progress(status, payload.remarks)
        elif isinstance(message, PipelineCompleted):
            self._success = True
            self._final_status = status
            self._final_remarks = payload.remarks
            self._print_progress(status, payload.remarks, final=True)
            self._done.set()
        elif isinstance(message, PipelineStopped):
            self._success = False
            self._final_status = status
            self._final_remarks = payload.remarks
            self._print_progress(status, payload.remarks, final=True)
            self._done.set()

    def wait(self) -> None:
        self._done.wait()

    @property
    def success(self) -> bool:
        return self._success

    @property
    def final_remarks(self) -> str:
        return self._final_remarks

    def _print_progress(
        self,
        status: PipelineStatus,
        remarks: str,
        final: bool = False,
    ) -> None:
        progress_pct = status.progress * 100 if status.progress else 0.0
        prefix = "OK" if final and self._success else "ER" if final else "--"
        step = status.step or "pending"
        print(f"{prefix} [{progress_pct:5.1f}%] {step}: {remarks}", flush=True)


def run_catalog_sync(service: CatalogService) -> list[EpisodeViewState]:
    print("Syncing catalog from Apple Podcasts cache...", flush=True)
    episodes = service.sync_from_cache()
    if not episodes:
        print("No episodes found in catalog. Add podcasts to the cache and retry.", flush=True)
    return episodes


def select_episode(episodes: Iterable[EpisodeViewState]) -> Optional[EpisodeViewState]:
    episode_list = list(episodes)
    if not episode_list:
        return None

    print("\nAvailable episodes:\n")
    for index, episode in enumerate(episode_list, start=1):
        show = episode.show_title or "Unknown Show"
        title = episode.episode_title or "Untitled Episode"
        print(f"{index:3d}. [{episode.episode_key}] {show} - {title}")
    print()

    while True:
        selection = input("Select an episode (number or key, blank to cancel): ").strip()
        if not selection:
            return None
        if selection.isdigit():
            choice = int(selection)
            if 1 <= choice <= len(episode_list):
                return episode_list[choice - 1]
        else:
            for episode in episode_list:
                if episode.episode_key == selection:
                    return episode
        print("Invalid selection. Try again.", flush=True)


def select_pipeline_scope(
    episode: EpisodeViewState,
    checkpoints: CheckpointStore,
) -> tuple[Optional[tuple[str, ...]], str]:
    completed = _completed_step_names(checkpoints, episode.episode_id)
    all_done = all(step in completed for step in STEP_ORDER)
    options: dict[str, str] = {"1": "Full pipeline"}
    if not all_done:
        options["2"] = "Resume pipeline"
    options["3"] = "Partial pipeline"

    print("Pipeline scopes:\n")
    for key, label in options.items():
        print(f"  {key}. {label}")
    print()

    default = "2" if "2" in options else "1"
    while True:
        selection = input(f"Select pipeline scope [{default}]: ").strip() or default
        label = options.get(selection)
        if not label:
            print("Invalid selection. Try again.", flush=True)
            continue

        if label == "Full pipeline":
            return None, label

        if label == "Partial pipeline":
            plan = _prompt_partial_plan(episode, checkpoints)
            if not plan:
                # user aborted partial selection; restart scope selection
                continue
            warning = _validate_scope_requirements(
                episode_id=episode.episode_id,
                plan=plan,
                checkpoints=checkpoints,
            )
            if warning:
                print(warning, flush=True)
                # loop back to scope selection
                continue
            return tuple(plan), label

        # Resume: start from first pending step through the end
        first_pending_index = 0
        for idx, step in enumerate(STEP_ORDER):
            if step not in completed:
                first_pending_index = idx
                break
        plan = STEP_ORDER[first_pending_index:]
        warning = _validate_scope_requirements(
            episode_id=episode.episode_id,
            plan=plan,
            checkpoints=checkpoints,
        )
        if warning:
            print(warning, flush=True)
            continue
        return tuple(plan), label


def _prompt_partial_plan(
    episode: EpisodeViewState,
    checkpoints: CheckpointStore,
) -> Optional[tuple[str, ...]]:
    status_row = checkpoints.get_pipeline_status(episode.episode_id)
    current_step = (status_row or {}).get("current_step") if status_row else None
    last_completed = (status_row or {}).get("last_completed_step") if status_row else None
    # Determine maximum allowed start index per constraint: at or before in-progress step,
    # otherwise at or before last completed step if available; else allow any.
    max_start_index = len(STEP_ORDER) - 1
    if isinstance(current_step, str) and current_step in STEP_ORDER:
        max_start_index = STEP_ORDER.index(current_step)
    elif isinstance(last_completed, str) and last_completed in STEP_ORDER:
        max_start_index = STEP_ORDER.index(last_completed)

    allowed_starts = STEP_ORDER[: max_start_index + 1]

    print("\nPartial pipeline selection:", flush=True)
    print("Select starting step (must be at or before current in-progress step).", flush=True)
    for idx, step in enumerate(allowed_starts, start=1):
        print(f"  {idx}. {step}")
    print()

    start_choice: Optional[int] = None
    while True:
        raw = input(f"Start at [1-{len(allowed_starts)}] or name (blank to cancel): ").strip()
        if not raw:
            return None
        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(allowed_starts):
                start_choice = num - 1
                break
        else:
            lowered = raw.lower()
            if lowered in allowed_starts:
                start_choice = allowed_starts.index(lowered)
                break
        print("Invalid selection. Try again.", flush=True)

    start_index = start_choice
    end_candidates = STEP_ORDER[start_index :]  # inclusive range selection
    print("\nSelect ending step (must be at or after the starting step).", flush=True)
    for idx, step in enumerate(end_candidates, start=1):
        print(f"  {idx}. {step}")
    print()

    while True:
        raw = input(f"End at [1-{len(end_candidates)}] or name (blank to cancel): ").strip()
        if not raw:
            return None
        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(end_candidates):
                end_index = start_index + (num - 1)
                break
        else:
            lowered = raw.lower()
            if lowered in end_candidates:
                end_index = STEP_ORDER.index(lowered)
                break
        print("Invalid selection. Try again.", flush=True)

    selected = STEP_ORDER[start_index : end_index + 1]
    return tuple(selected)


def _has_completed_step(checkpoints: CheckpointStore, episode_id: str, step: str) -> bool:
    checkpoint = checkpoints.get_step(episode_id, step)
    if not (checkpoint and checkpoint.status == "completed"):
        return False
    details = checkpoint.details or {}
    payload = checkpoint.payload or {}
    if step == "transcribe":
        path_str = details.get("transcript_path") or payload.get("path")
        return bool(path_str and Path(path_str).exists())
    if step == "diarize":
        path_str = details.get("rttm_path")
        return bool(path_str and Path(path_str).exists())
    if step == "assign":
        path_str = details.get("assignment_path") or payload.get("path")
        return bool(path_str and Path(path_str).exists())
    if step == "prettify":
        path_str = details.get("readable_path") or payload.get("path")
        return bool(path_str and Path(path_str).exists())
    if step == "thematize":
        path_str = details.get("themes_path") or payload.get("path")
        return bool(path_str and Path(path_str).exists())
    if step == "classify":
        path_str = details.get("classified_path") or payload.get("path")
        return bool(path_str and Path(path_str).exists())
    if step == "vocative":
        path_str = details.get("vocative_path") or payload.get("path")
        return bool(path_str and Path(path_str).exists())
    return True


def _validate_scope_requirements(
    *,
    episode_id: str,
    plan: Optional[tuple[str, ...]],
    checkpoints: CheckpointStore,
) -> Optional[str]:
    if not plan:
        return None

    def _ensure(condition: bool, message: str) -> Optional[str]:
        return None if condition else message

    requires_transcript = ("diarize" in plan) and "transcribe" not in plan
    if requires_transcript:
        warning = _ensure(
            _has_completed_step(checkpoints, episode_id, "transcribe"),
            "Selected scope requires a completed transcript. Run transcription first or include it in the plan.",
        )
        if warning:
            return warning

    requires_diarization = any(step in plan for step in ("assign", "prettify", "thematize")) and "diarize" not in plan
    if requires_diarization:
        warning = _ensure(
            _has_completed_step(checkpoints, episode_id, "diarize"),
            "Selected scope requires completed diarization. Run diarization first or include it in the plan.",
        )
        if warning:
            return warning

    requires_readable_for_assign = ("assign" in plan) and "prettify" not in plan
    if requires_readable_for_assign:
        warning = _ensure(
            _has_completed_step(checkpoints, episode_id, "prettify"),
            "Assignment requires a readable transcript from prettify. Run prettify first or include it in the plan.",
        )
        if warning:
            return warning

    requires_readable = "thematize" in plan and "prettify" not in plan
    if requires_readable:
        warning = _ensure(
            _has_completed_step(checkpoints, episode_id, "prettify"),
            "Thematize-only requires a readable transcript. Run prettify first or include it in the plan.",
        )
        if warning:
            return warning

    requires_themes = "classify" in plan and "thematize" not in plan
    if requires_themes:
        warning = _ensure(
            _has_completed_step(checkpoints, episode_id, "thematize"),
            "Classification requires a thematized transcript. Run thematize first or include it in the plan.",
        )
        if warning:
            return warning

    requires_classified = "vocative" in plan and "classify" not in plan
    if requires_classified:
        warning = _ensure(
            _has_completed_step(checkpoints, episode_id, "classify"),
            "Vocative detection requires a classified transcript. Run classify first or include it in the plan.",
        )
        if warning:
            return warning

    return None


def _completed_step_names(checkpoints: CheckpointStore, episode_id: str) -> set[str]:
    return {
        cp.step
        for cp in checkpoints.get_episode(episode_id)
        if cp.status == "completed"
    }


def _prompt_resume_choice(episode: EpisodeViewState, completed: Sequence[str]) -> bool:
    completed_list = ", ".join(step.capitalize() for step in completed)
    print(
        f"\nExisting checkpoints found for {episode.episode_key}: {completed_list}",
        flush=True,
    )
    print("Choose whether to resume from the last completed step or start over.", flush=True)
    while True:
        choice = input("Resume from checkpoint? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("", "n", "no"):
            return False
        print("Please enter 'y' or 'n'.", flush=True)


def prepare_execution_plan(
    episode: EpisodeViewState,
    selected_steps: Optional[tuple[str, ...]],
    checkpoints: CheckpointStore,
) -> tuple[Optional[tuple[str, ...]], bool]:
    desired_plan: tuple[str, ...] = selected_steps or STEP_ORDER
    completed = _completed_step_names(checkpoints, episode.episode_id)
    overlapping = tuple(step for step in desired_plan if step in completed)
    if not overlapping:
        # No overlap with existing checkpoints, treat as fresh start
        return selected_steps, False

    # Auto-resume semantics: continue with steps not yet completed
    remaining = tuple(step for step in desired_plan if step not in completed)
    if remaining:
        return remaining, True

    # Everything in desired plan already completed; restart from scratch
    if selected_steps is None:
        print(
            "All requested steps are already complete. Restarting pipeline from scratch.",
            flush=True,
        )
        checkpoints.delete_episode(episode.episode_id)
        return None, False
    # For partial plans, honor the user's selection without wiping checkpoints
    # so we do not unintentionally alter the plan based on status consistency checks.
    return selected_steps, False


def init_dependencies(config: MywConfig) -> tuple[CatalogService, PipelineRunner, QueueController]:
    catalog = PodcastCatalog(data_root=config.data_dir)
    catalog_service = CatalogService(config, catalog)
    queue = QueueController(enqueue_delay=0.0)
    checkpoints = CheckpointStore(config.db_path)
    runner = PipelineRunner(
        config=config,
        catalog=catalog,
        queue=queue,
        checkpoints=checkpoints,
    )
    return catalog_service, runner, queue


def configure_logging(config: MywConfig) -> logging.Logger:
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    return setup_logging(config.log_level, config.data_dir / "logs", handlers=[console_handler])


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    configure_logging(config)
    catalog_service, pipeline_runner, queue = init_dependencies(config)

    episodes = run_catalog_sync(catalog_service)
    if not episodes:
        return 1

    episode = select_episode(episodes)
    if episode is None:
        print("No episode selected. Exiting.")
        return 0

    selected_steps, scope_label = select_pipeline_scope(episode, pipeline_runner.checkpoints)
    # Determine pipeline_id and handle consistency
    pipeline_id: Optional[str]
    if scope_label == "Full pipeline":
        # Hard reset for full pipeline: wipe checkpoints and status, start from the beginning
        pipeline_id = str(uuid4())
        pipeline_runner.checkpoints.delete_pipeline(episode.episode_id, None)
        steps = STEP_ORDER
        resume_requested = False
    else:
        # Compute plan respecting existing artefacts
        steps, resume_requested = prepare_execution_plan(episode, selected_steps, pipeline_runner.checkpoints)
        status_row = pipeline_runner.checkpoints.get_pipeline_status(episode.episode_id)
        pipeline_id = (status_row or {}).get("pipeline_id") if status_row else None
        if not pipeline_id:
            pipeline_id = str(uuid4())
        # Consistency check: if last_completed_step is missing, restart from that step
        # Only apply this check when user hasn't explicitly selected a partial pipeline
        # (i.e., when it's a resume scenario, not a partial pipeline selection)
        if selected_steps is None and status_row and status_row.get("last_completed_step"):
            last_completed = str(status_row["last_completed_step"])
            cp = pipeline_runner.checkpoints.get_step(episode.episode_id, last_completed)
            if not (cp and cp.status == "completed"):
                print(f"Warning: Inconsistent pipeline state detected (missing checkpoint for {last_completed}). Restarting from that step.", flush=True)
                if last_completed in STEP_ORDER:
                    start_index = STEP_ORDER.index(last_completed)
                    steps = STEP_ORDER[start_index:]
                    resume_requested = True
    monitor = PipelineMonitor(episode.episode_id)
    pipeline_runner.callback = monitor

    pipeline_runner.start()
    queue.enqueue(episode.episode_id, resume=resume_requested, steps=steps, pipeline_id=pipeline_id)
    status_label = f"{scope_label} (resume)" if resume_requested else scope_label
    queue.set_status(episode.episode_id, "In progress", f"Queued ({status_label})")

    print(f"\nRunning {scope_label.lower()} for episode {episode.episode_key}...", flush=True)
    try:
        monitor.wait()
    except KeyboardInterrupt:
        print("\nStopping pipeline...", flush=True)
        queue.stop_current()
        monitor.wait()
    finally:
        pipeline_runner.shutdown()

    if monitor.success:
        print(f"Pipeline completed: {monitor.final_remarks or 'Done.'}")
        return 0

    print(f"Pipeline stopped: {monitor.final_remarks or 'Unknown reason.'}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

