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
        print(f"{index:3d}. {show} - {title} [{episode.episode_id}]")
    print()

    while True:
        selection = input("Select an episode (number or ID, blank to cancel): ").strip()
        if not selection:
            return None
        if selection.isdigit():
            choice = int(selection)
            if 1 <= choice <= len(episode_list):
                return episode_list[choice - 1]
        else:
            for episode in episode_list:
                if episode.episode_id == selection:
                    return episode
        print("Invalid selection. Try again.", flush=True)


def select_pipeline_scope(
    episode: EpisodeViewState,
    checkpoints: CheckpointStore,
) -> tuple[Optional[tuple[str, ...]], str]:
    options: dict[str, tuple[str, Optional[tuple[str, ...]]]] = {
        "1": ("Full pipeline", None),
        "2": ("Transcription only", ("transcribe",)),
        "3": ("Diarization only", ("diarize",)),
        "4": ("Assignment only", ("assign",)),
    }
    print("Pipeline scopes:\n")
    for key, (label, _) in options.items():
        print(f"  {key}. {label}")
    print()

    default = "1"
    while True:
        selection = input(f"Select pipeline scope [{default}]: ").strip() or default
        option = options.get(selection)
        if not option:
            print("Invalid selection. Try again.", flush=True)
            continue
        label, plan = option
        if plan == ("diarize",) and not _has_completed_step(checkpoints, episode.episode_id, "transcribe"):
            print(
                "Diarization-only requires an existing transcript. Run transcription first or choose a different option.",
                flush=True,
            )
            continue
        if plan == ("assign",):
            missing_requirements = [
                name
                for name, step in (("transcription", "transcribe"), ("diarization", "diarize"))
                if not _has_completed_step(checkpoints, episode.episode_id, step)
            ]
            if missing_requirements:
                requirement_text = " and ".join(missing_requirements)
                print(
                    f"Assignment-only requires completed {requirement_text}. Run the necessary steps first or choose a different option.",
                    flush=True,
                )
                continue
        normalized = tuple(plan) if plan else None
        return normalized, label


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
    return True


def _completed_step_names(checkpoints: CheckpointStore, episode_id: str) -> set[str]:
    return {
        cp.step
        for cp in checkpoints.get_episode(episode_id)
        if cp.status == "completed"
    }


def _prompt_resume_choice(episode: EpisodeViewState, completed: Sequence[str]) -> bool:
    completed_list = ", ".join(step.capitalize() for step in completed)
    print(
        f"\nExisting checkpoints found for {episode.episode_id}: {completed_list}",
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
        return selected_steps, False

    if _prompt_resume_choice(episode, overlapping):
        remaining = tuple(step for step in desired_plan if step not in completed)
        if remaining:
            return remaining, True
        print(
            "All requested steps are already complete. Restarting pipeline from scratch.",
            flush=True,
        )
        checkpoints.delete_episode(episode.episode_id)
        return (selected_steps if selected_steps is not None else None), False

    checkpoints.delete_episode(episode.episode_id)
    return (selected_steps if selected_steps is not None else None), False


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
    steps, resume_requested = prepare_execution_plan(episode, selected_steps, pipeline_runner.checkpoints)
    monitor = PipelineMonitor(episode.episode_id)
    pipeline_runner.callback = monitor

    pipeline_runner.start()
    queue.enqueue(episode.episode_id, resume=resume_requested, steps=steps)
    status_label = f"{scope_label} (resume)" if resume_requested else scope_label
    queue.set_status(episode.episode_id, "In progress", f"Queued ({status_label})")

    print(f"\nRunning {scope_label.lower()} for episode {episode.episode_id}...", flush=True)
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

