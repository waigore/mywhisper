from __future__ import annotations

import logging
import sys
import threading
from typing import Iterable, Optional

from mywhisper.checkpoints import CheckpointStore
from mywhisper.myw.config import ConfigError, MywConfig, load_config
from mywhisper.myw.logging import setup_logging
from mywhisper.myw.messages import PipelineCompleted, PipelineProgress, PipelineStopped
from mywhisper.myw.models import EpisodeViewState, PipelineStatus
from mywhisper.myw.services.catalog import CatalogService
from mywhisper.myw.services.pipeline import PipelineRunner
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

    monitor = PipelineMonitor(episode.episode_id)
    pipeline_runner.callback = monitor

    pipeline_runner.start()
    queue.enqueue(episode.episode_id)
    queue.set_status(episode.episode_id, "In progress", "Queued for CLI run")

    print(f"\nRunning pipeline for episode {episode.episode_id}...", flush=True)
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

