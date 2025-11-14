# Myw Textual Frontend Specification

## Purpose

- Provide a terminal-native frontend (`myw.py`) for managing podcast catalog entries and invoking mywhisper transcription pipelines.
- Abstract mywhisper’s generator-driven workflows behind a responsive Textual UI that keeps users informed without blocking.
- Maintain parity with existing CLI capabilities while introducing richer status tracking and queue control.

## Design Principles

- **KISS:** keep navigation shallow (two primary screens) and reuse mywhisper primitives wherever possible.
- **Non-blocking UI:** all long-running work moves to a dedicated worker thread that communicates progress back to the Textual app.
- **Observability:** log every user action and pipeline transition for auditability and debugging.
- **Predictable state:** only one pipeline runs at a time; UI reflects authoritative state derived from the queue manager.
- **Modularization:** isolate UI concerns from queue/pipeline orchestration and from I/O integrations.

## Architecture Overview

- `Textual` application with two screens:
  - `PodcastListingScreen`: default entry point showing catalog and pipeline state.
  - `PodcastViewScreen`: detail view with enqueue action.
- Core services:
  - `CatalogService`: reads Apple Podcasts cache, syncs with `mywhisper.podcasts.PodcastCatalog`.
  - `QueueController`: manages FIFO queue, exposes start/stop/resume controls, persists queue state in memory (no DB).
  - `PipelineRunner`: consumes queue entries, runs mywhisper pipeline steps (transcribe → diarize → assign), broadcasts progress messages/events.
- Communication patterns:
  - Worker thread executes `PipelineRunner`.
  - Main UI thread listens for queue/pipeline events via Textual `Message` subclasses.
  - Shared data structures (e.g., queue, episode status map) protected with threading locks.

## Environment & Configuration

- Load `.env` via `python-dotenv` at app startup (`myw/__main__.py`).
- Support variables:
  - `MYW_LOG_LEVEL`, `MYW_DATA_DIR`, `MYW_PODCAST_CACHE_PATH`, `MYW_WHISPER_MODEL`, etc.
  - `MYW_DB_PATH` pointing to `myw.db` (SQLite) that stores queue checkpoints and pipeline metadata.
- Defer to existing mywhisper config helpers (e.g., `mywhisper.config.configure_logging`) when possible.
- Fail fast with clear messaging if required env vars or cache paths are missing.

## Module Layout

```
mywhisper/myw/
  __init__.py
  app.py                # Textual App subclass, bootstraps screens and services
  config.py             # Env loading, defaults, path validation
  logging.py            # Logging setup bridging Textual + mywhisper logging
  models.py             # UI-facing dataclasses (EpisodeViewState, PipelineStatus)
  services/
    catalog.py          # CatalogService implementation
    queue.py            # QueueController with FIFO queue and state transitions
    pipeline.py         # PipelineRunner consuming queue entries
  screens/
    listing.py          # PodcastListingScreen
    view.py             # PodcastViewScreen
  widgets/
    episode_table.py    # Sortable DataTable + selection handling
    progress_bar.py     # Conditional progress display
  messages.py           # Textual Message definitions for cross-thread updates
  myw.py                # CLI entry: `textual run mywhisper.myw.myw:MywApp` (or direct `python -m mywhisper.myw.myw`)
```

- Keep public entrypoint slim; business logic resides in services.
- Provide unit-testable services independent of Textual by using plain Python classes.

## Startup & Catalog Sync Flow

1. Load env/config, set up logging.
2. Initialize `CatalogService` with Apple Podcasts cache path and `PodcastCatalog`.
3. Call `CatalogService.sync_from_cache()`, which:
   - Scans cache directory for downloaded episodes.
   - Registers episodes in catalog (via `mywhisper.podcasts.register_episode` or equivalent).
   - Produces in-memory `EpisodeViewState` list with status `Downloaded`.
4. Emit `CatalogSynced` message to populate listing table.
5. Provide manual sync via `(r)efresh`, reusing same service call and diffing statuses to update table efficiently.

## Podcast Listing Screen

- Components:
  - Sortable `DataTable` with columns: Episode, Podcast, Downloaded At, Status, Remarks.
    - Column semantics:
      - `Episode`: display the human-readable episode title sourced from the cache `metadata.plist` or, if missing, the Podcasts database (`ZMTEPISODE.ZTITLE`); fall back to the audio filename stem only as a last resort.
      - `Podcast`: display the show title from cache metadata or the Podcasts database (`ZMTPODCAST.ZTITLE`); fall back to `Unknown Show`.
      - `Downloaded At`: show the filesystem modification timestamp of the imported audio asset (preserving the original cache download time when copied); leave blank if the timestamp cannot be determined.
- Status enumerations: `Downloaded`, `In progress`, `Stopped`, `Completed`.
  - `Downloaded` explicitly denotes episodes discovered in the local Apple Podcasts cache that have not yet entered the pipeline.
  - Remarks:
    - For active pipeline: show current step description (e.g., `Transcribing`, `Diarizing`, `Assigning speakers`).
    - For stopped/completed: show last completed step or completion summary.
  - Optional `Label` displayed when no episodes available.
  - Footer progress widget (custom `ProgressBar`):
    - Only visible when `PipelineStatus.active`.
    - Shows percent overall progress + text of current step.
- Interactions:
  - Selection with arrow keys; `enter` or `(v)` opens Podcast View.
  - `(r)` triggers `CatalogService.sync_from_cache()`; disable while sync in progress.
  - `stop/resume` toggle command (`s` key):
    - Stop: requests `QueueController.stop_current()`; transitions status to `Stopped`.
    - Resume: when the current head is stopped, pressing `s` re-enqueues from the most recent checkpoint.
- Screen listens for `PipelineProgress`, `PipelineFinished`, `PipelineStopped` messages to update table and progress bar.
- Sorting:
  - Default sort by `Downloaded At` descending.
  - Allow toggling sort column via built-in DataTable headers.

## Podcast View Screen

- Displays metadata for selected episode:
  - Podcast title, Episode title, Description (render `N/A` when unavailable), File size, Duration, Download path, Current status.
  - If transcription artifacts exist, show summary (transcript path, diarization status).
- Commands:
  - `(b)` returns to `PodcastListingScreen`, preserving the prior table selection.
  - `(e)` / `enter` triggers `QueueController.enqueue(episode_id)`.
  - If queue empty and pipeline idle, enqueue starts pipeline immediately.
  - Provide confirmation toast/message to user.
- Navigation:
  - Support explicit `(b)` back navigation alongside default Textual shortcuts (escape/`ctrl+q`).
  - Breadcrumb header showing `Listing > Episode`.

## Queue & Pipeline Processing

- QueueController responsibilities:
  - Maintain ordered queue of episode IDs.
  - Track per-episode state (`Downloaded`, `In progress`, `Stopped`, `Completed`).
  - Expose `enqueue`, `dequeue`, `stop_current`, `resume`, `current_episode` APIs.
  - Persist checkpoint metadata (e.g., last completed pipeline step, intermediate artefact paths, timestamps) to `myw.db` so progress survives app restarts.
  - Broadcast queue updates via Textual messages.
- PipelineRunner:
  - Runs in background thread started at app init.
  - Waits on queue condition variable; pulls next episode when available.
  - Executes pipeline steps sequentially using mywhisper components:
    1. Transcribe (Whisper)
    2. Diarize (PyAnnote)
    3. Assign speaker names (LLM)
  - Each step uses a `PipelineEventAdapter` that iterates the generator outputs (with `yield_progress=True`) to derive progress percentages and assemble checkpoint payloads (step id, chunk index, artefact paths).
  - After each step:
    - Update episode remarks/status.
    - Emit progress events with percent and current step name.
    - Persist intermediate outputs under `data/` and record their paths in `myw.db` to make them discoverable on resume.
  - On completion:
    - Mark status `Completed`, remove from queue, emit completion event.
  - Supports stop:
    - `stop_current` sets stop flag; runner checks between steps, gracefully cancels, marks `Stopped`, writes checkpoint metadata and ensures step outputs remain accessible.
  - Supports resume:
    - Runner reads checkpoint data from `myw.db`, verifies intermediate outputs exist, skips finished steps, restarts next pending step.
- Thread Safety:
  - Use `threading.Lock` for shared state (statuses, queue list).
  - Use `Queue` or `deque` + `Condition`.

## Logging & Telemetry

- Configure structured logging with a dedicated `myw` logger namespace.
- Log levels:
  - `INFO` for user actions (screen commands, enqueue, stop/resume, refresh).
  - `DEBUG` for pipeline checkpoints and queue transitions.
  - `WARNING/ERROR` for exceptions or failed Syncs.
- Integrate Textual log handler to surface warnings in UI status bar (optional).
- Ensure pipeline logs include episode ID, step name, elapsed time.

## Error Handling & Recovery

- Catalog sync failures:
  - Show non-blocking alert in UI, log error, keep previous listing.
- Pipeline execution errors:
  - Mark episode `Stopped`, store error message in remarks, allow resume.
  - Capture stack trace in log file.
- Thread failures:
  - Watchdog in `QueueController` restarts worker thread if it exits unexpectedly and queue non-empty.
- Input validation:
  - Sanitize user commands; ignore invalid key bindings gracefully.

## Extensibility Notes

- Future screens (e.g., settings, transcript viewer) can live under `screens/`.
- Queue persistence can later move to SQLite without breaking UI by swapping `QueueController`.
- Additional pipeline steps should extend a simple step registry consumed by `PipelineRunner`.
- Provide hooks for publishing events to external observers (e.g., WebSocket, notifications) via message bus wrapper.

