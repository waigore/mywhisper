# Myw Textual Frontend — Condensed Specification

## Purpose

- Provide a terminal-native Textual UI (`myw.py`) that manages podcast catalogs and runs the full mywhisper pipeline (transcribe → diarize → assign → prettify → thematize).
- Wrap generator-driven jobs in a responsive, non-blocking experience while matching existing CLI capabilities.

## Guiding Principles

- **KISS navigation:** just `PodcastListingScreen` and `PodcastViewScreen`.
- **Non-blocking:** pipeline work lives in a worker thread; UI stays responsive.
- **Single source of truth:** queue controller owns authoritative state; only one pipeline at a time.
- **Observability:** every user action and pipeline transition is logged/audited.
- **Modular boundaries:** keep UI, queue orchestration, and I/O integrations independent.

## Architecture

- Textual app boots env/logging, instantiates screens plus services: `CatalogService`, `QueueController`, `PipelineRunner`.
- Worker thread runs `PipelineRunner`; UI thread receives Textual `Message` updates. Shared data uses locks/conditions.
- `CatalogService` reads Apple Podcasts cache into `mywhisper.podcasts.PodcastCatalog`, emitting `EpisodeViewState` rows.
- `QueueController` exposes enqueue/dequeue/stop/resume, persists state to SQLite, and broadcasts status to widgets.
- `PipelineRunner` executes the selected `step_plan`, persists checkpoints, emits progress, and records artefact paths.

## Configuration & Storage

- Load `.env` via `python-dotenv`; reuse mywhisper helpers for logging/config.
- Required vars: `MYW_LOG_LEVEL`, `MYW_DATA_DIR`, `MYW_PODCAST_CACHE_PATH`, `MYW_WHISPER_MODEL`, `MYW_DB_PATH`. Fail fast when missing.
- `myw.db` schema (all timestamps ISO8601):
  - `queue_items`: `episode_id` PK, `position`, `status` (`downloaded|queued|in_progress|stopped|completed|failed`), `step_plan` JSON, `current_step`, `progress_percent`, `remarks`, `resume_token`, `created_at`, `updated_at` (index on `position`).
  - `queue_events`: append-only audit log with `event_type` (`enqueue|start|stop|resume|dequeue|error`) and `payload_json` (index on `episode_id`).
  - `pipeline_checkpoints`: existing table extended to store every canonical step, `plan_hash`, artefact paths (including `_readable.txt`, `_themes.json`), payload/details JSON, elapsed seconds.
  - `artefact_registry`: `(episode_id, artefact_kind)` PK mapping `transcript|diarization|with_names|readable|themes` to file paths + metadata JSON.
- Migration plan: additive schema updates, backfill queue/artefact tables from in-memory queue and existing checkpoints.

## Module Layout

```
mywhisper/myw/
  app.py            # Textual App + screen wiring
  config.py         # env + path validation
  logging.py        # shared logging setup
  models.py         # EpisodeViewState, PipelineStatus
  services/
    catalog.py
    queue.py
    pipeline.py
  screens/
    listing.py
    view.py
  widgets/
    episode_table.py
    progress_bar.py
  messages.py
  myw.py            # entrypoint (textual run / python -m)
```

Business logic resides in services; UI components remain thin/testable.

## Core Flows

- **Startup:** load config → init services → `CatalogService.sync_from_cache()` → emit `CatalogSynced` to render listing; `(r)` refresh repeats diffed sync.
- **Podcast listing:** sortable table (`Episode`, `Podcast`, `Downloaded At`, `Status`, `Remarks`), optional empty-state label, footer progress bar visible only when pipeline active. Keyboard: arrows for selection, `enter`/`v` to view, `s` to stop/resume, `r` to refresh.
- **Podcast view:** shows metadata (titles, description, size, duration, paths, IDs), artefact summary, `(e)`/`enter` enqueue, `(b)` back, breadcrumb `Listing > Episode`.

## Queue & Pipeline

- Queue allows only one active job; remaining episodes stay FIFO but can be dequeued. Statuses: `Downloaded`, `In progress`, `Stopped`, `Completed`.
- Each queue item carries a validated `step_plan`; runner enforces prerequisites (e.g., diarization requires transcript) and scales progress to planned steps.
- `PipelineRunner` stages:
  1. Transcribe (Whisper)
  2. Diarize (PyAnnote)
  3. Assign (LLM-produced speaker names)
  4. Prettify (build `_readable.txt` from assignments)
  5. Thematize (turn readable transcript into `_themes.json`)
- Stage gating: prettify only runs when an assignment artefact exists; thematize requires a readable transcript. Missing artefacts trigger automatic reloads from checkpoints (or regeneration) before progressing.
- After every step, persist checkpoints, update artefact registry, send Textual progress messages, and refresh listing remarks/progress bar.
- Stop/resume:
  - `stop_current` sets flag, runner halts between steps, marks `Stopped`, writes checkpoints.
  - Resume loads checkpoints, verifies artefacts, skips completed steps, continues next pending step.
- Partial plans reuse artefacts; restarts can optionally wipe checkpoints before re-run.

## Logging, Telemetry, and Recovery

- Dedicated `myw` logger; `INFO` for user actions, `DEBUG` for queue/pipeline, `WARNING/ERROR` for failures. Optionally surface warnings in UI.
- Queue events mirror user actions for audit. Pipeline logs always include episode ID, step, elapsed, artefacts produced.
- Error handling:
  - Catalog sync failure → non-blocking alert + log, keep prior table.
  - Pipeline error → mark `Stopped`, show message in remarks, allow resume, capture stack trace.
  - Worker watchdog restarts background thread if it dies with pending work.
  - Ignore invalid shortcuts; sanitize all user input.

## Extensibility

- Additional screens (settings, transcript viewer) can be dropped under `screens/` without changing services.
- Queue persistence can evolve (e.g., richer SQLite schema) by swapping `QueueController`.
- New pipeline steps register in a step registry consumed by `PipelineRunner`.
- Message bus hooks can forward progress to WebSocket/notification listeners.

## Conversational CLI (`mywconv.py`)

- Minimal guided CLI that reuses the same services: load config, sync catalog, prompt for episode, then choose between scopes:
  - Full pipeline (from beginning)
  - Resume pipeline (only shown if the episode is not fully completed)
  - Partial pipeline (choose starting and ending steps)
- Partial pipeline behavior:
  - Steps are chosen from the canonical order: `transcribe`, `diarize`, `assign`, `prettify`, `thematize`.
  - Starting step is constrained to be at or before the current in-progress step in `pipeline_status.current_step` (when present). If no current step, any step can be selected.
  - Ending step must be at or after the starting step. If the same, only that step runs.
  - Artefact prerequisites still apply when skipping steps (e.g., thematize requires a readable transcript); the CLI validates selected ranges and prompts to adjust if prerequisites are not met.
- Resume semantics: start from the next pending step (e.g., if assignment is last completed, run assign → prettify → thematize).
- Warn when prerequisites are missing; validate artefacts and automatically regenerate missing ones when required by downstream steps.
- Translate choice into `step_plan`, enqueue via `QueueController`, stream progress through `PipelineMonitor`, exit `0` on success or `2` on stop/prereq failure.

## Pipeline IDs and Status Persistence

- Every pipeline run has a `pipeline_id` (UUID).
- Starting a Full pipeline wipes previous checkpoints associated with any prior `pipeline_id` for the episode and creates a new `pipeline_id`.
- Resuming reuses the existing `pipeline_id` recorded for the episode.
- `myw.db` maintains a `pipeline_status` table that tracks: `episode_id` (PK), `pipeline_id`, `status` (`queued|in_progress|stopped|completed|failed`), `current_step`, `last_completed_step`, `progress`, `remarks`, `updated_at`.
- Checkpoint rows also record `pipeline_id` to correlate artefacts with a specific run.
- Consistency checks: if `pipeline_status` indicates a last completed step but the corresponding checkpoint is missing (or later-step checkpoints exist while the claimed last completed is missing), warn the user and restart from the first missing step.

