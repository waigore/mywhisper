## MyWhisper Specification

### Purpose
`mywhisper` delivers reproducible podcast-processing pipelines (transcribe → diarize → label speakers → prettify → thematize) driven by a cached Apple Podcasts catalog. Every stage reuses the original cache file, keeps artefacts deterministic, and emits progress through generators for batch or interactive control.

---

### Package Layout
- `mywhisper/__init__.py` bootstrap + exports
- `mywhisper/transcribe.py` Whisper pipelines
- `mywhisper/diarize.py` PyAnnote diarization
- `mywhisper/assign.py` speaker naming
- `mywhisper/podcasts.py` Apple Podcasts catalog + importer
- `mywhisper/config.py` shared config helpers
- Tests mirror the package under `tests/`

---

### Core Principles
- Keep submodules cohesive, class-based, and factory-backed (`from_config` hides model and IO setup).
- Emit work as generators returning `PipelineEvent` structures when `yield_progress=True`.
- Use the Apple Podcasts cache path stored on `PodcastEpisode.source_path`; never copy `xxxx.mp3` into `mywhisper` data directories except for temporary derived chunks.
- All persistent artefacts live under `data/` (configurable) and start with the deterministic eight-digit episode key (e.g. `12345678_with_names.json`).
- Maintain ≥80 % automated coverage and bias toward simple error handling (fail fast on unexpected states).

---

### Cross-Cutting Conventions
- **Logging**: root logger `mywhisper`; modules derive child loggers and optionally accept injected `logging.Logger` instances. Provide `configure_logging(level)` helper.
- **Domain Models**: `PodcastEpisode`, `TranscriptSegment`, `SpeakerCluster`, `SpeakerAssignment`, and `PipelineEvent` dataclasses define shared IO shapes.
- **Artefact Rules**: `data_root` defaults to `data/`; optional podcast-slug directories, but filenames always begin with the episode key. Temporary chunk/export folders also sit under `data_root`.
- **Progress Hooks**: long-running tasks (chunking, diarization, LLM inference) may stream events via callbacks for CLI/GUI consumption.

---

## Module Requirements

### `mywhisper/transcribe`
- Responsibilities: normalize cached audio (downmix + resample), invoke Whisper.cpp models, persist transcripts as `{episode_key}_whisper.json`, and optionally reuse cached segments.
- Key classes: `TranscriptionConfig` (paths, language, chunking, device), `WhisperModelFactory`, `AudioChunker`, and `PodcastTranscriber`.
- Artefacts: chunk WAVs (only when chunking), transcript JSON aligned with example schema.

### `mywhisper/diarize`
- Responsibilities: run PyAnnote end-to-end on the cache file, surface `ProgressHook` updates, write RTTM plus optional diarization JSON.
- Key classes: `DiarizationConfig` (HF token, device, output dirs), `WaveformLoader`, `PyAnnotePipelineFactory`, `DiarizationPipeline`.
- Artefacts: `{episode_key}.rttm`, optional `{episode_key}_diarization.json`.

### `mywhisper/assign`
- Responsibilities: merge transcripts + diarization, build speaker profiles, gather candidate names, drive LLM-based inference (default Ollama), and persist `{episode_key}_with_names.json`.
- Key classes: `AssignmentConfig`, `SpeakerProfileBuilder`, `CandidateRoster`, `LLMClient`/`OllamaClient`, `SpeakerInferenceEngine`.
- Behaviour: enforce confidence threshold (fallback `"UNKNOWN"`), support iterative refinement, expose generator progress hooks.
- Artefacts: enriched transcript JSON, optional CSV analytics export.

### Prettify Step
- Module: `mywhisper/prettify.py` exporting `PrettifyConfig` + `TranscriptPrettifier`.
- Collapse contiguous segments (speaker match, ≤`collapse_gap_seconds` pause, `max_block_characters` guard), format as `<speaker name> (<speaker_id>): <text>`, and write `{episode_key}_readable.txt`.
- Register artefact kind `readable_transcript` via `PodcastCatalog.record_artefact(...)` and emit generator-driven `PipelineEvent(stage="prettify", step_name="prettify")` for load/collapse/persist.
- Config knobs: `data_root`, `collapse_gap_seconds` (default 1.5 s), optional `max_block_characters`, output subdir override.

### Thematize Step
- Module: `mywhisper/thematize.py` exposing `ThematizeConfig` + `EpisodeThematizer`.
- Ingest readable transcript, normalize into ≤2 000-token chunks with ~15 % overlap, prompt configurable LLM (`llm_model`, default Ollama) using templated instructions, and parse JSON `[{"theme","summary","highlights"}]` payloads.
- Merge adjacent identical themes, persist `{episode_key}_themes.json`, register artefact kind `themes`, and emit `PipelineEvent(stage="thematize")` per chunk plus final persist event (checkpoint includes `themes_path`).
- On LLM failure or empty transcript, fall back to a single `fallback_theme` section summarizing the transcript snippet + failure reason.

### `mywhisper/podcasts`
- Responsibilities: index cache-resident episodes, expose queries (by show title, GUID, date, path), and orchestrate ingestion without copying audio.
- Storage: SQLite `data/catalog.db` with `episodes` and `artefacts` tables; artefact keys reuse the episode key stub.
- Classes: `PodcastCatalog` (CRUD + artefact registration) and `ApplePodcastsImporter` (scan cache, emit discovery events, optional metadata enrichment).
- Integration: pipelines rely on `PodcastEpisode.source_path` (always the cache path) and call `catalog.record_artefact(...)` whenever they persist outputs.

---

## Workflow Summary
1. **Discover** episodes via `ApplePodcastsImporter`, store metadata + cache paths in `PodcastCatalog`.
2. **Transcribe** using `PodcastTranscriber`; reuse cached chunks and save `{episode_key}_whisper.json`.
3. **Diarize** through `DiarizationPipeline`, writing `{episode_key}.rttm` and optional diarization JSON.
4. **Assign speakers** via `SpeakerInferenceEngine`, yielding `{episode_key}_with_names.json`.
5. **Prettify** to readable text blocks.
6. **Thematize** into structured topic sections.

Each stage resumes from artefacts identified by the episode key and records outputs in the catalog for traceability.

---

## Logging, Metrics, and Events
- All pipelines emit `PipelineEvent(stage, step_name, payload, elapsed)` objects and may forward them through callbacks.
- Modules inherit logging configuration from `mywhisper` unless callers override per class.
- Metrics (elapsed time per stage, artefact paths) remain minimal but standardized within the event payloads.

---

## External Dependencies
- Core ML/audio: `pywhispercpp`, `torch`, `torchaudio`, `pyannote.audio`, `pyannote.core`, `huggingface_hub`.
- Utilities: `tqdm`, `numpy`, `scikit-learn`, `joblib`, `spaCy` (optional), `requests` (Ollama), `pydub`, standard library `sqlite3`.
- Philosophy: prefer upstream error handling; only guard predictable recoverable scenarios (e.g., missing spaCy model or HF token).

---

## Open Points
- Metadata gaps (description, roster) must not break pipelines; degrade gracefully.
- Prompt templates remain configurable and may evolve, but defaults match the examples.
- Parallelism is caller-managed; generators simply expose checkpoints for orchestration frameworks.

---

## CLI Resume Semantics and Persistence
- CLI offers three scopes: Full pipeline (from beginning), Resume pipeline (only when not fully completed), and Partial pipeline (user-selected start and end).
- Resume starts at the next pending step and runs through the end.
- Partial pipeline lets the user choose a starting step and ending step from the canonical order (`transcribe → diarize → assign → prettify → thematize`).
  - Constraint: the starting step must be at or before the current in-progress step recorded in `pipeline_status.current_step` for the episode (when present). If no active `current_step`, any step can be chosen as the start.
  - Constraint: the ending step must be at or after the selected starting step. If start equals end, only that step runs.
  - Artefact prerequisites still apply when skipping steps; validations ensure required artefacts exist (or the user must include the producing step in the selection).
- Persistence adds `pipeline_id` to checkpoints and a `pipeline_status` table tracking overall state per episode.
- Consistency guard: if `pipeline_status.last_completed_step` lacks a matching checkpoint (or later-step checkpoints exist without it), the system warns and restarts from the first missing step.

