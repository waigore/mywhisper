## MyWhisper Module Specification

### Purpose
`mywhisper` provides reproducible pipelines for podcast transcription, speaker diarization, and speaker identity inference, backed by a lightweight catalog of locally cached podcast episodes. The module targets batch and interactive usage, emphasizing reuse of raw audio without duplication, predictable temporary artefact layout, and composable, generator-driven pipelines.

---

### Package Layout
- `mywhisper/__init__.py` — package exports, logging bootstrap.
- `mywhisper/transcribe.py` — transcription pipeline classes.
- `mywhisper/diarize.py` — diarization and speaker clustering pipelines.
- `mywhisper/assign.py` — speaker name inference and transcript enrichment.
- `mywhisper/podcasts.py` — catalog of Apple Podcasts cache episodes.
- `mywhisper/config.py` (optional) — shared configuration helpers (e.g. data directories, HuggingFace token loader).

Tests live under `tests/` mirroring the package structure.

---

### Design Principles
- Implement submodules as cohesive class-based components with minimal surface area.
- Provide factory methods to hide complex configuration and model loading details.
- Configure logging per submodule; root logger `mywhisper` supplies baseline level.
- Expose pipeline stages via generators so callers can orchestrate long-running work.
- Reference original podcast files directly; avoid redundant audio copies.
- Store temporary artefacts under `data/` with 8-character alphanumeric keys for traceability.
- Maintain ≥90 % automated test coverage (unit and integration); enforce coverage thresholds in CI.
- Follow KISS: limit defensive handling of unlikely error states.

---

### Cross-Cutting Conventions
- **Logging**
  - Root logger name: `mywhisper`. Submodules derive child loggers (`mywhisper.transcribe`, etc.).
  - Each public class accepts an optional `logging.Logger` or log level; default is a child logger with module name.
  - Provide module-level `configure_logging(level: int | str)` that sets the root logger and optional stream/file handlers.

- **Temporary artefacts**
  - Base directory: `data/` (configurable). Pipelines operate relative to a `data_root: Path`.
  - Artefact keys: eight-character mixed-case base36 strings (e.g. `A9F3D2B1`) generated via `secrets.token_hex(4).upper()`.
  - Artefact naming template: `{podcast_slug}__{artefact_key}__{purpose}.{ext}` to allow reverse lookup.
  - Pipelines only reference the source podcast file by path; they never duplicate the original audio unless transformation is unavoidable (e.g. temporary chunk export for diarization).

- **Shared domain models**
  - `PodcastEpisode`: dataclass capturing `episode_id`, `show_title`, `episode_title`, `description` (optional str), `published_at`, `author`, `source_path`, `duration_sec`, `metadata` (dict).
  - `TranscriptSegment`: dataclass with `start`, `end`, `text`, optional `speaker_id`, `speaker_name`, `confidence`, `metadata`.
  - `SpeakerCluster`: dataclass wrapping diarization output (`speaker_id`, `segments: list[Segment]`, `profile: SpeakerProfile`).

- **Factories & configuration**
  - Each primary pipeline class offers `from_config(**kwargs)` factory reading high-level settings (e.g. sample rate, HF token, chunk size) and constructing dependencies.
  - Factories hide third-party model initialization details.

- **Pipelines as generators**
  - Long-running processes (chunk extraction, diarization, LLM inference) expose generator stages, allowing callers to iterate over progress and optionally short-circuit.
  - Public pipeline methods accept `yield_progress: bool = False`; when enabled they return iterables yielding structured progress events (`PipelineEvent` dataclass with `stage`, `step_name`, `payload`, `elapsed`).
  - Events expose enough metadata (episode id, step identifiers, artefact paths) for external adapters to persist checkpoints without modifying core pipeline control flow.

---

## Module Specifications

### `mywhisper/transcribe`

#### Responsibilities
- Load or stream podcast audio without duplicating source files.
- Normalise audio and sample rate for Whisper.cpp.
- Run transcription using a preloaded Whisper model.
- Persist transcript JSON for reuse by downstream modules.

#### Key Classes
- `TranscriptionConfig`
  - Fields: `model_path`, `language`, `target_sample_rate`, `chunk_duration`, `chunk_overlap`, `output_dir`, `device`.
  - Method `resolve_paths(data_root: Path) -> None` ensures directories exist.

- `WhisperModelFactory`
  - Static method `create(config: TranscriptionConfig) -> Model` returning a `pywhispercpp.model.Model` instance.

- `AudioChunker`
  - Initialised with `chunk_duration`, `overlap`, `target_sample_rate`.
  - Method `iterate_chunks(source: Path) -> Generator[AudioChunk, None, None]` yields `AudioChunk` dataclasses referencing in-memory tensors and on-disk wav paths inside `data_root/transcribe/{artefact_key}`.
  - Resamples via `torchaudio.transforms.Resample` only when necessary.
  - Ensures stereo audio is downmixed to mono, mirroring `examples/transcribe_audio.py`.

- `PodcastTranscriber`
  - Constructor accepts `podcast: PodcastEpisode`, `config: TranscriptionConfig`, `model: Model`, `chunker: AudioChunker`, and optional logger.
  - `from_config(podcast: PodcastEpisode, config: TranscriptionConfig) -> PodcastTranscriber` loads the model and chunker.
  - `transcribe(yield_progress: bool = False) -> list[TranscriptSegment] | Generator[PipelineEvent, None, list[TranscriptSegment]]`
    - Steps: `prepare_audio` (full audio or chunked), `run_model` (calls `model.transcribe`), `normalize_segments` (convert centiseconds to seconds via `WHISPER_TIME_FACTOR` equivalent), `persist_transcript`.
    - Persists transcripts under `data/transcripts/{podcast_slug}/{artefact_key}_whisper.json`.
  - `load_cached_segments() -> list[TranscriptSegment]` reads a previously stored transcript.

#### Artefacts
- Extracted chunk files only exist when chunking is required. Each chunk is stored once and reused between runs, keyed by artefact ID.
- Transcript JSON schema matches example script output: list of dicts with `start`, `end`, `text`.

---

### `mywhisper/diarize`

#### Responsibilities
- Chunk long-form audio for diarization while reusing existing chunk exports when possible.
- Run PyAnnote speaker diarization pipeline with configurable speaker counts.
- Extract speaker embeddings, cluster into consistent global speaker IDs, and publish RTTM plus JSON transcripts.

#### Key Classes
- `DiarizationConfig`
  - Fields: `hf_token`, `num_speakers`, `chunk_minutes`, `overlap_seconds`, `embedding_window`, `output_dir`, `device`.
  - Provides `auth()` helper to call `huggingface_hub.login`.

- `ChunkScheduler`
  - `schedule(podcast: PodcastEpisode) -> Generator[AudioChunk, None, None]`
  - Reuses existing chunk files in `data/audio_chunks/{podcast_slug}/`.
  - Emits events describing `global_start`, `global_end`, `path`.

- `PyAnnotePipelineFactory`
  - Static methods to create `Pipeline` and `Inference` instances (embedding extractor).
  - Handles moving models to `config.device`.

- `SpeakerClusterer`
  - Maintains reference embeddings (`joblib` persistence to `data/transcripts/{podcast_slug}/{artefact_key}_clusters.pkl`).
  - Methods: `fit_reference(chunk: AudioChunk, annotation: Annotation) -> None`, `assign(embeddings: np.ndarray) -> np.ndarray` (returns speaker ids), `save/load`.

- `DiarizationPipeline`
  - Constructor parameters: `podcast`, `config`, `chunk_scheduler`, `pipeline`, `embedding_inference`, `clusterer`.
  - `from_config(...)` builds dependencies, optionally seeding reference chunk (first chunk).
  - `run(yield_progress: bool = False)` generator with stages:
    1. `chunk_started`
    2. `local_annotation_ready`
    3. `embeddings_extracted`
    4. `cluster_assignment`
    5. `segment_committed`
  - Aggregates `pyannote.core.Annotation` into global annotation, stores RTTM under `data/transcripts/rttm/{podcast_slug}_{artefact_key}.rttm`.
  - Returns structured diarization results: `list[DiarizedTurn]` where `DiarizedTurn` includes `start`, `end`, `speaker_id`.
  - Exposes `write_json_transcript(target: Path, transcript: list[TranscriptSegment])` to combine with raw transcript (keeping `speaker_id` placeholders).

#### Artefacts
- Chunk WAV files: `data/audio_chunks/{podcast_slug}/{artefact_key}/chunk_{idx:03d}.wav`.
- Cluster cache: `data/transcripts/{podcast_slug}/{artefact_key}_clusters.pkl`.
- Diarization JSON (optional) mirrors `whisper_diarization.json`.

---

### `mywhisper/assign`

#### Responsibilities
- Combine diarized transcripts with metadata to infer human-readable speaker names.
- Interact with local LLM (via Ollama) or alternative inference providers.
- Persist enriched transcripts including confidence and justification.

#### Key Classes
- `AssignmentConfig`
  - Fields: `ollama_model`, `max_iterations`, `high_confidence_threshold`, `spacy_model`, `sample_utterances_start`, `sample_utterances_end`, `output_dir`.
  - `load_spacy_model()` caches the spaCy pipeline if available.

- `SpeakerProfileBuilder`
  - `build(segments: Sequence[TranscriptSegment]) -> dict[str, SpeakerProfile]` (reuses dataclass from examples).
  - Extracts per-speaker stats, sample quotes.

- `CandidateRoster`
  - Combines metadata sources (`PodcastEpisode`, stored `EpisodeMetadata`, manual roster) and spaCy-extracted names.
  - `compile(additional: Optional[Iterable[str]] = None) -> list[str]`.

- `LLMClient`
  - Abstract base supporting `generate(prompt: str) -> str`.
  - Default implementation `OllamaClient` posts to `http://localhost:11434/api/generate`, matching example script.
  - Optionally support retries and timeout configuration, but keep logic simple per KISS principle.

- `SpeakerInferenceEngine`
  - Orchestrates prompt construction (`build_prompt` akin to `build_inference_prompt`), LLM calls, JSON parsing, and critic pass.
  - Methods:
    - `infer(profiles, roster, context, target_speakers) -> list[SpeakerAssignment]`
    - `critic(assignments) -> dict[str, bool]`
    - `consolidate(prior, new) -> dict[str, SpeakerAssignment]`

- `TranscriptAssigner`
  - Constructor parameters: `podcast`, `config`, `profile_builder`, `roster`, `llm_client`, `logger`.
  - `from_config(podcast, diarized_segments, transcript_path, config)` to auto-load transcript and dependencies.
  - `assign_names(yield_progress: bool = False) -> list[TranscriptSegment] | Generator[PipelineEvent, None, list[TranscriptSegment]]`
    - Steps: `load_diarized_transcript`, `build_profiles`, `prepare_roster`, `run_inference_cycle`, `critic_pass`, optional refinement.
    - Enforces high-confidence rule: assignments below threshold are labelled `"UNKNOWN"`.
    - Persists enriched transcript under `data/transcripts/{podcast_slug}/{artefact_key}_with_names.json`.

#### Artefacts
- Assignment summary persisted as JSON list aligning with `examples/infer_speaker_names.py`.
- Optional CSV export for analytics (`speaker_assignments.csv`) containing speaker_id, name, confidence.

---

### `mywhisper/podcasts`

#### Responsibilities
- Discover and track podcast episodes copied from the Apple Podcasts cache.
- Provide query interfaces by show title, GUID, publication date, or filesystem path.
- Support ingestion from cache (via logic inspired by `examples/copy_podcasts_from_cache.py`).

#### Schema
- SQLite database stored at `data/podcasts/catalog.db`.
- Tables:
  - `episodes(id TEXT PRIMARY KEY, show_title TEXT, episode_title TEXT, author TEXT, guid TEXT, published_at TEXT, cache_path TEXT, audio_path TEXT, duration_sec REAL, metadata_json TEXT)`.
  - `artefacts(artefact_key TEXT PRIMARY KEY, episode_id TEXT, kind TEXT, path TEXT, created_at TEXT)`.
- Indices on `show_title`, `guid`, and `published_at`.

#### Key Classes
- `PodcastCatalog`
  - Constructor accepts `db_path: Path`, ensures schema initialised.
  - `upsert_episode(episode: PodcastEpisode) -> None`, `get_episode(id | guid | path) -> PodcastEpisode | None`.
  - `list_episodes(show_title: Optional[str] = None, since: Optional[datetime] = None) -> Iterable[PodcastEpisode]`.
  - `record_artefact(episode_id: str, kind: str, path: Path, artefact_key: str)`.

- `ApplePodcastsImporter`
  - Initialized with cache roots (`cache_root`, `db_path`), optional `mdls` enrichment flag.
  - `scan() -> Generator[PodcastEpisode, None, None]` replicates extraction from example script but stops at metadata mapping; copy/move handled by caller.
  - `register_in_catalog(catalog: PodcastCatalog, output_dir: Path, move: bool = False) -> Generator[PipelineEvent, None, None]`
    - Events include `episode_discovered`, `audio_copied`, `guid_recorded`.
  - Sanitizes file names and writes GUID sidecars mirroring `extract_episode`.

#### Integration with Pipelines
- `PodcastEpisode.source_path` is the canonical audio file consumed by `PodcastTranscriber` and `DiarizationPipeline`.
- When pipelines produce artefacts, they call `catalog.record_artefact(...)` to maintain traceability.

---

## Inter-Module Workflow
1. **Episode discovery**: `ApplePodcastsImporter` scans cache, registers episodes via `PodcastCatalog`, and stores original audio path without duplication.
2. **Transcription**: `PodcastTranscriber` loads the episode, optionally chunking, transcribes via Whisper, and stores transcript JSON + artefact record.
3. **Diarization**: `DiarizationPipeline` reuses the same `PodcastEpisode` to create RTTM and diarized transcript artefacts.
4. **Speaker assignment**: `TranscriptAssigner` merges transcript and diarization outputs, queries metadata (show title, description) from `PodcastEpisode.metadata`, and persists enriched transcript.

Each step can resume from persisted artefacts thanks to consistent naming and catalog records.

---

## Logging & Metrics
- Pipelines emit structured `PipelineEvent` instances with attributes `stage`, `message`, `payload`.
- Default logger level derived from root `mywhisper` logger; per-module configuration allows enabling verbose logs for targeted debugging.
- Optional integration hook: `PipelineEvent` generators accept a callback `on_event(event: PipelineEvent)` for streaming progress to CLI/GUI.

---

## Minimal External Dependencies
- `pywhispercpp` (Whisper inference).
- `torchaudio`, `torch`.
- `pyannote.audio`, `pyannote.core`, `huggingface_hub`, `tqdm`.
- `numpy`, `scikit-learn`, `joblib`.
- `spaCy` (optional; downgrade gracefully when unavailable).
- `requests` (for Ollama).
- `pydub` (chunk splitting).
- `sqlite3` (standard library).

The package avoids heavy custom error handling; errors bubble to the caller unless they relate to predictable recoverable states (e.g. missing spaCy model, absent HF token).

---

## Open Questions & Assumptions
- Episode metadata (`description`, `host_roster`) may be absent; pipelines must degrade gracefully without these fields.
- LLM prompt templates remain configurable but initial implementation mirrors example behaviour.
- No built-in parallelism; generators enable callers to adopt async/concurrent orchestration externally.

