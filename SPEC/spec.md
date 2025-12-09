## MyWhisper Specification

### Purpose
`mywhisper` delivers reproducible podcast-processing pipelines (transcribe → diarize → prettify → thematize → classify → vocative → assign) driven by a cached Apple Podcasts catalog. Every stage reuses the original cache file, keeps artefacts deterministic, and emits progress through generators for batch or interactive control.

---

### Package Layout
- `mywhisper/__init__.py` bootstrap + exports
- `mywhisper/transcribe.py` Whisper pipelines
- `mywhisper/diarize.py` PyAnnote diarization
- `mywhisper/prettify.py` transcript formatting
- `mywhisper/thematize.py` theme generation
- `mywhisper/classify.py` content classification
- `mywhisper/vocative.py` vocative detection
- `mywhisper/assign.py` speaker naming
- `mywhisper/podcasts.py` Apple Podcasts catalog + importer
- `mywhisper/config.py` shared config helpers
- Tests mirror the package under `tests/`

---

### Core Principles
- **Clean Coding:** All code must adhere to the [Clean Coding Principles](clean_coding_principles.md), which enforce abstraction/delegation patterns, guard clause usage for control flow (max 2 levels of nesting), and module-level imports (no inline imports).
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

### Transcribe Step
- Module: `mywhisper/transcribe.py` exporting `TranscriptionConfig` + `PodcastTranscriber`.
- Responsibilities: normalize cached audio (downmix + resample), invoke Whisper.cpp models, persist transcripts as `{episode_key}_whisper.json`, and optionally reuse cached segments.
- Key classes: `TranscriptionConfig` (paths, language, chunking, device), `WhisperModelFactory`, `AudioChunker`, and `PodcastTranscriber`.
- Configuration: `chunk_duration` defaults to 600.0 seconds (10 minutes); when set, audio is split into chunks of this duration with optional overlap. When `None` or unset, the entire audio is processed as a single chunk.
- Artefacts: chunk WAVs (only when chunking), transcript JSON aligned with example schema.

### Diarize Step
- Module: `mywhisper/diarize.py` exporting `DiarizationConfig` + `DiarizationPipeline`.
- Responsibilities: run PyAnnote end-to-end on the cache file, surface `ProgressHook` updates, write RTTM plus optional diarization JSON.
- Key classes: `DiarizationConfig` (HF token, device, output dirs), `WaveformLoader`, `PyAnnotePipelineFactory`, `DiarizationPipeline`.
- Artefacts: `{episode_key}.rttm`, optional `{episode_key}_diarization.json`.

### Prettify Step
- Module: `mywhisper/prettify.py` exporting `PrettifyConfig` + `TranscriptPrettifier`.
- Process transcript segments through four stages:
  1. **Load segments**: Load transcript segments from assignment file (raw segments without diarization labels).
  2. **Sentence-boundary merging**: Merge consecutive segments that belong to the same sentence (segments that don't end with sentence-ending punctuation `.`, `!`, `?` are merged with following segments until a sentence boundary is found). This merging is text-based only and does not depend on speaker assignments. A failsafe prevents merging when the combined text exceeds 40 words without any punctuation (comma `,`, semicolon `;`, full stop `.`, question mark `?`, exclamation `!`). If the combined word stream exceeds 40 words without punctuation, segments are left as-is. Segments are only merged if they are contiguous (gaps ≤`collapse_gap_seconds`).
  3. **Apply diarization labels**: Apply speaker IDs from diarization results to the merged segments based on time overlap. This happens after sentence merging so that diarization labels are applied to complete sentences rather than partial segments. When all speaker overlaps are below the `min_overlap_threshold` (default 0.3s), the speaker with the largest overlap is assigned and the segment is marked with `indeterminate: true` to indicate uncertain assignment.
  4. **Collapse contiguous segments**: Merge segments with matching speakers, gaps ≤`collapse_gap_seconds`, and within `max_block_characters` limit.
  5. **Split large blocks**: Split blocks that exceed 10 sentences into smaller blocks.
- Format using placeholder speaker identifiers from diarization (e.g., `SPEAKER_0: <text>`), and write two artefacts:
  - `{episode_key}_readable.txt` (human-readable transcript)
  - `{episode_key}_condensed.json` (JSON array of collapsed blocks with fields `start,end,speaker_id,speaker_name,text`, and optionally `indeterminate` when speaker assignment is uncertain)
- Register artefact kinds `readable_transcript` and `condensed_transcript` via `PodcastCatalog.record_artefact(...)` and emit generator-driven `PipelineEvent(stage="prettify", step_name="prettify")` for load/collapse/persist.
- Config knobs: `data_root`, `collapse_gap_seconds` (default 1.5 s), optional `max_block_characters`, output subdir override.

### Assign Step
- Module: `mywhisper/assign.py` exporting `AssignmentConfig` + `TranscriptAssigner`.
- Responsibilities: infer speaker names from vocatives using graph-based bipartite matching and contextual turn-taking inference, persist `{episode_key}_inferred_names.json`.
- Key classes: `AssignmentConfig`, `GraphBasedInference`, `ContextualTurnInference`, `TranscriptAssigner`.
- Behaviour: 
  - Graph-based inference: when a speaker uses a vocative, creates edges from OTHER speakers (next speaker with weight 1.0, previous speaker with weight 0.5) to that vocative, since the speaker using the vocative is addressing someone else. Builds bipartite graph (speakers ↔ vocatives), uses maximum weighted matching to assign vocatives to speakers, calculates confidence as edge weight relative to total weights from speaker. **Important constraint**: A speaker cannot be assigned a name they use as a vocative themselves (when SPEAKER_X addresses "Name" as a vocative, they're talking to someone else, so SPEAKER_X should NOT be assigned "Name"). Invalid edges are excluded from the graph before matching.
  - Contextual turn-taking inference: iterates sequentially through segments, boosts scores for next speaker (1.0) and previous speaker (0.5) when vocative is found, uses softmax normalization for confidence.
  - Outputs separate JSON with graph and context inferences for each speaker, including sentences containing matched vocatives.
- Artefacts: `{episode_key}_inferred_names.json` containing speaker list with graph/context inferences and associated sentences.

### Thematize Step
- Module: `mywhisper/thematize.py` exposing `ThematizeConfig` + `EpisodeThematizer`.
- Ingest the condensed JSON produced by the prettify step. Iterate per speaker-collapsed segment (no overlap, no token windowing). For each segment, prompt a configurable LLM (`llm_model`, default Ollama) for:
  - `"theme"`: short title (≤6 words)
  - `"summary"`: concise description up to 50 words
- Persist `{episode_key}_with_themes.json` as an array mirroring each condensed segment and adding `theme` and `summary`. Do not merge adjacent identical themes; each segment receives its own summary.
- Register artefact kind `with_themes` and emit `PipelineEvent(stage="thematize")` per segment plus final persist event (checkpoint includes `themes_path`).
- On LLM failure or empty transcript, fall back to a single section using a general episode overview.

### Classify Step
- Module: `mywhisper/classify.py` exposing `ClassifyConfig` + `EpisodeClassifier`.
- Ingest the thematized JSON produced by the thematize step. For each segment, use zero-shot classification to identify non-editorial content (ads, sponsorships, promos, intros, outros).
- Process each segment's text:
  - If text ≤ 300 words: classify the entire segment as one chunk
  - If text > 300 words: split into sentence-based chunks of at most 300 words each, classify each chunk individually
- Collect all distinct classifications if chunks differ in top label.
- Persist `{episode_key}_classified.json` as an array mirroring each thematized segment and adding `classifications` field (array of `{"label": str, "score": float, "is_non_editorial": bool}`).
- Register artefact kind `classified` and emit `PipelineEvent(stage="classify")` per segment plus final persist event (checkpoint includes `classified_path`).
- Classification uses `transformers.pipeline("zero-shot-classification")` with model `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` and candidate labels: "podcast advertisement or sponsorship", "promo or call-to-action", "episode intro or outro filler", "main editorial content".

### Vocative Step
- Module: `mywhisper/vocative.py` exposing `VocativeConfig` + `EpisodeVocativeDetector`.
- Ingest the classified JSON produced by the classify step. For each segment, detect direct named addresses (vocatives) using a two-stage process: rule-based detection followed by LLM classification.
- Process each segment's text:
  - **Stage 1 - Rule-based detection:**
    - Parse text with SpaCy to extract sentences, tokens, and NER entities
    - Extract PERSON entities using Named Entity Recognition
    - Apply punctuation-based heuristics to identify vocative candidates:
      - Case 1: Name at sentence beginning, followed by punctuation, then verb
      - Case 2: Name at sentence end, preceded by punctuation
      - Case 3: Name in the middle, surrounded by punctuation (e.g., "..., Josh, ...")
    - Return all identified candidates (may be multiple per segment)
  - **Stage 2 - LLM classification:**
    - For each vocative candidate identified by rule-based detection, find all occurrences of that name within the segment
    - For each occurrence separately:
      - Extract the sentence containing that specific occurrence (only the surrounding sentence, not the full segment)
      - Prompt an LLM to determine whether this occurrence is serving as a vocative (direct address) or some other linguistic function
      - LLM returns classification: "VOCATIVE" (direct address) or "OTHER" (other linguistic function)
    - If LLM is unavailable, returns invalid response, or sentence extraction fails, default classification is "UNKNOWN"
    - All candidates (VOCATIVE, OTHER, and UNKNOWN) are included in the output, with each occurrence of the same name tracked separately
  - Add `addressed_person_candidates` field (array of candidate objects) to each segment. Each candidate object has:
    - `name`: string - the name of the candidate
    - `classification`: string - either "VOCATIVE", "OTHER", or "UNKNOWN"
    - `justification`: string - explanation for the classification decision
    - `sentence`: string - the sentence text that was used for classification
  - Multiple occurrences of the same name within a segment result in multiple separate entries, each with its own classification based on its context
  - Empty array when no candidates are found
- Persist `{episode_key}_vocative.json` as an array mirroring each classified segment and adding `addressed_person_candidates` field.
- Register artefact kind `vocative` and emit `PipelineEvent(stage="vocative")` per segment plus final persist event (checkpoint includes `vocative_path`).
- Uses SpaCy model (default `en_core_web_sm`) for NER and dependency parsing.
- Uses LLM client (default Ollama with `llama3` model) for classification. LLM configuration is exposed via `VocativeConfig` (`llm_model`, `llm_endpoint`).

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
4. **Prettify** diarized segments into readable text blocks and a condensed JSON of collapsed segments.
5. **Thematize** by segment using the condensed JSON, yielding `{episode_key}_with_themes.json`.
6. **Classify** segments using zero-shot classification to identify non-editorial content, yielding `{episode_key}_classified.json`.
7. **Detect vocatives** by segment using SpaCy NER and dependency parsing with punctuation-based heuristics, followed by LLM classification, yielding `{episode_key}_vocative.json` with `addressed_person_candidates` field.
8. **Assign speakers** via graph-based and contextual inference (using vocative data) to infer real names, yielding `{episode_key}_inferred_names.json` with separate graph and context inferences for each speaker.

Each stage resumes from artefacts identified by the episode key and records outputs in the catalog for traceability.

---

## Logging, Metrics, and Events
- All pipelines emit `PipelineEvent(stage, step_name, payload, elapsed)` objects and may forward them through callbacks.
- Modules inherit logging configuration from `mywhisper` unless callers override per class.
- Metrics (elapsed time per stage, artefact paths) remain minimal but standardized within the event payloads.

### Automatic Function Logging
All functions automatically log their inputs and outputs at appropriate log levels:
- **INFO level** for externally callable functions (public methods without underscore prefix)
- **DEBUG level** for internal functions (private methods with underscore prefix like `_method_name`)

**Implementation:**
- Classes automatically inherit logging behavior via `LoggingBase` base class or `LoggingMeta` metaclass from `mywhisper.logging_utils`
- Standalone functions use the `@log_function` decorator from `mywhisper.logging_utils`
- Log level determination:
  - Public methods (no `_` prefix): logged at INFO
  - Private methods (`_` prefix): logged at DEBUG
  - Manual override: decorator parameters can override default log level
- Input/output serialization:
  - Default: summarized format (type, size, key attributes for complex objects)
  - Configurable: per-function configuration via decorator parameters
  - Exceptions are not logged; they propagate normally
- Applies automatically to all classes via metaclass or base class inheritance
- Generator functions log entry/exit but not individual yields
- Async functions are supported
- Properties and classmethods are handled appropriately

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
- Partial pipeline lets the user choose a starting step and ending step from the canonical order (`transcribe → diarize → prettify → thematize → classify → vocative → assign`).
  - Constraint: the starting step must be at or before the current in-progress step recorded in `pipeline_status.current_step` for the episode (when present). If no active `current_step`, any step can be chosen as the start.
  - Constraint: the ending step must be at or after the selected starting step. If start equals end, only that step runs.
  - Artefact prerequisites still apply when skipping steps; validations ensure required artefacts exist (or the user must include the producing step in the selection).
- Persistence adds `pipeline_id` to checkpoints and a `pipeline_status` table tracking overall state per episode.
- Consistency guard: if `pipeline_status.last_completed_step` lacks a matching checkpoint (or later-step checkpoints exist without it), the system warns and restarts from the first missing step.

