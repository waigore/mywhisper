# mywhisper

`mywhisper` provides object-oriented pipelines for podcast transcription, diarization, speaker name assignment, and podcast cataloging. The package is built around generator-driven workflows so callers can stream progress, and artefacts are stored under `data/` with traceable identifiers.

## Installation

Install project dependencies (requires Python 3.11+):

```bash
pip install -r requirements.txt  # or use Pipenv as configured in Pipfile
```

Optional components:

- `pywhispercpp`, `torchaudio`, `torch`, `soundfile` for transcription.
- `pyannote.audio`, `huggingface_hub`, `numpy`, `scikit-learn`, `joblib` for diarization.
- `spaCy`, `requests` for speaker assignment.

## Quick Start

```python
from pathlib import Path
from mywhisper.models import PodcastEpisode
from mywhisper.transcribe import PodcastTranscriber, TranscriptionConfig

episode = PodcastEpisode(
    episode_id="guid-123",
    show_title="Sample Show",
    episode_title="Great Conversation",
    source_path=Path("audio.m4a"),
)

config = TranscriptionConfig(model_path=Path("ggml-base.bin"))
transcriber = PodcastTranscriber.from_config(episode, config)
segments = transcriber.transcribe()
```

See `examples/README.md` for end-to-end pipelines that combine `PodcastTranscriber`, `DiarizationPipeline`, and `TranscriptAssigner`.

## Testing & Coverage

Tests enforce 90 % minimum coverage:

```bash
pytest
```

Coverage settings are configured in `pytest.ini`.

## Modules

- `mywhisper.transcribe` — Whisper transcription pipeline.
- `mywhisper.diarize` — PyAnnote-based diarization with clustering.
- `mywhisper.assign` — Speaker inference via LLMs.
- `mywhisper.podcasts` — SQLite-backed catalog of imported episodes.

Generated artefacts live in `data/` by default. Configure logging via `mywhisper.configure_logging`.

