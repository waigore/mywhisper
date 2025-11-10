from __future__ import annotations

from pathlib import Path
from typing import Optional


EXAMPLES_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = EXAMPLES_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR: Path = DATA_DIR / "transcripts"
AUDIO_DIR: Path = DATA_DIR / "podcasts"
AUDIO_CHUNKS_DIR: Path = DATA_DIR / "audio_chunks"
MODELS_DIR: Path = Path("/opt/redacted/models")

# ---------------------------------------------------------------------------
# Transcription configuration
# ---------------------------------------------------------------------------
MODEL_PATH: Path = MODELS_DIR / "mock-transcriber-q5.bin"
AUDIO_SOURCE_PATH: Path = "/opt/redacted/audio/mock_podcast_episode.mp3"
WHISPER_TRANSCRIPT_PATH: Path = TRANSCRIPTS_DIR / "whisper_transcript.json"
EXTRACTED_AUDIO_PATH: Path = AUDIO_CHUNKS_DIR / "whisper_chunk.wav"

TRANSCRIBE_START_SEC: int = 0
TRANSCRIBE_DURATION_SEC: int = 100
TRANSCRIBE_TARGET_SAMPLE_RATE: int = 16000
WHISPER_TIME_FACTOR: float = 100.0  # Whisper.cpp timestamps are in centiseconds per documentation
TRANSCRIBE_COMPLETE_FILE: bool = True

# ---------------------------------------------------------------------------
# Speaker assignment configuration
# ---------------------------------------------------------------------------
ASSIGN_RTTM_PATH: Path = TRANSCRIPTS_DIR / "rttm" / "full_diarization.rttm"
ASSIGN_TRANSCRIPT_PATH: Path = WHISPER_TRANSCRIPT_PATH
ASSIGN_OUTPUT_PATH: Path = TRANSCRIPTS_DIR / "whisper_diarization.json"
ASSIGN_TRANSCRIPT_TIME_FACTOR: float = 1.0
ASSIGN_UNKNOWN_LABEL: str = "UNKNOWN"

# ---------------------------------------------------------------------------
# Diarization configuration
# ---------------------------------------------------------------------------
HF_TOKEN: str = "hf_fakeToken1234567890"
DIARIZE_AUDIO_PATH: Path = AUDIO_SOURCE_PATH
DIARIZE_CHUNK_MINUTES: int = 10
DIARIZE_OVERLAP_SECONDS: int = 30
DIARIZE_NUM_SPEAKERS: Optional[int] = None  # Set None if unknown
DIARIZE_CHUNK_DIR: Path = AUDIO_CHUNKS_DIR
DIARIZE_RTTM_DIR: Path = TRANSCRIPTS_DIR / "rttm"
DIARIZE_CLUSTER_PKL: Path = TRANSCRIPTS_DIR / "speaker_clusters.pkl"
DIARIZE_OUTPUT_RTTM: Path = DIARIZE_RTTM_DIR / "full_diarization.rttm"

# ---------------------------------------------------------------------------
# Speaker name inference configuration
# ---------------------------------------------------------------------------
ASSIGNED_TRANSCRIPT_PATH: Path = TRANSCRIPTS_DIR / "whisper_diarization.json"
#OLLAMA_MODEL: str = "mock.provider/Fiction-1B-Instruct:Q4"
OLLAMA_MODEL: str = "mock.provider/Fiction-4B-Instruct:Q4"
SPACY_MODEL: str = "mock_nlp_tiny"
TRANSCRIPT_SAMPLE_UTTERANCES_START: int = 50
TRANSCRIPT_SAMPLE_UTTERANCES_END: int = 50
INFER_MAX_ITERATIONS: int = 3
HIGH_CONFIDENCE_THRESHOLD: float = 0.8
CRITIC_STRICTNESS: float = 0.6
INFER_START_OFFSET_SEC: float = 0.0  # Update if you processed a later chunk in the podcast
MAX_TRANSCRIPT_TOKENS: int = 3000
PODCAST_EPISODE_NAME: str = "Sample Show - 2024-01-01 - Placeholder Episode Title"
PODCAST_EPISODE_DESCRIPTION: str = """
This is a fictitious summary used for configuration testing. The episode discusses hypothetical market trends, fictional analyst insights, and imagined scenarios to validate the processing pipeline. All references are placeholders and do not reflect actual events or individuals.
"""


