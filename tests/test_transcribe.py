from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pytest

from mywhisper.models import AudioChunk, PodcastEpisode, TranscriptSegment
import torch

from mywhisper.transcribe import AudioChunker, PodcastTranscriber, TranscriptionConfig


class FakeTensor:
    def __init__(self, data: List[float]):
        self._data = data

    def dim(self) -> int:
        return 1

    def squeeze(self, *_args, **_kwargs) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self):
        import numpy as np

        return np.asarray(self._data, dtype="float32")


class FakeModel:
    def __init__(self) -> None:
        self.calls: List[float] = []

    def transcribe(self, audio_np, language: str, new_segment_callback=None):
        self.calls.append(float(audio_np.sum()))

        class Segment:
            def __init__(self, t0: float, t1: float, text: str) -> None:
                self.t0 = t0
                self.t1 = t1
                self.text = text

        segment = Segment(0.0, 200.0, "Hello world")
        if new_segment_callback:
            new_segment_callback(segment)
        return [segment]


class StubTranscriber(PodcastTranscriber):
    def _transcribe_chunk(self, chunk: AudioChunk, chunk_index: int):
        return (yield from super()._transcribe_chunk(chunk, chunk_index))


class StubChunker:
    def __init__(self):
        self._chunks: Iterable[AudioChunk] = []

    def set_chunks(self, chunks: Iterable[AudioChunk]) -> None:
        self._chunks = list(chunks)

    def iterate_chunks(self, *args, **kwargs):
        return iter(self._chunks)


def build_transcriber(tmp_path: Path) -> StubTranscriber:
    data_root = tmp_path / "data"
    config = TranscriptionConfig(
        model_path=tmp_path / "model.bin",
        data_root=data_root,
        chunk_duration=None,
    )
    podcast = PodcastEpisode(
        episode_id="episode-1",
        show_title="Test Show",
        episode_title="Episode 1",
        source_path=tmp_path / "source.wav",
        metadata={"episode_key": "12345678"},
    )
    chunker = StubChunker()
    transcriber = StubTranscriber(
        podcast=podcast,
        config=config,
        model=FakeModel(),  # type: ignore[arg-type]
        chunker=chunker,  # type: ignore[arg-type]
    )
    return transcriber


def test_transcribe_persists_segments(tmp_path, monkeypatch):
    transcriber = build_transcriber(tmp_path)

    chunk = AudioChunk(
        path=Path("noop"),
        global_start=0.0,
        global_end=2.0,
        tensor=FakeTensor([0.1, 0.2, 0.3]),
        sample_rate=16000,
    )

    transcriber.chunker.set_chunks([chunk])  # type: ignore[operator]

    segments = transcriber.transcribe()
    assert len(segments) == 1
    assert segments[0].text == "Hello world"

    transcript_path = transcriber._last_transcript_path  # type: ignore[attr-defined]
    assert transcript_path.exists()
    data = json.loads(transcript_path.read_text())
    assert data[0]["text"] == "Hello world"


def test_transcribe_generator_yields_events(tmp_path, monkeypatch):
    transcriber = build_transcriber(tmp_path)

    chunk = AudioChunk(
        path=Path("noop"),
        global_start=1.0,
        global_end=3.0,
        tensor=FakeTensor([0.5, 0.6]),
        sample_rate=16000,
    )
    transcriber.chunker.set_chunks([chunk])  # type: ignore[operator]

    generator = transcriber.transcribe(yield_progress=True)
    stages = [event.stage for event in generator]
    assert stages[0] == "start"
    assert "persisted" in stages
    assert "segment_detected" in stages


def test_load_cached_segments(tmp_path, monkeypatch):
    transcriber = build_transcriber(tmp_path)
    episode_key = transcriber.podcast.episode_key
    transcript_path = transcriber.config.transcript_path(transcriber.podcast, episode_key)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "text": "Sample", "speaker_id": "SPEAKER_00"},
            ]
        )
    )

    segments = transcriber.load_cached_segments(episode_key=episode_key)
    assert segments[0].speaker_id == "SPEAKER_00"


def test_transcribe_errors_without_chunks(tmp_path):
    transcriber = build_transcriber(tmp_path)

    with pytest.raises(FileNotFoundError):
        transcriber.load_cached_segments()


def test_transcriber_from_config_uses_factory(monkeypatch, tmp_path):
    config = TranscriptionConfig(
        model_path=tmp_path / "model.bin",
        data_root=tmp_path / "data",
        chunk_duration=None,
    )
    podcast = PodcastEpisode(
        episode_id="episode-2",
        show_title="Test Show",
        episode_title="Episode 2",
        source_path=tmp_path / "source.wav",
        metadata={"episode_key": "23456789"},
    )

    monkeypatch.setattr(
        "mywhisper.transcribe.WhisperModelFactory.create",
        lambda _config: FakeModel(),
    )

    class DummyChunker(AudioChunker):
        def __init__(self, _config: TranscriptionConfig) -> None:
            self.config = _config

        def iterate_chunks(self, *_args, **_kwargs):
            return iter([])

    monkeypatch.setattr("mywhisper.transcribe.AudioChunker", DummyChunker)

    transcriber = PodcastTranscriber.from_config(podcast, config)
    assert isinstance(transcriber, PodcastTranscriber)


def test_audio_chunker_iterate_chunks(tmp_path, monkeypatch):
    config = TranscriptionConfig(
        model_path=tmp_path / "model.bin",
        data_root=tmp_path / "data",
        chunk_duration=0.01,
        target_sample_rate=10,
    )
    chunker = AudioChunker(config)

    waveform = torch.arange(0, 20, dtype=torch.float32).view(1, -1)

    monkeypatch.setattr(
        "mywhisper.transcribe.torchaudio.load",
        lambda _path: (waveform.clone(), 10),
    )

    writes = []

    def fake_write(path: str, data, sr) -> None:
        writes.append((path, len(data)))

    monkeypatch.setattr("mywhisper.transcribe.sf.write", fake_write)

    podcast = PodcastEpisode(
        episode_id="episode-3",
        show_title="Show",
        episode_title="Chunk Episode",
        source_path=Path("dummy.wav"),
        metadata={"episode_key": "34567890"},
    )

    chunks = list(chunker.iterate_chunks(podcast, podcast.episode_key))
    assert len(chunks) >= 1
    assert writes

