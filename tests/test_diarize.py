from __future__ import annotations

from pathlib import Path

import torch

from mywhisper.diarize import DiarizationConfig, WaveformLoader
from mywhisper.models import PodcastEpisode


def test_diarization_config_creates_expected_paths(tmp_path):
    config = DiarizationConfig(data_root=tmp_path)
    episode = PodcastEpisode(
        episode_id="episode-42",
        show_title="Test Show",
        episode_title="Sample Episode",
        source_path=Path("audio.wav"),
    )

    paths = config.artefact_paths(episode)

    assert paths["rttm_path"].name.endswith(f"{episode.artefact_slug()}_{episode.episode_key}.rttm")
    assert paths["rttm_path"].parent.exists()
    assert paths["json_path"].parent.exists()


def test_waveform_loader_resamples_and_downmixes(monkeypatch):
    loader = WaveformLoader(target_sample_rate=8000)

    waveform = torch.vstack(
        (
            torch.arange(0, 8, dtype=torch.float32),
            torch.arange(8, 16, dtype=torch.float32),
        )
    )

    monkeypatch.setattr(
        "mywhisper.diarize.torchaudio.load",
        lambda _path: (waveform.clone(), 16000),
    )

    called = {}

    class DummyResample:
        def __init__(self, orig_freq, new_freq):
            called["orig_freq"] = orig_freq
            called["new_freq"] = new_freq

        def __call__(self, data):
            called["was_called"] = True
            return data.clone()

    monkeypatch.setattr("mywhisper.diarize.torchaudio.transforms.Resample", DummyResample)

    payload = loader.load(Path("fake.wav"))

    assert payload["sample_rate"] == 8000
    assert payload["waveform"].shape == (1, waveform.shape[1])
    assert called["was_called"]
    assert called["orig_freq"] == 16000
    assert called["new_freq"] == 8000

