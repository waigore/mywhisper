from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mywhisper.diarize import (
    AgglomerativeClustering,
    ChunkScheduler,
    DiarizationConfig,
    SpeakerClusterer,
)
from mywhisper.models import PodcastEpisode


class DummyAgglomerative(AgglomerativeClustering):
    def __init__(self, n_clusters: int, metric: str, linkage: str) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.labels_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> None:
        labels = [idx % self.n_clusters for idx in range(len(embeddings))]
        self.labels_ = np.asarray(labels)


def test_speaker_clusterer_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("mywhisper.diarize.AgglomerativeClustering", DummyAgglomerative)

    clusterer = SpeakerClusterer(tmp_path)
    cluster_path = tmp_path / "clusters.pkl"
    clusterer.prepare(cluster_path)

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.1],
        ]
    )
    clusterer.fit_reference(embeddings, num_speakers=2)

    assignments = clusterer.assign(np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert assignments == [0, 1]

    clusterer.save()
    assert cluster_path.exists()

    # Clear state and ensure load restores it.
    clusterer.centroids = None
    clusterer.load()
    assert clusterer.centroids is not None


def test_chunk_scheduler_creates_chunks(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    config = DiarizationConfig(
        chunk_minutes=0.001,
        overlap_seconds=0.0,
        data_root=data_root,
    )
    scheduler = ChunkScheduler(config)

    podcast = PodcastEpisode(
        episode_id="episode-42",
        show_title="Test Show",
        episode_title="Sample Episode",
        source_path=Path("audio.wav"),
        metadata={"episode_key": "45678901"},
    )

    waveform = torch.arange(0, 320, dtype=torch.float32).view(1, -1)

    monkeypatch.setattr(
        "mywhisper.diarize.torchaudio.load",
        lambda _path: (waveform.clone(), 16000),
    )

    writes: list[Path] = []

    def fake_write(path: str, _data, _sr) -> None:
        writes.append(Path(path))

    monkeypatch.setattr("mywhisper.diarize.sf.write", fake_write)

    chunks = list(scheduler.schedule(podcast, podcast.episode_key))
    assert len(chunks) >= 1
    assert writes

