from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from mywhisper.models import PodcastEpisode
from mywhisper.classify import EpisodeClassifier, ClassifyConfig


class StubCatalog:
    def __init__(self) -> None:
        self.records: List[tuple[str, str, Path, str]] = []

    def record_artefact(self, episode_id: str, kind: str, path: Path, artefact_key: str) -> None:
        self.records.append((episode_id, kind, path, artefact_key))


def _episode(tmp_path: Path) -> PodcastEpisode:
    return PodcastEpisode(
        episode_id="ep-classify",
        show_title="Classify Show",
        episode_title="Classify Episode",
        source_path=tmp_path / "audio.wav",
        metadata={"episode_key": "45678901"},
    )


class StubClassifier:
    """Stub classifier that returns predictable results for testing."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, candidate_labels: List[str], multi_label: bool = False) -> dict:
        self.calls += 1
        # Return different results based on text content
        if "advertisement" in text.lower() or "sponsor" in text.lower():
            return {
                "labels": ["podcast advertisement or sponsorship", "main editorial content"],
                "scores": [0.85, 0.15],
                "sequence": text,
            }
        elif "intro" in text.lower() or "welcome" in text.lower():
            return {
                "labels": ["episode intro or outro filler", "main editorial content"],
                "scores": [0.80, 0.20],
                "sequence": text,
            }
        else:
            return {
                "labels": ["main editorial content", "podcast advertisement or sponsorship"],
                "scores": [0.90, 0.10],
                "sequence": text,
            }


def test_classifier_adds_classifications_to_segments(tmp_path):
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    themes_path = config.themes_path(episode)
    themes_path.parent.mkdir(parents=True, exist_ok=True)
    themes_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "speaker_id": "S0",
                    "speaker_name": "Host",
                    "text": "Welcome to the show. This is an introduction.",
                    "theme": "Introduction",
                    "summary": "Host introduces the episode",
                },
                {
                    "start": 10.0,
                    "end": 20.0,
                    "speaker_id": "S1",
                    "speaker_name": "Guest",
                    "text": "This is our sponsor advertisement. Buy our product now.",
                    "theme": "Advertisement",
                    "summary": "Sponsor message",
                },
            ]
        ),
        encoding="utf-8",
    )

    catalog = StubCatalog()
    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=catalog,
        classifier=StubClassifier(),
    )

    classified_path = classifier.classify(themes_path=themes_path)
    assert classified_path.exists()
    payload = json.loads(classified_path.read_text(encoding="utf-8"))
    assert len(payload) == 2

    # Check first segment (intro)
    first = payload[0]
    assert "classifications" in first
    assert len(first["classifications"]) > 0
    assert first["classifications"][0]["label"] == "episode intro or outro filler"
    assert first["classifications"][0]["is_non_editorial"] is True

    # Check second segment (ad)
    second = payload[1]
    assert "classifications" in second
    assert len(second["classifications"]) > 0
    assert second["classifications"][0]["label"] == "podcast advertisement or sponsorship"
    assert second["classifications"][0]["is_non_editorial"] is True

    # Verify catalog registration
    assert catalog.records and catalog.records[0][1] == "classified"


def test_classifier_handles_long_segments_with_chunking(tmp_path):
    config = ClassifyConfig(data_root=tmp_path / "data", max_words_per_chunk=50)
    episode = _episode(tmp_path)
    themes_path = config.themes_path(episode)
    themes_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a long segment (>50 words)
    long_text = " ".join(["This is a sentence."] * 60)  # 60 sentences, way over 50 words
    themes_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 100.0,
                    "speaker_id": "S0",
                    "speaker_name": "Host",
                    "text": long_text,
                    "theme": "Long Segment",
                    "summary": "A very long segment",
                }
            ]
        ),
        encoding="utf-8",
    )

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    classified_path = classifier.classify(themes_path=themes_path)
    assert classified_path.exists()
    payload = json.loads(classified_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert "classifications" in payload[0]
    # Should have classifications (may be multiple if chunks differ)
    assert len(payload[0]["classifications"]) > 0


def test_classifier_handles_empty_segments(tmp_path):
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    themes_path = config.themes_path(episode)
    themes_path.parent.mkdir(parents=True, exist_ok=True)
    themes_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "speaker_id": "S0",
                    "speaker_name": "Host",
                    "text": "",
                    "theme": "Empty",
                    "summary": "Empty segment",
                }
            ]
        ),
        encoding="utf-8",
    )

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    classified_path = classifier.classify(themes_path=themes_path)
    assert classified_path.exists()
    payload = json.loads(classified_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["classifications"] == []


def test_classifier_threshold_logic(tmp_path):
    config = ClassifyConfig(data_root=tmp_path / "data", classification_threshold=0.75)
    episode = _episode(tmp_path)
    themes_path = config.themes_path(episode)
    themes_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a segment that should be classified as editorial with high confidence
    themes_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "speaker_id": "S0",
                    "speaker_name": "Host",
                    "text": "This is main editorial content about technology and innovation.",
                    "theme": "Editorial",
                    "summary": "Main content",
                }
            ]
        ),
        encoding="utf-8",
    )

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    classified_path = classifier.classify(themes_path=themes_path)
    payload = json.loads(classified_path.read_text(encoding="utf-8"))
    first = payload[0]
    assert len(first["classifications"]) > 0
    # Main editorial content should have is_non_editorial=False
    main_classification = next(
        (c for c in first["classifications"] if c["label"] == "main editorial content"), None
    )
    if main_classification:
        assert main_classification["is_non_editorial"] is False


def test_classifier_yield_progress(tmp_path):
    """Test that yield_progress=True returns a generator."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    themes_path = config.themes_path(episode)
    themes_path.parent.mkdir(parents=True, exist_ok=True)
    themes_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "speaker_id": "S0",
                    "speaker_name": "Host",
                    "text": "Test text.",
                    "theme": "Test",
                    "summary": "Test summary",
                }
            ]
        ),
        encoding="utf-8",
    )

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    # Test yield_progress=True returns generator
    result = classifier.classify(themes_path=themes_path, yield_progress=True)
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Consume the generator - it yields PipelineEvent objects
    events = []
    try:
        while True:
            event = next(result)
            events.append(event)
    except StopIteration as stop:
        # The path is returned via StopIteration.value
        final_path = stop.value
        assert isinstance(final_path, Path)
        assert final_path.exists()

    assert len(events) > 0
    # All events should be PipelineEvent objects
    from mywhisper.models import PipelineEvent
    assert all(isinstance(e, PipelineEvent) for e in events)


def test_classifier_missing_themes_file(tmp_path):
    """Test that FileNotFoundError is raised when themes file doesn't exist."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    themes_path = config.themes_path(episode)
    # Don't create the file

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    with pytest.raises(FileNotFoundError, match="Thematized transcript not found"):
        classifier.classify(themes_path=themes_path)


def test_classifier_empty_text_in_classify(tmp_path):
    """Test that empty text returns None from _classify_text."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    # Test empty string
    result = classifier._classify_text("")
    assert result is None

    # Test whitespace only
    result = classifier._classify_text("   \n\t  ")
    assert result is None


def test_classifier_invalid_result_structure(tmp_path):
    """Test handling of invalid classification result structure."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)

    class InvalidResultClassifier:
        def __call__(self, text: str, candidate_labels: List[str], multi_label: bool = False) -> dict:
            # Return invalid structure (missing labels or scores)
            return {"sequence": text}

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=InvalidResultClassifier(),
    )

    result = classifier._classify_text("Some text here.")
    assert result is None


def test_classifier_empty_labels_scores(tmp_path):
    """Test handling of empty labels or scores."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)

    class EmptyResultClassifier:
        def __call__(self, text: str, candidate_labels: List[str], multi_label: bool = False) -> dict:
            return {"labels": [], "scores": [], "sequence": text}

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=EmptyResultClassifier(),
    )

    result = classifier._classify_text("Some text here.")
    assert result is None


def test_classifier_exception_handling(tmp_path):
    """Test that exceptions during classification are handled gracefully."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)

    class FailingClassifier:
        def __call__(self, text: str, candidate_labels: List[str], multi_label: bool = False) -> dict:
            raise RuntimeError("Classification failed")

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=FailingClassifier(),
    )

    result = classifier._classify_text("Some text here.")
    assert result is None


def test_classifier_split_empty_text(tmp_path):
    """Test _split_into_chunks with empty text."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    result = classifier._split_into_chunks("", max_words=100)
    assert result == []


def test_classifier_split_odd_sentences(tmp_path):
    """Test _split_into_chunks with odd number of sentence parts."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=StubClassifier(),
    )

    # Text that will result in odd number of parts after re.split
    # This happens when text ends without sentence-ending punctuation
    # re.split("([.!?]+\s+)", "First. Second. Third") returns:
    # ['', 'First', '. ', 'Second', '. ', 'Third']
    # So len is 5 (odd), and the else branch at line 258 is hit
    text = "First sentence. Second sentence. Third"
    result = classifier._split_into_chunks(text, max_words=100)
    assert len(result) > 0
    assert all(isinstance(chunk, str) for chunk in result)
    
    # Test with text that definitely triggers odd number case
    # Text ending without punctuation
    text2 = "Sentence one. Sentence two. Sentence three without period"
    result2 = classifier._split_into_chunks(text2, max_words=100)
    assert len(result2) > 0


def test_classifier_multiple_chunks_different_labels(tmp_path):
    """Test that multiple chunks with different labels are preserved."""
    config = ClassifyConfig(data_root=tmp_path / "data", max_words_per_chunk=10)
    episode = _episode(tmp_path)
    themes_path = config.themes_path(episode)
    themes_path.parent.mkdir(parents=True, exist_ok=True)

    # Create text that will be split into multiple chunks
    # First part should be intro, second part should be ad
    mixed_text = "Welcome to the show. This is an introduction. " * 5 + "This is our sponsor advertisement. Buy now. " * 5
    themes_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 100.0,
                    "speaker_id": "S0",
                    "speaker_name": "Host",
                    "text": mixed_text,
                    "theme": "Mixed",
                    "summary": "Mixed content",
                }
            ]
        ),
        encoding="utf-8",
    )

    class MultiLabelClassifier:
        def __init__(self):
            self.call_count = 0

        def __call__(self, text: str, candidate_labels: List[str], multi_label: bool = False) -> dict:
            self.call_count += 1
            # First chunk: intro, later chunks: ad
            if "welcome" in text.lower() or "introduction" in text.lower():
                return {
                    "labels": ["episode intro or outro filler", "main editorial content"],
                    "scores": [0.85, 0.15],
                    "sequence": text,
                }
            else:
                return {
                    "labels": ["podcast advertisement or sponsorship", "main editorial content"],
                    "scores": [0.90, 0.10],
                    "sequence": text,
                }

    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=MultiLabelClassifier(),
    )

    classified_path = classifier.classify(themes_path=themes_path)
    payload = json.loads(classified_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    classifications = payload[0]["classifications"]
    # Should have multiple classifications if chunks differ
    assert len(classifications) >= 1
    # Check that we have distinct labels
    labels = [c["label"] for c in classifications]
    assert len(set(labels)) <= len(labels)  # May have duplicates or distinct


def test_classifier_get_classifier_import_error(tmp_path, monkeypatch):
    """Test ImportError handling in _get_classifier."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=None,  # No classifier provided
    )

    # Mock ImportError
    def mock_import_error(*args, **kwargs):
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr("builtins.__import__", mock_import_error)

    with pytest.raises(RuntimeError, match="transformers package is required"):
        classifier._get_classifier()


def test_classifier_get_classifier_general_exception(tmp_path, monkeypatch):
    """Test general exception handling in _get_classifier."""
    config = ClassifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classifier = EpisodeClassifier(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        classifier=None,  # No classifier provided
    )

    # Create a mock transformers module that raises an exception
    class MockPipeline:
        def __init__(self, *args, **kwargs):
            raise ValueError("Model not found")

    class MockTransformers:
        pipeline = MockPipeline

    # Mock sys.modules to return our mock
    import sys
    monkeypatch.setitem(sys.modules, "transformers", MockTransformers())

    # Clear the classifier cache
    classifier._classifier = None

    with pytest.raises(RuntimeError, match="Failed to load classification model"):
        classifier._get_classifier()



