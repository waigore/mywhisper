from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from mywhisper.models import PodcastEpisode, PipelineEvent
from mywhisper.vocative import (
    VocativeConfig,
    EpisodeVocativeDetector,
    EXCLUDED_PROPER_NOUNS,
)


class StubCatalog:
    """Stub catalog for testing."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, Path, str]] = []

    def record_artefact(self, episode_id: str, kind: str, path: Path, artefact_key: str) -> None:
        self.records.append((episode_id, kind, path, artefact_key))




def _episode(tmp_path: Path, episode_key: str = "12345678") -> PodcastEpisode:
    return PodcastEpisode(
        episode_id="ep-vocative",
        show_title="Vocative Show",
        episode_title="Vocative Episode",
        source_path=tmp_path / "audio.wav",
        metadata={"episode_key": episode_key},
    )


def test_vocative_config_paths(tmp_path):
    """Test VocativeConfig path methods."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path, "87654321")

    classified_path = config.classified_path(episode)
    assert "87654321" in str(classified_path)
    assert classified_path.name == "87654321_classified.json"

    vocative_path = config.vocative_path(episode)
    assert "87654321" in str(vocative_path)
    assert vocative_path.name == "87654321_vocative.json"

    # Test with explicit episode_key
    classified_path2 = config.classified_path(episode, episode_key="11111111")
    assert "11111111" in str(classified_path2)
    assert classified_path2.name == "11111111_classified.json"


def test_vocative_config_defaults():
    """Test VocativeConfig default values."""
    config = VocativeConfig()
    assert config.spacy_model == "en_core_web_sm"
    assert config.output_subdir == "transcripts"


def test_episode_vocative_detector_initialization(tmp_path):
    """Test EpisodeVocativeDetector initialization."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    catalog = StubCatalog()

    detector = EpisodeVocativeDetector(
        podcast=episode,
        config=config,
        catalog=catalog,
    )

    assert detector.podcast == episode
    assert detector.config == config
    assert detector.catalog == catalog
    assert detector._last_vocative_path is None
    assert detector._nlp is None


def test_detect_vocatives_missing_classified_file(tmp_path):
    """Test that FileNotFoundError is raised when classified file doesn't exist."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    with pytest.raises(FileNotFoundError, match="Classified transcript not found"):
        detector.detect_vocatives()


def test_detect_vocatives_empty_segments(tmp_path):
    """Test detection with empty segments."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(json.dumps([]), encoding="utf-8")

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    vocative_path = detector.detect_vocatives(classified_path=classified_path)
    assert vocative_path.exists()

    payload = json.loads(vocative_path.read_text(encoding="utf-8"))
    assert payload == []


def test_detect_vocatives_empty_text_segments(tmp_path):
    """Test detection with segments that have empty text."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 10.0, "text": "", "speaker_id": "S0"},
                {"start": 10.0, "end": 20.0, "text": "   ", "speaker_id": "S1"},
            ]
        ),
        encoding="utf-8",
    )

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    vocative_path = detector.detect_vocatives(classified_path=classified_path)
    assert vocative_path.exists()

    payload = json.loads(vocative_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["addressed_person_candidates"] == []
    assert payload[1]["addressed_person_candidates"] == []


def test_detect_vocatives_yield_progress(tmp_path):
    """Test that yield_progress=True returns a generator."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 10.0, "text": "Hello, John.", "speaker_id": "S0"},
            ]
        ),
        encoding="utf-8",
    )

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    result = detector.detect_vocatives(classified_path=classified_path, yield_progress=True)
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Consume the generator
    events = []
    try:
        while True:
            event = next(result)
            events.append(event)
    except StopIteration as stop:
        final_path = stop.value
        assert isinstance(final_path, Path)
        assert final_path.exists()

    assert len(events) > 0
    assert all(isinstance(e, PipelineEvent) for e in events)


def test_detect_vocatives_pipeline_events(tmp_path):
    """Test that pipeline yields proper events."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 10.0, "text": "Hello, John.", "speaker_id": "S0"},
            ]
        ),
        encoding="utf-8",
    )

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    pipeline = detector._pipeline(classified_path=classified_path)
    events = []
    try:
        while True:
            event = next(pipeline)
            events.append(event)
    except StopIteration as stop:
        final_path = stop.value
        assert final_path.exists()

    # Check first event is loading
    assert events[0].stage == "vocative"
    assert "Loading" in events[0].message
    assert "classified" in events[0].artefact_paths

    # Check segment processing events
    segment_events = [e for e in events if "segment" in e.payload.get("step", "")]
    assert len(segment_events) > 0

    # Check final event
    final_event = events[-1]
    assert final_event.stage == "vocative"
    assert "Persisted" in final_event.message
    assert "vocative" in final_event.artefact_paths
    assert final_event.elapsed is not None


def test_detect_vocatives_catalog_registration(tmp_path):
    """Test that vocative detections are registered in catalog."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 10.0, "text": "Hello, John.", "speaker_id": "S0"},
            ]
        ),
        encoding="utf-8",
    )

    catalog = StubCatalog()
    detector = EpisodeVocativeDetector(podcast=episode, config=config, catalog=catalog)

    detector.detect_vocatives(classified_path=classified_path)

    assert len(catalog.records) == 1
    assert catalog.records[0][1] == "vocative"
    assert catalog.records[0][0] == episode.episode_id


def test_detect_vocative_in_segment_empty_text():
    """Test _detect_vocative_in_segment with empty text."""
    config = VocativeConfig()
    episode = _episode(Path("/tmp"))
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    result = detector._detect_vocative_in_segment("")
    assert result == []

    result = detector._detect_vocative_in_segment("   ")
    assert result == []


def test_detect_vocative_in_segment_no_candidates(tmp_path, monkeypatch):
    """Test _detect_vocative_in_segment when no proper nouns are found."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return empty list
    monkeypatch.setattr(detector, "_identify_vocatives", lambda text: [])

    result = detector._detect_vocative_in_segment("This is a regular sentence.")
    assert result == []


def test_detect_vocative_in_segment_ner_based_detection(tmp_path, monkeypatch):
    """Test _detect_vocative_in_segment with NER-based detection."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return a vocative candidate
    monkeypatch.setattr(detector, "_identify_vocatives", lambda text: ["John"])
    # Mock finding occurrences and sentence extraction
    monkeypatch.setattr(detector, "_find_all_occurrences", lambda text, name: [6] if name == "John" else [])
    monkeypatch.setattr(detector, "_extract_sentence_with_vocative_at_position", lambda text, voc, pos: "Hello, John.")
    monkeypatch.setattr(detector, "_classify_candidate_with_llm", lambda sent, cand: {"classification": "VOCATIVE", "justification": "Test justification"})

    result = detector._detect_vocative_in_segment("Hello, John. How are you?")
    assert len(result) == 1
    assert result[0]["name"] == "John"
    assert result[0]["classification"] == "VOCATIVE"
    assert result[0]["justification"] == "Test justification"
    assert result[0]["sentence"] == "Hello, John."


def test_detect_vocative_in_segment_ner_no_match(tmp_path, monkeypatch):
    """Test _detect_vocative_in_segment when no vocative is found."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return empty list
    monkeypatch.setattr(detector, "_identify_vocatives", lambda text: [])

    result = detector._detect_vocative_in_segment("I saw John yesterday.")
    assert result == []


def test_extract_person_names_no_nlp(tmp_path, monkeypatch):
    """Test _extract_person_names when spaCy model is not available."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _get_nlp to return None
    monkeypatch.setattr(detector, "_get_nlp", lambda: None)

    result = detector._extract_person_names("Hello, John.")
    assert result == []


def test_extract_person_names_with_person_entities(tmp_path, monkeypatch):
    """Test _extract_person_names extracts PERSON entities."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Create mock spaCy objects
    class MockEnt:
        def __init__(self, text: str, label_: str, start: int, end: int) -> None:
            self.text = text
            self.label_ = label_
            self.start = start
            self.end = end

    class MockToken:
        def __init__(self, text: str, pos_: str, i: int) -> None:
            self.text = text
            self.pos_ = pos_
            self.i = i

    class MockDoc:
        def __init__(self) -> None:
            # "John Smith" spans tokens 1-2, "Paris" spans token 3
            self.ents = [
                MockEnt("John Smith", "PERSON", 1, 3),
                MockEnt("Paris", "GPE", 3, 4),
            ]
            self.tokens = [
                MockToken("Hello", "INTJ", 0),
                MockToken("John", "PROPN", 1),
                MockToken("Smith", "PROPN", 2),
            ]

        def __iter__(self):
            return iter(self.tokens)

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._extract_person_names("Hello, John Smith.")
    assert "John Smith" in result
    assert "Paris" not in result  # GPE is not PERSON


def test_extract_person_names_with_propn_tokens(tmp_path, monkeypatch):
    """Test _extract_person_names extracts PROPN tokens that are within PERSON entities."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    class MockEnt:
        def __init__(self, text: str, label_: str, start: int, end: int) -> None:
            self.text = text
            self.label_ = label_
            self.start = start
            self.end = end

    class MockToken:
        def __init__(self, text: str, pos_: str, i: int) -> None:
            self.text = text
            self.pos_ = pos_
            self.i = i

    class MockDoc:
        def __init__(self) -> None:
            # "Alice" is a PERSON entity spanning token 1
            self.ents = [MockEnt("Alice", "PERSON", 1, 2)]
            self.tokens = [
                MockToken("Hello", "INTJ", 0),
                MockToken("Alice", "PROPN", 1),  # Should be included (within PERSON entity)
                MockToken("bitcoin", "PROPN", 2),  # Should be excluded (lowercase in EXCLUDED_PROPER_NOUNS)
                MockToken("Bitcoin", "PROPN", 3),  # Should be excluded (not in PERSON entity, in EXCLUDED_PROPER_NOUNS)
                MockToken("A", "PROPN", 4),  # Should be excluded (len <= 1)
            ]

        def __iter__(self):
            return iter(self.tokens)

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._extract_person_names("Hello, Alice and Bitcoin.")
    assert "Alice" in result
    assert "bitcoin" not in result
    assert "Bitcoin" not in result
    assert "A" not in result


def test_extract_person_names_excludes_lowercase_propn(tmp_path, monkeypatch):
    """Test _extract_person_names excludes lowercase PROPN tokens."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    class MockEnt:
        def __init__(self, text: str, label_: str, start: int, end: int) -> None:
            self.text = text
            self.label_ = label_
            self.start = start
            self.end = end

    class MockToken:
        def __init__(self, text: str, pos_: str, i: int) -> None:
            self.text = text
            self.pos_ = pos_
            self.i = i

    class MockDoc:
        def __init__(self) -> None:
            # "Hello" is a PERSON entity spanning token 1
            self.ents = [MockEnt("Hello", "PERSON", 1, 2)]
            self.tokens = [
                MockToken("hello", "PROPN", 0),  # Lowercase, should be excluded
                MockToken("Hello", "PROPN", 1),  # Uppercase, should be included (within PERSON entity)
            ]

        def __iter__(self):
            return iter(self.tokens)

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._extract_person_names("hello Hello")
    assert "Hello" in result
    assert "hello" not in result


def test_identify_vocatives_sentence_beginning(tmp_path, monkeypatch):
    """Test _identify_vocatives detects vocative at sentence beginning."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Create mock spaCy objects for "John, how are you?"
    class MockToken:
        def __init__(self, text: str, pos_: str, i: int, is_punct: bool = False) -> None:
            self.text = text
            self.pos_ = pos_
            self.i = i
            self.is_punct = is_punct

    class MockSpan:
        def __init__(self, tokens: list, ents: list) -> None:
            self._tokens = tokens
            self._ents = ents

        def __len__(self) -> int:
            return len(self._tokens)

        def __getitem__(self, idx: int) -> MockToken:
            return self._tokens[idx]

    class MockEnt:
        def __init__(self, text: str, label_: str, start: int, end: int) -> None:
            self.text = text
            self.label_ = label_
            self.start = start
            self.end = end

    class MockDoc:
        def __init__(self) -> None:
            tokens = [
                MockToken("John", "PROPN", 0),
                MockToken(",", "PUNCT", 1, is_punct=True),
                MockToken("how", "VERB", 2),
                MockToken("are", "AUX", 3),
            ]
            self.ents = [MockEnt("John", "PERSON", 0, 1)]
            self._tokens = tokens

        @property
        def sents(self) -> list:
            return [MockSpan(self._tokens, self.ents)]

        def __iter__(self):
            return iter(self._tokens)

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._identify_vocatives("John, how are you?")
    assert result == ["John"]


def test_identify_vocatives_sentence_end(tmp_path, monkeypatch):
    """Test _identify_vocatives detects vocative at sentence end."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    class MockToken:
        def __init__(self, text: str, pos_: str, i: int, is_punct: bool = False) -> None:
            self.text = text
            self.pos_ = pos_
            self.i = i
            self.is_punct = is_punct

    class MockSpan:
        def __init__(self, tokens: list, ents: list) -> None:
            self._tokens = tokens
            self._ents = ents

        def __len__(self) -> int:
            return len(self._tokens)

        def __getitem__(self, idx: int) -> MockToken:
            if idx == -1:
                return self._tokens[-1]
            if idx == -2:
                return self._tokens[-2]
            return self._tokens[idx]

    class MockEnt:
        def __init__(self, text: str, label_: str, start: int, end: int) -> None:
            self.text = text
            self.label_ = label_
            self.start = start
            self.end = end

    class MockDoc:
        def __init__(self) -> None:
            tokens = [
                MockToken("Hello", "INTJ", 0),
                MockToken(",", "PUNCT", 1, is_punct=True),
                MockToken("John", "PROPN", 2),
            ]
            self.ents = [MockEnt("John", "PERSON", 2, 3)]
            self._tokens = tokens

        @property
        def sents(self) -> list:
            return [MockSpan(self._tokens, self.ents)]

        def __iter__(self):
            return iter(self._tokens)

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._identify_vocatives("Hello, John.")
    assert result == ["John"]


def test_identify_vocatives_no_nlp(tmp_path, monkeypatch):
    """Test _identify_vocatives returns empty list when NLP is not available."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    monkeypatch.setattr(detector, "_get_nlp", lambda: None)

    result = detector._identify_vocatives("Hello, John.")
    assert result == []


def test_identify_vocatives_no_match(tmp_path, monkeypatch):
    """Test _identify_vocatives returns empty list when no vocative is found."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    class MockToken:
        def __init__(self, text: str, pos_: str, i: int, is_punct: bool = False) -> None:
            self.text = text
            self.pos_ = pos_
            self.i = i
            self.is_punct = is_punct

    class MockSpan:
        def __init__(self, tokens: list, ents: list) -> None:
            self._tokens = tokens
            self._ents = ents

        def __len__(self) -> int:
            return len(self._tokens)

        def __getitem__(self, idx: int) -> MockToken:
            return self._tokens[idx]

    class MockEnt:
        def __init__(self, text: str, label_: str, start: int, end: int) -> None:
            self.text = text
            self.label_ = label_
            self.start = start
            self.end = end

    class MockDoc:
        def __init__(self) -> None:
            tokens = [
                MockToken("I", "PRON", 0),
                MockToken("saw", "VERB", 1),
                MockToken("John", "PROPN", 2),
            ]
            self.ents = [MockEnt("John", "PERSON", 2, 3)]
            self._tokens = tokens

        @property
        def sents(self) -> list:
            return [MockSpan(self._tokens, self.ents)]

        def __iter__(self):
            return iter(self._tokens)

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._identify_vocatives("I saw John yesterday.")
    assert result == []


def test_get_nlp_successful_load(tmp_path, monkeypatch):
    """Test _get_nlp successfully loads spaCy model."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    mock_nlp = Mock()
    monkeypatch.setattr("spacy.load", lambda model: mock_nlp)

    result = detector._get_nlp()
    assert result == mock_nlp
    assert detector._nlp == mock_nlp

    # Second call should return cached version
    result2 = detector._get_nlp()
    assert result2 == mock_nlp


def test_get_nlp_oserror_handling(tmp_path, monkeypatch):
    """Test _get_nlp handles OSError (model not found)."""
    config = VocativeConfig(spacy_model="en_core_web_sm")
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    def mock_load(model: str):
        raise OSError("Can't find model")

    monkeypatch.setattr("spacy.load", mock_load)

    result = detector._get_nlp()
    assert result is None
    assert detector._nlp is None


def test_get_nlp_general_exception(tmp_path, monkeypatch):
    """Test _get_nlp handles general exceptions."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    def mock_load(model: str):
        raise ValueError("Unexpected error")

    monkeypatch.setattr("spacy.load", mock_load)

    result = detector._get_nlp()
    assert result is None
    assert detector._nlp is None


def test_detect_vocatives_full_pipeline(tmp_path, monkeypatch):
    """Test full pipeline with mocked dependencies."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 10.0, "text": "John, how are you?", "speaker_id": "S0"},
                {"start": 10.0, "end": 20.0, "text": "I'm doing well, thanks.", "speaker_id": "S1"},
                {"start": 20.0, "end": 30.0, "text": "Alice, welcome to the show.", "speaker_id": "S0"},
            ]
        ),
        encoding="utf-8",
    )

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return vocatives based on text
    def mock_identify_vocatives(text: str) -> list[str]:
        if "John, how are you?" in text or "John," in text:
            return ["John"]
        if "Alice, welcome" in text or "Alice," in text:
            return ["Alice"]
        return []

    def mock_find_occurrences(text: str, name: str) -> list[int]:
        if name == "John" and "John" in text:
            # Find first occurrence
            pos = text.find("John")
            return [pos] if pos != -1 else []
        elif name == "Alice" and "Alice" in text:
            pos = text.find("Alice")
            return [pos] if pos != -1 else []
        return []

    def mock_extract_sentence(text: str, voc: str, pos: int) -> str:
        if voc == "John":
            return "John, how are you?"
        elif voc == "Alice":
            return "Alice, welcome to the show."
        return ""

    monkeypatch.setattr(detector, "_identify_vocatives", mock_identify_vocatives)
    monkeypatch.setattr(detector, "_find_all_occurrences", mock_find_occurrences)
    # Mock sentence extraction and LLM classification
    monkeypatch.setattr(detector, "_extract_sentence_with_vocative_at_position", mock_extract_sentence)
    monkeypatch.setattr(detector, "_classify_candidate_with_llm", lambda sent, cand: {"classification": "VOCATIVE" if cand in ("John", "Alice") else "OTHER", "justification": "Test justification"})

    vocative_path = detector.detect_vocatives(classified_path=classified_path)
    assert vocative_path.exists()

    payload = json.loads(vocative_path.read_text(encoding="utf-8"))
    assert len(payload) == 3
    assert payload[0]["addressed_person_candidates"] == [{"name": "John", "classification": "VOCATIVE", "justification": "Test justification", "sentence": "John, how are you?"}]
    assert payload[1]["addressed_person_candidates"] == []
    assert payload[2]["addressed_person_candidates"] == [{"name": "Alice", "classification": "VOCATIVE", "justification": "Test justification", "sentence": "Alice, welcome to the show."}]


def test_detect_vocatives_non_dict_records(tmp_path):
    """Test that non-dict records in classified JSON are filtered out."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 10.0, "text": "Hello.", "speaker_id": "S0"},
                "not a dict",
                123,
                None,
                {"start": 10.0, "end": 20.0, "text": "World.", "speaker_id": "S1"},
            ]
        ),
        encoding="utf-8",
    )

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    vocative_path = detector.detect_vocatives(classified_path=classified_path)
    assert vocative_path.exists()

    payload = json.loads(vocative_path.read_text(encoding="utf-8"))
    # Should only have 2 dict records
    assert len(payload) == 2
    assert all(isinstance(seg, dict) for seg in payload)


def test_detect_vocatives_preserves_segment_fields(tmp_path):
    """Test that all original segment fields are preserved in output."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": "Hello.",
                    "speaker_id": "S0",
                    "speaker_name": "Host",
                    "theme": "Introduction",
                    "custom_field": "custom_value",
                },
            ]
        ),
        encoding="utf-8",
    )

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    vocative_path = detector.detect_vocatives(classified_path=classified_path)
    assert vocative_path.exists()

    payload = json.loads(vocative_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    segment = payload[0]
    assert segment["start"] == 0.0
    assert segment["end"] == 10.0
    assert segment["text"] == "Hello."
    assert segment["speaker_id"] == "S0"
    assert segment["speaker_name"] == "Host"
    assert segment["theme"] == "Introduction"
    assert segment["custom_field"] == "custom_value"
    assert "addressed_person_candidates" in segment
    assert isinstance(segment["addressed_person_candidates"], list)


def test_detect_vocatives_last_path_stored(tmp_path):
    """Test that _last_vocative_path is stored after detection."""
    config = VocativeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    classified_path = config.classified_path(episode)
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 10.0, "text": "Hello.", "speaker_id": "S0"},
            ]
        ),
        encoding="utf-8",
    )

    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    assert detector._last_vocative_path is None

    vocative_path = detector.detect_vocatives(classified_path=classified_path)
    assert detector._last_vocative_path == vocative_path


def test_extract_sentence_with_vocative_success(tmp_path, monkeypatch):
    """Test _extract_sentence_with_vocative successfully extracts sentence."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    class MockToken:
        def __init__(self, text: str) -> None:
            self.text = text

    class MockSpan:
        def __init__(self, text: str) -> None:
            self.text = text

    class MockDoc:
        def __init__(self) -> None:
            self.sents = [
                MockSpan("Hello, John. How are you?"),
                MockSpan("I'm doing well."),
            ]

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._extract_sentence_with_vocative("Hello, John. How are you? I'm doing well.", "John")
    assert result == "Hello, John. How are you?"


def test_extract_sentence_with_vocative_not_found(tmp_path, monkeypatch):
    """Test _extract_sentence_with_vocative returns None when vocative not found."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    class MockSpan:
        def __init__(self, text: str) -> None:
            self.text = text

    class MockDoc:
        def __init__(self) -> None:
            self.sents = [MockSpan("Hello, how are you?")]

    class MockNLP:
        def __call__(self, text: str) -> MockDoc:
            return MockDoc()

    monkeypatch.setattr(detector, "_get_nlp", lambda: MockNLP())

    result = detector._extract_sentence_with_vocative("Hello, how are you?", "John")
    assert result is None


def test_extract_sentence_with_vocative_empty_input(tmp_path):
    """Test _extract_sentence_with_vocative with empty input."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    result = detector._extract_sentence_with_vocative("", "John")
    assert result is None

    result = detector._extract_sentence_with_vocative("Hello", "")
    assert result is None


def test_classify_candidate_with_llm_vocative(tmp_path):
    """Test _classify_candidate_with_llm returns VOCATIVE classification."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    
    # Create mock LLM client
    class MockLLMClient:
        def generate(self, prompt: str) -> str:
            return '{"classification": "VOCATIVE", "justification": "Direct address to Josh"}'
    
    client = MockLLMClient()
    detector = EpisodeVocativeDetector(podcast=episode, config=config, client=client)

    result = detector._classify_candidate_with_llm("Josh, what do you think?", "Josh")
    assert result == {"classification": "VOCATIVE", "justification": "Direct address to Josh"}


def test_classify_candidate_with_llm_other(tmp_path):
    """Test _classify_candidate_with_llm returns OTHER classification."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    
    # Create mock LLM client
    class MockLLMClient:
        def generate(self, prompt: str) -> str:
            return '{"classification": "OTHER", "justification": "EGS is part of a list, not a direct address"}'
    
    client = MockLLMClient()
    detector = EpisodeVocativeDetector(podcast=episode, config=config, client=client)

    result = detector._classify_candidate_with_llm(
        "We uploaded a video, a podcast, EGS, to Gemini",
        "EGS"
    )
    assert result == {"classification": "OTHER", "justification": "EGS is part of a list, not a direct address"}


def test_classify_candidate_with_llm_json_decode_error(tmp_path):
    """Test _classify_candidate_with_llm handles JSON decode errors."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    
    # Create mock LLM client that returns invalid JSON
    class MockLLMClient:
        def generate(self, prompt: str) -> str:
            return "Invalid JSON response"
    
    client = MockLLMClient()
    detector = EpisodeVocativeDetector(podcast=episode, config=config, client=client)

    result = detector._classify_candidate_with_llm("Josh, what do you think?", "Josh")
    assert result["classification"] == "UNKNOWN"
    assert "JSON decode error" in result["justification"]


def test_classify_candidate_with_llm_exception(tmp_path):
    """Test _classify_candidate_with_llm handles exceptions."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    
    # Create mock LLM client that raises exception
    class MockLLMClient:
        def generate(self, prompt: str) -> str:
            raise Exception("LLM service unavailable")
    
    client = MockLLMClient()
    detector = EpisodeVocativeDetector(podcast=episode, config=config, client=client)

    result = detector._classify_candidate_with_llm("Josh, what do you think?", "Josh")
    assert result["classification"] == "UNKNOWN"
    assert "LLM call failed" in result["justification"]


def test_classify_candidate_with_llm_request_exception(tmp_path):
    """Test _classify_candidate_with_llm handles HTTP/connection errors gracefully."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    
    import requests
    
    # Create mock LLM client that raises HTTPError (like when Ollama is not running)
    class MockLLMClient:
        def generate(self, prompt: str) -> str:
            raise requests.exceptions.HTTPError("404 Client Error: Not Found")
    
    client = MockLLMClient()
    detector = EpisodeVocativeDetector(podcast=episode, config=config, client=client)

    result = detector._classify_candidate_with_llm("Josh, what do you think?", "Josh")
    assert result["classification"] == "UNKNOWN"
    assert "LLM service unavailable" in result["justification"]


def test_classify_candidate_with_llm_invalid_classification_value(tmp_path):
    """Test _classify_candidate_with_llm handles invalid classification values."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    
    # Create mock LLM client that returns invalid classification
    class MockLLMClient:
        def generate(self, prompt: str) -> str:
            return '{"classification": "INVALID"}'
    
    client = MockLLMClient()
    detector = EpisodeVocativeDetector(podcast=episode, config=config, client=client)

    result = detector._classify_candidate_with_llm("Josh, what do you think?", "Josh")
    assert result["classification"] == "UNKNOWN"
    assert "Invalid response format" in result["justification"]


def test_classify_candidate_with_llm_empty_input(tmp_path):
    """Test _classify_candidate_with_llm with empty input."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    result = detector._classify_candidate_with_llm("", "Josh")
    assert result["classification"] == "UNKNOWN"
    assert "Empty input provided" in result["justification"]

    result = detector._classify_candidate_with_llm("Sentence", "")
    assert result["classification"] == "UNKNOWN"
    assert "Empty input provided" in result["justification"]


def test_detect_vocative_in_segment_sentence_extraction_failure(tmp_path, monkeypatch):
    """Test _detect_vocative_in_segment when sentence extraction fails."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return a vocative candidate
    monkeypatch.setattr(detector, "_identify_vocatives", lambda text: ["John"])
    # Mock finding occurrences and sentence extraction to return None
    monkeypatch.setattr(detector, "_find_all_occurrences", lambda text, name: [6] if name == "John" else [])
    monkeypatch.setattr(detector, "_extract_sentence_with_vocative_at_position", lambda text, voc, pos: None)

    result = detector._detect_vocative_in_segment("Hello, John. How are you?")
    assert len(result) == 1
    assert result[0]["name"] == "John"
    assert result[0]["classification"] == "UNKNOWN"
    assert result[0]["justification"] == "Sentence extraction failed, cannot classify without context"
    assert result[0]["sentence"] == ""


def test_detect_vocative_in_segment_multiple_candidates(tmp_path, monkeypatch):
    """Test _detect_vocative_in_segment with multiple different candidates."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return multiple candidates
    monkeypatch.setattr(detector, "_identify_vocatives", lambda text: ["John", "Alice"])
    # Mock finding occurrences for each name
    def mock_find_occurrences(text, name):
        if name == "John":
            return [0]  # "John" at position 0
        elif name == "Alice":
            return [25]  # "Alice" at position 25
        return []
    monkeypatch.setattr(detector, "_find_all_occurrences", mock_find_occurrences)
    # Mock sentence extraction and LLM classification
    def mock_extract_sentence(text, voc, pos):
        if voc == "John":
            return "John, how are you?"
        elif voc == "Alice":
            return "Alice, welcome."
        return None
    monkeypatch.setattr(detector, "_extract_sentence_with_vocative_at_position", mock_extract_sentence)
    def mock_classify(sent, cand):
        classification = "VOCATIVE" if cand == "John" else "OTHER"
        return {"classification": classification, "justification": f"Test justification for {cand}"}
    monkeypatch.setattr(detector, "_classify_candidate_with_llm", mock_classify)

    result = detector._detect_vocative_in_segment("John, how are you? Alice, welcome.")
    assert len(result) == 2
    assert result[0]["name"] == "John"
    assert result[0]["classification"] == "VOCATIVE"
    assert result[0]["justification"] == "Test justification for John"
    assert result[0]["sentence"] == "John, how are you?"
    assert result[1]["name"] == "Alice"
    assert result[1]["classification"] == "OTHER"
    assert result[1]["justification"] == "Test justification for Alice"
    assert result[1]["sentence"] == "Alice, welcome."


def test_detect_vocative_in_segment_multiple_mentions_same_name(tmp_path, monkeypatch):
    """Test _detect_vocative_in_segment with multiple occurrences of the same name."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return a single unique name
    monkeypatch.setattr(detector, "_identify_vocatives", lambda text: ["Josh"])
    # Mock finding multiple occurrences of the same name
    monkeypatch.setattr(detector, "_find_all_occurrences", lambda text, name: [0, 28] if name == "Josh" else [])
    # Mock sentence extraction to return different sentences for each position
    def mock_extract_sentence(text, voc, pos):
        if pos == 0:
            return "Josh, when I say this."
        elif pos == 28:
            return "Do you see that, Josh?"
        return None
    monkeypatch.setattr(detector, "_extract_sentence_with_vocative_at_position", mock_extract_sentence)
    # Mock LLM classification - both could be VOCATIVE or different classifications
    def mock_classify(sent, cand):
        if "when I say" in sent:
            return {"classification": "VOCATIVE", "justification": "Direct address at sentence beginning"}
        elif "Do you see" in sent:
            return {"classification": "VOCATIVE", "justification": "Direct address at sentence end"}
        return {"classification": "OTHER", "justification": "Other usage"}
    monkeypatch.setattr(detector, "_classify_candidate_with_llm", mock_classify)

    result = detector._detect_vocative_in_segment("Josh, when I say this. Do you see that, Josh?")
    # Should have 2 entries for "Josh", each with its own classification
    assert len(result) == 2
    assert all(r["name"] == "Josh" for r in result)
    assert result[0]["classification"] == "VOCATIVE"
    assert result[0]["sentence"] == "Josh, when I say this."
    assert result[1]["classification"] == "VOCATIVE"
    assert result[1]["sentence"] == "Do you see that, Josh?"
    # Each should have its own justification based on context
    assert "beginning" in result[0]["justification"]
    assert "end" in result[1]["justification"]


def test_detect_vocative_in_segment_sentence_field_present(tmp_path, monkeypatch):
    """Test that sentence field is present in all candidate objects."""
    config = VocativeConfig()
    episode = _episode(tmp_path)
    detector = EpisodeVocativeDetector(podcast=episode, config=config)

    # Mock _identify_vocatives to return a vocative candidate
    monkeypatch.setattr(detector, "_identify_vocatives", lambda text: ["John"])
    monkeypatch.setattr(detector, "_find_all_occurrences", lambda text, name: [6] if name == "John" else [])
    monkeypatch.setattr(detector, "_extract_sentence_with_vocative_at_position", lambda text, voc, pos: "Hello, John.")
    monkeypatch.setattr(detector, "_classify_candidate_with_llm", lambda sent, cand: {"classification": "VOCATIVE", "justification": "Test"})

    result = detector._detect_vocative_in_segment("Hello, John. How are you?")
    assert len(result) == 1
    assert "sentence" in result[0]
    assert result[0]["sentence"] == "Hello, John."

