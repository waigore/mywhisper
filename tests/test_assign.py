from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from mywhisper.assign import (
    AssignmentConfig,
    CandidateRoster,
    SpeakerInferenceEngine,
    TranscriptAssigner,
)
from mywhisper.models import (
    PodcastEpisode,
    SpeakerAssignment,
    SpeakerProfile,
    TranscriptSegment,
)


class StubEngine(SpeakerInferenceEngine):
    def __init__(self, assignments: Sequence[SpeakerAssignment]) -> None:
        config = AssignmentConfig()
        super().__init__(config, client=None)  # type: ignore[arg-type]
        self._assignments = list(assignments)
        self.infer_calls: List[Dict[str, Sequence[str]]] = []

    def infer(
        self,
        profiles: Dict[str, SpeakerProfile],
        roster: Sequence[str],
        context_summary: str,
        target_speakers: Sequence[str],
    ) -> List[SpeakerAssignment]:
        self.infer_calls.append(
            {
                "profiles": tuple(sorted(profiles.keys())),
                "roster": tuple(roster),
                "targets": tuple(target_speakers),
            }
        )
        return list(self._assignments)

    def critic(self, assignments: Sequence[SpeakerAssignment]) -> Dict[str, bool]:
        return {assignment.speaker_id: True for assignment in assignments}


def test_candidate_roster_handles_missing_metadata(monkeypatch):
    config = AssignmentConfig()
    roster = CandidateRoster(config)

    monkeypatch.setattr(roster, "load_spacy_model", lambda: None)

    episode = PodcastEpisode(
        episode_id="ep",
        show_title="Show Title",
        episode_title="Episode",
        source_path=Path("audio.m4a"),
        metadata={},
    )
    result = roster.compile(episode, additional=["Alice"])
    assert "Alice" in result
    assert "Unknown Host" in result


def test_transcript_assigner_persists_results(tmp_path, monkeypatch):
    config = AssignmentConfig(data_root=tmp_path / "data")
    episode = PodcastEpisode(
        episode_id="ep-1",
        show_title="Test Show",
        episode_title="Episode 1",
        source_path=tmp_path / "audio.wav",
        metadata={"description": "A conversation with Bob"},
    )

    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="SPEAKER_00"),
        TranscriptSegment(start=1.0, end=2.0, text="Hi", speaker_id="SPEAKER_01"),
    ]

    engine = StubEngine(
        assignments=[
            SpeakerAssignment(speaker_id="SPEAKER_00", proposed_name="Alice", confidence=0.9),
            SpeakerAssignment(speaker_id="SPEAKER_01", proposed_name="Bob", confidence=0.8),
        ]
    )

    assigner = TranscriptAssigner(
        podcast=episode,
        config=config,
        inference_engine=engine,
    )

    monkeypatch.setattr(CandidateRoster, "compile", lambda self, *_args, **_kwargs: ["Alice", "Bob"])

    results = assigner.assign_names(segments)
    assert [seg.speaker_name for seg in results] == ["Alice", "Bob"]

    assignment_path = assigner._last_assignment_path  # type: ignore[attr-defined]
    assert assignment_path.exists()


def test_candidate_roster_with_spacy(monkeypatch):
    config = AssignmentConfig()
    roster = CandidateRoster(config)

    class DummyEnt:
        def __init__(self, text: str, label_: str) -> None:
            self.text = text
            self.label_ = label_

    class DummyDoc:
        def __init__(self) -> None:
            self.ents = [DummyEnt("Charlie", "PERSON"), DummyEnt("Paris", "GPE")]

    class DummyNLP:
        def __call__(self, _text: str) -> DummyDoc:
            return DummyDoc()

    monkeypatch.setattr(CandidateRoster, "load_spacy_model", lambda self: DummyNLP())

    episode = PodcastEpisode(
        episode_id="ep",
        show_title="Show Title",
        episode_title="Episode Name",
        source_path=Path("audio.m4a"),
        metadata={"description": "Interview with Charlie"},
    )

    result = roster.compile(episode, additional=["Alice"])
    assert "Alice" in result
    assert "Charlie" in result
    assert "Unknown Host" in result


def test_inference_engine_prompt_and_parse():
    config = AssignmentConfig()

    class DummyClient:
        def generate(self, prompt: str) -> str:
            assert "Candidate Names" in prompt
            return '[{"speaker_id":"SPEAKER_00","proposed_name":"Dana","confidence":0.8,"justification":"Host"}]'

    engine = SpeakerInferenceEngine(config, client=DummyClient())  # type: ignore[arg-type]

    profiles = {
        "SPEAKER_00": SpeakerProfile(speaker_id="SPEAKER_00"),
    }
    roster = ["Dana"]
    assignments = engine.infer(profiles, roster, "Episode Context", ["SPEAKER_00"])
    assert assignments[0].proposed_name == "Dana"

    critic = engine.critic(assignments)
    assert critic["SPEAKER_00"]

    merged = engine.consolidate({}, assignments)
    assert "SPEAKER_00" in merged


def test_inference_engine_recovers_unquoted_speaker_ids():
    config = AssignmentConfig()
    engine = SpeakerInferenceEngine(config, client=None)  # type: ignore[arg-type]
    raw = """
    [
      {
        "speaker_id": SPEAKER_03,
        "proposed_name": "Pomp",
        "confidence": 1.0,
        "justification": "Host introduction"
      }
    ]
    """
    assignments = engine._parse_assignments(raw)
    assert assignments and assignments[0].speaker_id == "SPEAKER_03"


def test_assignment_snapshots(tmp_path):
    config = AssignmentConfig(
        data_root=tmp_path / "data",
        ollama_model="llama3",
        spacy_model="en_core_web_sm",
        sample_utterances_start=2,
        sample_utterances_end=4,
    )
    episode = PodcastEpisode(
        episode_id="ep-2",
        show_title="Snapshot Show",
        episode_title="Snapshot Episode",
        source_path=tmp_path / "audio.wav",
    )
    engine = StubEngine(
        assignments=[
            SpeakerAssignment(speaker_id="S0", proposed_name="Alex", confidence=0.7),
        ]
    )
    assigner = TranscriptAssigner(
        podcast=episode,
        config=config,
        inference_engine=engine,
    )

    config_snapshot = assigner._config_snapshot()
    assert config_snapshot["ollama_model"] == "llama3"
    assert config_snapshot["sample_range"] == (2, 4)
    assert config_snapshot["data_root"] == str(config.data_root)

    assignment_snapshot = assigner._assignment_snapshot(
        [
            SpeakerAssignment(speaker_id="S0", proposed_name="Host", confidence=0.91),
            SpeakerAssignment(speaker_id="S1", proposed_name="Guest", confidence=0.42),
        ]
    )
    assert assignment_snapshot == [
        {"speaker_id": "S0", "name": "Host", "confidence": 0.91},
        {"speaker_id": "S1", "name": "Guest", "confidence": 0.42},
    ]

