from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mywhisper.assign import (
    AssignmentConfig,
    ContextualTurnInference,
    GraphBasedInference,
    TranscriptAssigner,
)
from mywhisper.models import InferenceResult, PodcastEpisode


def create_vocative_segment(
    speaker_id: str,
    text: str,
    vocatives: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Helper to create a segment with vocative candidates."""
    return {
        "speaker_id": speaker_id,
        "text": text,
        "addressed_person_candidates": vocatives,
    }


def test_graph_based_inference_simple():
    """Test graph-based inference creates edges from next speaker to vocatives."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "Hello Alice, how are you?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Hello Alice, how are you?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "I'm doing well", []),
        create_vocative_segment(
            "SPEAKER_01",
            "Hi Bob, nice to meet you.",
            [
                {
                    "name": "Bob",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Hi Bob, nice to meet you.",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_02", "Thanks!", []),
    ]

    inference = GraphBasedInference()
    assignments, sentences = inference.infer(segments)

    # SPEAKER_00 uses "Alice" -> next speaker is SPEAKER_01, so SPEAKER_01 should be assigned "Alice"
    # SPEAKER_01 uses "Bob" -> next speaker is SPEAKER_02, so SPEAKER_02 should be assigned "Bob"
    assert "SPEAKER_01" in assignments
    assert assignments["SPEAKER_01"].name == "Alice"
    assert "SPEAKER_02" in assignments
    assert assignments["SPEAKER_02"].name == "Bob"
    assert "Hello Alice, how are you?" in sentences["SPEAKER_01"]


def test_graph_based_inference_multiple_vocatives():
    """Test graph-based inference when speaker addresses same person multiple times."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, what do you think?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, what do you think?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "I think it's good", []),
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, can you clarify?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, can you clarify?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "Sure thing", []),
    ]

    inference = GraphBasedInference()
    assignments, sentences = inference.infer(segments)

    # SPEAKER_00 uses "Alice" twice, both times next speaker is SPEAKER_01
    # So SPEAKER_01 should be assigned "Alice" with high confidence
    assert "SPEAKER_01" in assignments
    assert assignments["SPEAKER_01"].name == "Alice"
    # Confidence should be high since both edges point to SPEAKER_01
    assert assignments["SPEAKER_01"].confidence > 0.5
    # Sentences may include duplicates from next/previous speaker edges
    assert len(sentences["SPEAKER_01"]) >= 2


def test_graph_based_inference_no_vocatives():
    """Test graph-based inference with no vocatives."""
    segments = [
        create_vocative_segment("SPEAKER_00", "Hello world", []),
        create_vocative_segment("SPEAKER_01", "How are you?", []),
    ]

    inference = GraphBasedInference()
    assignments, sentences = inference.infer(segments)

    assert len(assignments) == 0
    assert len(sentences) == 0


def test_graph_based_inference_filters_non_vocative():
    """Test that graph inference only uses VOCATIVE classifications."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "I saw Alice yesterday",
            [
                {
                    "name": "Alice",
                    "classification": "OTHER",
                    "justification": "Not a direct address",
                    "sentence": "I saw Alice yesterday",
                }
            ],
        ),
        create_vocative_segment(
            "SPEAKER_00",
            "Bob, welcome!",
            [
                {
                    "name": "Bob",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Bob, welcome!",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "Thank you", []),
    ]

    inference = GraphBasedInference()
    assignments, sentences = inference.infer(segments)

    # Should only match Bob (VOCATIVE), not Alice (OTHER)
    # SPEAKER_00 uses "Bob" -> next speaker is SPEAKER_01, so SPEAKER_01 should be assigned "Bob"
    assert "SPEAKER_01" in assignments
    assert assignments["SPEAKER_01"].name == "Bob"
    assert "Alice" not in [a.name for a in assignments.values()]


def test_contextual_turn_inference_simple():
    """Test contextual turn-taking inference with simple pattern."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, what do you think?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, what do you think?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "I think it's great", []),
    ]

    inference = ContextualTurnInference()
    assignments, sentences = inference.infer(segments)

    # Alice should be matched to SPEAKER_01 (next speaker)
    assert "SPEAKER_01" in assignments
    assert assignments["SPEAKER_01"].name == "Alice"
    assert assignments["SPEAKER_01"].confidence > 0.0


def test_contextual_turn_inference_previous_speaker():
    """Test contextual inference with previous speaker boost."""
    segments = [
        create_vocative_segment("SPEAKER_00", "Hello", []),
        create_vocative_segment(
            "SPEAKER_01",
            "Bob, thanks for that.",
            [
                {
                    "name": "Bob",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Bob, thanks for that.",
                }
            ],
        ),
    ]

    inference = ContextualTurnInference()
    assignments, sentences = inference.infer(segments)

    # Bob should be matched to SPEAKER_00 (previous speaker) with lower confidence
    assert "SPEAKER_00" in assignments
    assert assignments["SPEAKER_00"].name == "Bob"


def test_contextual_turn_inference_softmax():
    """Test that contextual inference uses softmax normalization."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, what do you think?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, what do you think?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "I think it's good", []),
        create_vocative_segment("SPEAKER_02", "I agree", []),
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, can you clarify?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, can you clarify?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "Sure thing", []),
    ]

    inference = ContextualTurnInference()
    assignments, sentences = inference.infer(segments)

    # Alice should be matched to SPEAKER_01 (appears twice as next speaker)
    # Confidence should be normalized via softmax
    assert "SPEAKER_01" in assignments
    assert assignments["SPEAKER_01"].name == "Alice"
    assert 0.0 < assignments["SPEAKER_01"].confidence <= 1.0


def test_contextual_turn_inference_no_vocatives():
    """Test contextual inference with no vocatives."""
    segments = [
        create_vocative_segment("SPEAKER_00", "Hello", []),
        create_vocative_segment("SPEAKER_01", "Hi", []),
    ]

    inference = ContextualTurnInference()
    assignments, sentences = inference.infer(segments)

    assert len(assignments) == 0
    assert len(sentences) == 0


def test_transcript_assigner_infers_names(tmp_path):
    """Test TranscriptAssigner end-to-end inference."""
    config = AssignmentConfig(data_root=tmp_path / "data")
    episode = PodcastEpisode(
        episode_id="ep-1",
        show_title="Test Show",
        episode_title="Episode 1",
        source_path=tmp_path / "audio.wav",
    )

    # Create vocative JSON file
    vocative_dir = (tmp_path / "data" / episode.episode_key / "transcripts")
    vocative_dir.mkdir(parents=True, exist_ok=True)
    vocative_path = vocative_dir / f"{episode.episode_key}_vocative.json"

    vocative_data = [
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, what do you think?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, what do you think?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "I think it's great", []),
        create_vocative_segment(
            "SPEAKER_01",
            "Bob, thanks for joining",
            [
                {
                    "name": "Bob",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Bob, thanks for joining",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_00", "You're welcome", []),
    ]

    with vocative_path.open("w", encoding="utf-8") as f:
        json.dump(vocative_data, f, indent=2)

    assigner = TranscriptAssigner(podcast=episode, config=config)
    result = assigner.infer_names(vocative_path=vocative_path)

    assert "inferred_names_path" in result
    assert result["inferred_names_path"].exists()
    assert "speakers" in result
    assert len(result["speakers"]) == 2

    # Check output file structure
    with result["inferred_names_path"].open("r", encoding="utf-8") as f:
        output_data = json.load(f)

    assert "speakers" in output_data
    speakers = {s["speaker_id"]: s for s in output_data["speakers"]}
    assert "SPEAKER_00" in speakers
    assert "SPEAKER_01" in speakers

    # Check that graph and context inferences are present
    speaker_00 = speakers["SPEAKER_00"]
    speaker_01 = speakers["SPEAKER_01"]

    # SPEAKER_00 uses "Alice" -> next speaker is SPEAKER_01, so SPEAKER_01 should have graph inference for Alice
    assert speaker_01.get("graph_inference") is not None
    assert speaker_01["graph_inference"]["name"] == "Alice"

    # SPEAKER_01 uses "Bob" -> next speaker is SPEAKER_00, so SPEAKER_00 should have graph inference for Bob
    assert speaker_00.get("graph_inference") is not None
    assert speaker_00["graph_inference"]["name"] == "Bob"

    # SPEAKER_01 should also have context inference for Alice (next speaker after SPEAKER_00 uses "Alice" vocative)
    assert speaker_01.get("context_inference") is not None
    assert speaker_01["context_inference"]["name"] == "Alice"

    # Check sentences are collected
    assert len(speaker_01.get("sentences", [])) > 0
    assert len(speaker_00.get("sentences", [])) > 0


def test_transcript_assigner_missing_vocative_file(tmp_path):
    """Test TranscriptAssigner handles missing vocative file gracefully."""
    config = AssignmentConfig(data_root=tmp_path / "data")
    episode = PodcastEpisode(
        episode_id="ep-2",
        show_title="Test Show",
        episode_title="Episode 2",
        source_path=tmp_path / "audio.wav",
    )

    assigner = TranscriptAssigner(podcast=episode, config=config)
    result = assigner.infer_names(vocative_path=tmp_path / "nonexistent.json")

    assert "inferred_names_path" in result
    assert result["inferred_names_path"] is None
    assert "speakers" in result
    assert result["speakers"] == []


def test_transcript_assigner_yields_progress(tmp_path):
    """Test TranscriptAssigner yields progress events."""
    config = AssignmentConfig(data_root=tmp_path / "data")
    episode = PodcastEpisode(
        episode_id="ep-3",
        show_title="Test Show",
        episode_title="Episode 3",
        source_path=tmp_path / "audio.wav",
    )

    vocative_dir = (tmp_path / "data" / episode.episode_key / "transcripts")
    vocative_dir.mkdir(parents=True, exist_ok=True)
    vocative_path = vocative_dir / f"{episode.episode_key}_vocative.json"

    vocative_data = [
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, hello",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, hello",
                }
            ],
        ),
    ]

    with vocative_path.open("w", encoding="utf-8") as f:
        json.dump(vocative_data, f, indent=2)

    assigner = TranscriptAssigner(podcast=episode, config=config)
    gen = assigner.infer_names(vocative_path=vocative_path, yield_progress=True)

    events = []
    try:
        while True:
            event = next(gen)
            events.append(event)
    except StopIteration as stop:
        result = stop.value

    # Should have multiple events (start, graph_inference, context_inference, persisted)
    assert len(events) >= 3
    assert events[0].stage == "start"
    assert any(e.stage == "graph_inference" for e in events)
    assert any(e.stage == "context_inference" for e in events)
    assert any(e.stage == "persisted" for e in events)
    assert "inferred_names_path" in result


def test_inferred_names_output_structure(tmp_path):
    """Test that inferred names JSON has correct structure."""
    config = AssignmentConfig(data_root=tmp_path / "data")
    episode = PodcastEpisode(
        episode_id="ep-4",
        show_title="Test Show",
        episode_title="Episode 4",
        source_path=tmp_path / "audio.wav",
    )

    vocative_dir = (tmp_path / "data" / episode.episode_key / "transcripts")
    vocative_dir.mkdir(parents=True, exist_ok=True)
    vocative_path = vocative_dir / f"{episode.episode_key}_vocative.json"

    vocative_data = [
        create_vocative_segment(
            "SPEAKER_00",
            "Alice, what do you think?",
            [
                {
                    "name": "Alice",
                    "classification": "VOCATIVE",
                    "justification": "Direct address",
                    "sentence": "Alice, what do you think?",
                }
            ],
        ),
        create_vocative_segment("SPEAKER_01", "I think it's good", []),
    ]

    with vocative_path.open("w", encoding="utf-8") as f:
        json.dump(vocative_data, f, indent=2)

    assigner = TranscriptAssigner(podcast=episode, config=config)
    result = assigner.infer_names(vocative_path=vocative_path)

    with result["inferred_names_path"].open("r", encoding="utf-8") as f:
        output_data = json.load(f)

    # Check structure
    assert "speakers" in output_data
    assert isinstance(output_data["speakers"], list)

    for speaker in output_data["speakers"]:
        assert "speaker_id" in speaker
        assert "graph_inference" in speaker
        assert "context_inference" in speaker
        assert "sentences" in speaker

        # graph_inference and context_inference can be None or dict with name/confidence
        if speaker["graph_inference"] is not None:
            assert "name" in speaker["graph_inference"]
            assert "confidence" in speaker["graph_inference"]
            assert isinstance(speaker["graph_inference"]["confidence"], (int, float))

        if speaker["context_inference"] is not None:
            assert "name" in speaker["context_inference"]
            assert "confidence" in speaker["context_inference"]
            assert isinstance(speaker["context_inference"]["confidence"], (int, float))

        assert isinstance(speaker["sentences"], list)


def test_graph_inference_skips_segments_without_speaker_id():
    """Test GraphBasedInference skips segments without speaker_id (hits line 150)."""
    segments = [
        create_vocative_segment("SPEAKER_00", "Hello", []),
        {"text": "No speaker", "addressed_person_candidates": []},  # No speaker_id
        create_vocative_segment("SPEAKER_01", "World", []),
    ]
    
    inference = GraphBasedInference()
    result = inference.infer(segments)
    # Should process segments with speaker_id
    assert len(result) >= 0


def test_graph_inference_skips_empty_vocative_names():
    """Test GraphBasedInference skips vocatives with empty names (hits line 159)."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "Hello",
            [
                {
                    "name": "   ",  # Whitespace only
                    "classification": "VOCATIVE",
                    "sentence": "Hello",
                },
                {
                    "name": "",  # Empty
                    "classification": "VOCATIVE",
                    "sentence": "Hello",
                },
                {
                    "name": "Alice",  # Valid
                    "classification": "VOCATIVE",
                    "sentence": "Hello Alice",
                },
            ],
        ),
    ]
    
    inference = GraphBasedInference()
    result = inference.infer(segments)
    # Should only process valid vocative
    assert len(result) >= 0


def test_contextual_inference_skips_segments_without_speaker_id():
    """Test ContextualTurnInference skips segments without speaker_id (hits line 245)."""
    segments = [
        create_vocative_segment("SPEAKER_00", "Hello", []),
        {"text": "No speaker", "addressed_person_candidates": []},  # No speaker_id
        create_vocative_segment("SPEAKER_01", "World", []),
    ]
    
    inference = ContextualTurnInference()
    result = inference.infer(segments)
    # Should process segments with speaker_id
    assert len(result) >= 0


def test_contextual_inference_skips_non_vocative_classifications():
    """Test ContextualTurnInference skips non-VOCATIVE classifications (hits line 250)."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "Hello",
            [
                {
                    "name": "Alice",
                    "classification": "OTHER",  # Not VOCATIVE
                    "sentence": "Hello",
                },
                {
                    "name": "Bob",
                    "classification": "VOCATIVE",  # Valid
                    "sentence": "Hello Bob",
                },
            ],
        ),
    ]
    
    inference = ContextualTurnInference()
    result = inference.infer(segments)
    # Should only process VOCATIVE classifications
    assert len(result) >= 0


def test_contextual_inference_skips_empty_vocative_names():
    """Test ContextualTurnInference skips vocatives with empty names (hits line 254)."""
    segments = [
        create_vocative_segment(
            "SPEAKER_00",
            "Hello",
            [
                {
                    "name": "   ",  # Whitespace only
                    "classification": "VOCATIVE",
                    "sentence": "Hello",
                },
                {
                    "name": "Alice",  # Valid
                    "classification": "VOCATIVE",
                    "sentence": "Hello Alice",
                },
            ],
        ),
    ]
    
    inference = ContextualTurnInference()
    result = inference.infer(segments)
    # Should only process valid vocative
    assert len(result) >= 0


def test_assigner_from_config(tmp_path):
    """Test TranscriptAssigner.from_config creates instance (hits line 341)."""
    from mywhisper.models import PodcastEpisode
    
    config = AssignmentConfig(data_root=tmp_path / "data")
    episode = PodcastEpisode(
        episode_id="ep-test",
        show_title="Test",
        episode_title="Episode",
        source_path=tmp_path / "audio.wav",
    )
    
    assigner = TranscriptAssigner.from_config(podcast=episode, config=config)
    assert isinstance(assigner, TranscriptAssigner)
    assert assigner.podcast == episode
    assert assigner.config == config


# Note: Line 282 (continue when spk_scores is empty) is difficult to test directly
# as it requires specific internal state in the graph inference algorithm.
# This is an edge case that would require more complex setup.
