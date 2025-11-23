from __future__ import annotations

from mywhisper.models import (
    SpeakerAssignment,
    SpeakerNameGuesses,
    SpeakerProfile,
    TranscriptSegment,
)


def test_transcript_segment_duration():
    """Test TranscriptSegment.duration() method (hits line 99)."""
    segment = TranscriptSegment(
        start=10.0,
        end=15.5,
        text="Hello",
        speaker_id="S0",
    )
    assert segment.duration() == 5.5
    
    # Test with end < start (should return 0.0)
    segment_negative = TranscriptSegment(
        start=15.0,
        end=10.0,
        text="Invalid",
        speaker_id="S0",
    )
    assert segment_negative.duration() == 0.0


def test_speaker_profile_update_from_segments_empty():
    """Test SpeakerProfile.update_from_segments with empty segments (hits line 125)."""
    profile = SpeakerProfile(speaker_id="S0")
    profile.update_from_segments([], sample_start=2, sample_end=2)
    assert profile.total_turns == 0
    assert profile.total_duration == 0.0


def test_speaker_profile_update_from_segments():
    """Test SpeakerProfile.update_from_segments (hits lines 127-135)."""
    profile = SpeakerProfile(speaker_id="S0")
    segments = [
        TranscriptSegment(start=0.0, end=5.0, text="First", speaker_id="S0"),
        TranscriptSegment(start=10.0, end=15.0, text="Second", speaker_id="S0"),
        TranscriptSegment(start=20.0, end=25.0, text="Third", speaker_id="S0"),
    ]
    profile.update_from_segments(segments, sample_start=2, sample_end=2)
    assert profile.total_turns == 3
    assert profile.total_duration == 15.0
    assert profile.first_start == 0.0
    assert profile.last_end == 25.0
    assert len(profile.snippets) == 3
    assert len(profile.sample_quotes) > 0


def test_speaker_profile_to_prompt_block():
    """Test SpeakerProfile.to_prompt_block() method (hits lines 142-150)."""
    profile = SpeakerProfile(speaker_id="S0")
    segments = [
        TranscriptSegment(start=0.0, end=5.0, text="Hello world", speaker_id="S0"),
        TranscriptSegment(start=10.0, end=15.0, text="How are you?", speaker_id="S0"),
    ]
    profile.update_from_segments(segments, sample_start=2, sample_end=2)
    block = profile.to_prompt_block()
    assert "speaker_id: S0" in block
    assert "total_duration_sec" in block
    assert "turn_count: 2" in block
    assert "Hello world" in block or "How are you?" in block


def test_speaker_assignment_is_high_confidence():
    """Test SpeakerAssignment.is_high_confidence() method (hits line 163)."""
    assignment = SpeakerAssignment(
        speaker_id="S0",
        proposed_name="Alice",
        confidence=0.8,
    )
    assert assignment.is_high_confidence(0.7) is True
    assert assignment.is_high_confidence(0.8) is True
    assert assignment.is_high_confidence(0.9) is False


def test_speaker_name_guesses_add_proposal():
    """Test SpeakerNameGuesses.add_proposal() method (hits lines 179-194)."""
    guesses = SpeakerNameGuesses(speaker_id="S0")
    
    # Add first proposal
    proposal1 = SpeakerAssignment(
        speaker_id="S0",
        proposed_name="Alice",
        confidence=0.7,
    )
    guesses.add_proposal(proposal1)
    assert len(guesses.proposed_names) == 1
    
    # Add proposal with same name but higher confidence (should replace)
    proposal2 = SpeakerAssignment(
        speaker_id="S0",
        proposed_name="Alice",  # Same name
        confidence=0.9,  # Higher confidence
    )
    guesses.add_proposal(proposal2)
    assert len(guesses.proposed_names) == 1
    assert guesses.proposed_names[0].confidence == 0.9
    
    # Add proposal with different name (should add)
    proposal3 = SpeakerAssignment(
        speaker_id="S0",
        proposed_name="Bob",
        confidence=0.6,
    )
    guesses.add_proposal(proposal3)
    assert len(guesses.proposed_names) == 2
    # Should be sorted by confidence (descending)
    assert guesses.proposed_names[0].confidence >= guesses.proposed_names[1].confidence


def test_speaker_name_guesses_best():
    """Test SpeakerNameGuesses.best() method (hits line 199)."""
    guesses = SpeakerNameGuesses(speaker_id="S0")
    
    # Empty guesses should return None
    assert guesses.best() is None
    
    # Add proposals
    proposal1 = SpeakerAssignment(
        speaker_id="S0",
        proposed_name="Alice",
        confidence=0.7,
    )
    proposal2 = SpeakerAssignment(
        speaker_id="S0",
        proposed_name="Bob",
        confidence=0.9,  # Higher confidence
    )
    guesses.add_proposal(proposal1)
    guesses.add_proposal(proposal2)
    
    # Best should return highest confidence
    best = guesses.best()
    assert best is not None
    assert best.proposed_name == "Bob"
    assert best.confidence == 0.9

