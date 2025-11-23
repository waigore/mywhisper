from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, List, Optional

import requests
import spacy

from .assign import LLMClient, OllamaClient
from .config import ensure_episode_subdir, resolve_data_root
from .logging_utils import LoggingBase
from .models import PipelineEvent, PodcastEpisode
from .podcasts import PodcastCatalog

LOGGER = logging.getLogger("mywhisper.vocative")

# Common words to exclude from proper noun extraction
EXCLUDED_PROPER_NOUNS = {
    "bitcoin",
    "fed",
    "china",
    "qe",
    "ism",
    "repo",
    "real",
    "vision",
    "core",
    "taproot",
}


@dataclass(slots=True)
class VocativeConfig:
    """
    Configuration for vocative detection in podcast segments.
    """

    data_root: Path = field(default_factory=resolve_data_root)
    spacy_model: str = "en_core_web_sm"
    output_subdir: str = "transcripts"
    llm_model: str = "llama3"
    llm_endpoint: str = "http://localhost:11434/api/generate"

    def classified_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        """Locate the classified JSON input file."""
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_classified.json"

    def vocative_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        """Locate the vocative JSON output file."""
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_vocative.json"


class EpisodeVocativeDetector(LoggingBase):
    """
    Detect direct named addresses (vocatives) in podcast segments using SpaCy NER and dependency parsing.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: Optional[VocativeConfig] = None,
        catalog: Optional[PodcastCatalog] = None,
        client: Optional[LLMClient] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config or VocativeConfig()
        self.catalog = catalog
        self.client = client or OllamaClient(self.config.llm_model, endpoint=self.config.llm_endpoint)
        base_logger = logger or LOGGER
        self.logger = base_logger.getChild(podcast.episode_id)
        self._last_vocative_path: Optional[Path] = None
        self._nlp: Optional[Any] = None

    def detect_vocatives(
        self,
        *,
        classified_path: Optional[Path] = None,
        yield_progress: bool = False,
    ) -> Path | Generator[PipelineEvent, None, Path]:
        pipeline = self._pipeline(classified_path=classified_path)
        if yield_progress:
            return pipeline

        try:
            while True:
                next(pipeline)
        except StopIteration as stop:
            return stop.value

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _pipeline(self, classified_path: Optional[Path]) -> Generator[PipelineEvent, None, Path]:
        self.logger.debug(
            "_pipeline called | classified_path=%s",
            classified_path,
        )
        episode_key = self.podcast.episode_key
        resolved_classified = classified_path or self.config.classified_path(self.podcast, episode_key)
        resolved_classified = resolved_classified.resolve()
        vocative_path = self.config.vocative_path(self.podcast, episode_key).resolve()

        if not resolved_classified.exists():
            raise FileNotFoundError(f"Classified transcript not found at {resolved_classified}")

        start_time = time.perf_counter()
        self.logger.info(
            "Vocative detection start | episode=%s | classified=%s | output=%s",
            self.podcast.episode_id,
            resolved_classified,
            vocative_path,
        )

        yield PipelineEvent(
            stage="vocative",
            step_name="vocative",
            episode_id=self.podcast.episode_id,
            message="Loading classified transcript",
            payload={
                "episode_key": episode_key,
                "classified_path": str(resolved_classified),
                "step": "load",
            },
            artefact_paths={"classified": resolved_classified},
            checkpoint={
                "status": "started",
                "step": "vocative",
                "classified_path": str(resolved_classified),
                "episode_key": episode_key,
            },
        )

        # Load classified records
        records = json.loads(resolved_classified.read_text(encoding="utf-8"))
        segments: List[dict] = [rec for rec in records if isinstance(rec, dict)]
        segment_count = len(segments)

        enriched: List[dict] = []

        for index, seg in enumerate(segments):
            segment_text = str(seg.get("text") or "").strip()
            if not segment_text:
                # Empty segment, add with empty candidates array
                enriched_segment = {
                    **seg,
                    "addressed_person_candidates": [],
                }
                enriched.append(enriched_segment)
                continue

            # Detect vocative candidates in segment
            candidates = self._detect_vocative_in_segment(segment_text)

            enriched_segment = {
                **seg,
                "addressed_person_candidates": candidates,
            }
            enriched.append(enriched_segment)

            payload = {
                "segment_index": index,
                "step": "segment_completed",
                "addressed_person_candidates": candidates,
            }
            yield PipelineEvent(
                stage="vocative",
                step_name="vocative",
                episode_id=self.podcast.episode_id,
                message=f"Processed segment {index + 1}/{segment_count}",
                payload=payload,
                checkpoint={
                    "status": f"segment_{index + 1}",
                    "step": "vocative",
                    "segment_index": index,
                },
            )

        vocative_path.parent.mkdir(parents=True, exist_ok=True)
        vocative_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(
            "Persisted vocative detections | path=%s | segments=%d",
            vocative_path,
            len(enriched),
        )

        artefact_key = f"{episode_key}_vocative"
        if self.catalog:
            self.catalog.record_artefact(
                episode_id=self.podcast.episode_id,
                kind="vocative",
                path=vocative_path,
                artefact_key=artefact_key,
            )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="vocative",
            step_name="vocative",
            episode_id=self.podcast.episode_id,
            message="Persisted vocative detections",
            payload={
                "path": str(vocative_path),
                "segments": len(enriched),
                "step": "completed",
            },
            artefact_paths={"vocative": vocative_path},
            checkpoint={
                "status": "completed",
                "step": "vocative",
                "vocative_path": str(vocative_path),
                "segments": len(enriched),
                "classified_path": str(resolved_classified),
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )

        self._last_vocative_path = vocative_path
        self.logger.debug(
            "_pipeline returning | vocative_path=%s | segments_processed=%d",
            vocative_path,
            len(enriched),
        )
        return vocative_path

    def _detect_vocative_in_segment(self, text: str) -> List[dict]:
        """
        Detect vocative candidates in a segment and classify each one.
        Returns a list of candidate objects, each with 'name', 'classification', 'justification', and 'sentence' fields.
        Each occurrence of the same name within the segment is processed separately.
        Returns empty list when no candidates found.
        """
        self.logger.debug(
            "_detect_vocative_in_segment called | text_length=%d | text_preview=%s",
            len(text) if text else 0,
            (text[:100] + "..." if text and len(text) > 100 else text) if text else None,
        )
        if not text or not text.strip():
            self.logger.debug("_detect_vocative_in_segment returning | result=[] (empty text)")
            return []

        # Use NER and dependency parsing to identify vocative candidates (returns unique names)
        unique_vocatives = self._identify_vocatives(text)
        if not unique_vocatives:
            self.logger.debug("_detect_vocative_in_segment returning | result=[] (no candidates found)")
            return []

        # Find all occurrences of each vocative name in the text
        # Process each occurrence separately to get context-specific classifications
        results = []
        for vocative_name in unique_vocatives:
            # Find all positions where this vocative name appears in the text
            occurrences = self._find_all_occurrences(text, vocative_name)
            
            for position in occurrences:
                # Extract sentence containing this specific occurrence
                sentence = self._extract_sentence_with_vocative_at_position(text, vocative_name, position)
                if sentence:
                    # Classify with LLM
                    classification_result = self._classify_candidate_with_llm(sentence, vocative_name)
                    results.append({
                        "name": vocative_name,
                        "classification": classification_result["classification"],
                        "justification": classification_result.get("justification", ""),
                        "sentence": sentence
                    })
                    self.logger.debug(
                        "_detect_vocative_in_segment | candidate=%s | position=%d | classification=%s | justification=%s",
                        vocative_name,
                        position,
                        classification_result["classification"],
                        classification_result.get("justification", "")[:100] if classification_result.get("justification") else "",
                    )
                else:
                    # Sentence extraction failed, default to UNKNOWN since we can't classify without context
                    self.logger.warning(
                        "_detect_vocative_in_segment | candidate=%s | position=%d | classification=UNKNOWN (sentence extraction failed)",
                        vocative_name,
                        position,
                    )
                    results.append({
                        "name": vocative_name,
                        "classification": "UNKNOWN",
                        "justification": "Sentence extraction failed, cannot classify without context",
                        "sentence": ""
                    })

        self.logger.debug(
            "_detect_vocative_in_segment returning | result=%s | count=%d",
            results,
            len(results),
        )
        return results

    def _extract_person_names(self, text: str) -> List[str]:
        """
        Extract person names from text using SpaCy.
        Returns a list of candidate person names.
        Only includes proper nouns that are labeled as PERSON entities.
        """
        self.logger.debug(
            "_extract_person_names called | text_length=%d | text_preview=%s",
            len(text) if text else 0,
            (text[:100] + "..." if text and len(text) > 100 else text) if text else None,
        )
        nlp = self._get_nlp()
        if not nlp:
            self.logger.debug("_extract_person_names returning | result=[] (nlp not available)")
            return []

        doc = nlp(text)
        candidates = set()

        # Named entities labeled as PERSON
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidates.add(ent.text)

        # Standalone proper nouns that are within PERSON entity spans
        # (catches names spaCy identified as PROPN that are part of PERSON entities)
        person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
        for token in doc:
            if token.pos_ == "PROPN" and token.text[0].isupper():
                if len(token.text) > 1 and token.text.lower() not in EXCLUDED_PROPER_NOUNS:
                    # Only include if this token is part of any PERSON entity
                    # Check if token is within any PERSON entity's token span
                    if any(token.i >= ent.start and token.i < ent.end for ent in person_entities):
                        candidates.add(token.text)

        result = sorted(list(candidates))
        self.logger.debug(
            "_extract_person_names returning | result=%s | count=%d",
            result,
            len(result),
        )
        return result

    def _identify_vocatives(self, text: str) -> List[str]:
        """
        Identifies potential vocatives in a given text using spaCy NER and dependency parsing.
        
        Returns a list of all identified vocative candidates (person names), empty list if none found.
        """
        self.logger.debug(
            "_identify_vocatives called | text_length=%d | text_preview=%s",
            len(text) if text else 0,
            (text[:100] + "..." if text and len(text) > 100 else text) if text else None,
        )
        nlp = self._get_nlp()
        if not nlp:
            self.logger.debug("_identify_vocatives returning | result=[] (nlp not available)")
            return []

        doc = nlp(text)
        vocatives = []
        potential_vocatives = set()

        for sent in doc.sents:
            # Check for vocatives separated by commas, which are often proper nouns (PROPN)
            # or common nouns (NOUN) used as address terms (e.g., 'sir', 'madam').

            # Case 1: Vocative at the beginning, followed by a comma (or other punctuation)
            if len(sent) > 1 and sent[0].pos_ in ["PROPN", "NOUN"] and sent[1].is_punct:
                # Further refinement: check if the next token is a verb, indicating direct action
                if len(sent) > 2 and sent[2].pos_ == "VERB":
                    potential_vocatives.add(sent[0].text)

            # Case 2: Vocative at the end, preceded by a comma (or other punctuation)
            if len(sent) > 1 and sent[-1].pos_ in ["PROPN", "NOUN"] and sent[-2].is_punct:
                potential_vocatives.add(sent[-1].text)

            # Case 3: Vocative in the middle, surrounded by punctuation (e.g., "..., Josh, ...")
            # This handles cases like "And the final thing, Josh, and this is where..."
            for i in range(1, len(sent) - 1):
                token = sent[i]
                prev_token = sent[i - 1]
                next_token = sent[i + 1]
                # Check if token is a proper noun or noun, surrounded by commas
                # Commas are the most common punctuation for vocatives in the middle of sentences
                if (token.pos_ in ["PROPN", "NOUN"] 
                    and prev_token.is_punct 
                    and next_token.is_punct 
                    and (prev_token.text == "," or next_token.text == ",")):
                    potential_vocatives.add(token.text)

        # Filter for names found by NER which also match our potential_vocatives set
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text in potential_vocatives:
                vocatives.append(ent.text)

        # Also check potential vocatives that are proper nouns but might not be recognized as PERSON entities
        # This handles cases where NER doesn't label a name as PERSON but it appears in a vocative pattern
        for token in doc:
            if token.text in potential_vocatives:
                # Check if it's a proper noun (PROPN) that starts with uppercase
                if token.pos_ == "PROPN" and token.text[0].isupper():
                    # Exclude common words that might match the pattern
                    if token.text.lower() not in EXCLUDED_PROPER_NOUNS:
                        vocatives.append(token.text)

        # Remove duplicates and return all candidates
        unique_vocatives = sorted(list(set(vocatives)))
        self.logger.debug(
            "_identify_vocatives returning | result=%s | count=%d",
            unique_vocatives,
            len(unique_vocatives),
        )
        return unique_vocatives

    def _extract_sentence_with_vocative(self, text: str, vocative: str) -> Optional[str]:
        """
        Extract the sentence containing the vocative candidate.
        Uses a more robust approach that finds the vocative position and extracts
        a context window with proper sentence boundaries, rather than relying solely
        on spaCy's sentence segmentation which can be incorrect.
        Returns the sentence text, or None if not found.
        """
        self.logger.debug(
            "_extract_sentence_with_vocative called | text_length=%d | vocative=%s",
            len(text) if text else 0,
            vocative,
        )
        if not text or not text.strip() or not vocative:
            self.logger.debug("_extract_sentence_with_vocative returning | result=None (empty input)")
            return None

        # First, try to find the vocative in the text
        vocative_pos = text.find(vocative)
        if vocative_pos == -1:
            self.logger.debug("_extract_sentence_with_vocative returning | result=None (vocative not found in text)")
            return None

        # Find the start of the sentence by looking backwards for sentence-ending punctuation
        # Look back up to 500 characters
        search_start = max(0, vocative_pos - 500)
        start = search_start  # Default to searching from 500 chars back
        found_sentence_start = False
        for i in range(vocative_pos - 1, search_start - 1, -1):
            char = text[i]
            # If we find sentence-ending punctuation followed by whitespace, start after it
            if char in '.!?' and i + 1 < len(text):
                # Check if there's whitespace or newline after the punctuation
                if i + 1 < len(text) and text[i + 1] in ' \n\t':
                    start = i + 1
                    # Skip any leading whitespace
                    while start < len(text) and text[start] in ' \n\t':
                        start += 1
                    found_sentence_start = True
                    break
            # If we hit a newline, that might be a boundary too (but less reliable)
            elif char == '\n' and i > search_start + 50:  # Only if we've gone back a bit
                start = i + 1
                while start < len(text) and text[start] in ' \n\t':
                    start += 1
                found_sentence_start = True
                break
        
        # If we didn't find a sentence start, use the search_start (500 chars back or beginning)
        if not found_sentence_start:
            start = search_start

        # Find the end of the sentence by looking forwards for sentence-ending punctuation
        # Look forward up to 500 characters to find the sentence end
        end = vocative_pos + len(vocative)
        search_end = min(len(text), vocative_pos + 500)
        found_sentence_end = False
        for i in range(vocative_pos + len(vocative), search_end):
            char = text[i]
            # If we find sentence-ending punctuation, end after it
            if char in '.!?':
                end = i + 1
                found_sentence_end = True
                break
            # If we hit a newline after some content, that might be a boundary
            elif char == '\n' and i > vocative_pos + len(vocative) + 20:
                end = i
                found_sentence_end = True
                break
        
        # If we didn't find a sentence end, extend to a reasonable limit
        # but try to avoid including multiple sentences
        if not found_sentence_end:
            # Extend to 200 characters max if no sentence end found
            end = min(len(text), vocative_pos + 200)

        # Extract the sentence
        result = text[start:end].strip()
        
        # Fallback: if the result is too short or doesn't seem like a complete sentence,
        # try using spaCy's sentence segmentation as a backup
        if len(result) < len(vocative) + 10:  # Result is suspiciously short
            nlp = self._get_nlp()
            if nlp:
                doc = nlp(text)
                for sent in doc.sents:
                    if vocative in sent.text:
                        result = sent.text.strip()
                        break

        self.logger.debug(
            "_extract_sentence_with_vocative returning | result=%s",
            result[:100] + "..." if len(result) > 100 else result,
        )
        return result if result else None

    def _find_all_occurrences(self, text: str, name: str) -> List[int]:
        """
        Find all character positions where a vocative name appears in the text.
        Returns a list of character positions (start indices) where the name occurs.
        Uses word boundary matching to avoid matching substrings within words.
        """
        self.logger.debug(
            "_find_all_occurrences called | text_length=%d | name=%s",
            len(text) if text else 0,
            name,
        )
        if not text or not name:
            self.logger.debug("_find_all_occurrences returning | result=[] (empty input)")
            return []

        positions = []
        search_start = 0
        name_len = len(name)
        
        while True:
            # Find the next occurrence of the name
            pos = text.find(name, search_start)
            if pos == -1:
                break
            
            # Check word boundaries to ensure we're matching a whole word
            # Check character before (if any)
            before_ok = pos == 0 or not text[pos - 1].isalnum()
            # Check character after (if any)
            after_ok = pos + name_len >= len(text) or not text[pos + name_len].isalnum()
            
            if before_ok and after_ok:
                positions.append(pos)
                self.logger.debug(
                    "_find_all_occurrences | found occurrence | name=%s | position=%d",
                    name,
                    pos,
                )
            
            # Move search start past this occurrence
            search_start = pos + 1

        self.logger.debug(
            "_find_all_occurrences returning | name=%s | positions=%s | count=%d",
            name,
            positions,
            len(positions),
        )
        return positions

    def _extract_sentence_with_vocative_at_position(self, text: str, vocative: str, position: int) -> Optional[str]:
        """
        Extract the sentence containing the vocative candidate at a specific character position.
        Uses the same robust approach as _extract_sentence_with_vocative but with a known position.
        Returns the sentence text, or None if position is invalid.
        """
        self.logger.debug(
            "_extract_sentence_with_vocative_at_position called | text_length=%d | vocative=%s | position=%d",
            len(text) if text else 0,
            vocative,
            position,
        )
        if not text or not text.strip() or not vocative or position < 0 or position >= len(text):
            self.logger.debug("_extract_sentence_with_vocative_at_position returning | result=None (invalid input)")
            return None

        # Verify the vocative is actually at this position
        if text[position:position + len(vocative)] != vocative:
            self.logger.warning(
                "_extract_sentence_with_vocative_at_position | vocative not found at position | vocative=%s | position=%d",
                vocative,
                position,
            )
            return None

        vocative_pos = position

        # Find the start of the sentence by looking backwards for sentence-ending punctuation
        # Look back up to 500 characters
        search_start = max(0, vocative_pos - 500)
        start = search_start  # Default to searching from 500 chars back
        found_sentence_start = False
        for i in range(vocative_pos - 1, search_start - 1, -1):
            char = text[i]
            # If we find sentence-ending punctuation followed by whitespace, start after it
            if char in '.!?' and i + 1 < len(text):
                # Check if there's whitespace or newline after the punctuation
                if i + 1 < len(text) and text[i + 1] in ' \n\t':
                    start = i + 1
                    # Skip any leading whitespace
                    while start < len(text) and text[start] in ' \n\t':
                        start += 1
                    found_sentence_start = True
                    break
            # If we hit a newline, that might be a boundary too (but less reliable)
            elif char == '\n' and i > search_start + 50:  # Only if we've gone back a bit
                start = i + 1
                while start < len(text) and text[start] in ' \n\t':
                    start += 1
                found_sentence_start = True
                break
        
        # If we didn't find a sentence start, use the search_start (500 chars back or beginning)
        if not found_sentence_start:
            start = search_start

        # Find the end of the sentence by looking forwards for sentence-ending punctuation
        # Look forward up to 500 characters to find the sentence end
        end = vocative_pos + len(vocative)
        search_end = min(len(text), vocative_pos + 500)
        found_sentence_end = False
        for i in range(vocative_pos + len(vocative), search_end):
            char = text[i]
            # If we find sentence-ending punctuation, end after it
            if char in '.!?':
                end = i + 1
                found_sentence_end = True
                break
            # If we hit a newline after some content, that might be a boundary
            elif char == '\n' and i > vocative_pos + len(vocative) + 20:
                end = i
                found_sentence_end = True
                break
        
        # If we didn't find a sentence end, extend to a reasonable limit
        # but try to avoid including multiple sentences
        if not found_sentence_end:
            # Extend to 200 characters max if no sentence end found
            end = min(len(text), vocative_pos + 200)

        # Extract the sentence
        result = text[start:end].strip()
        
        # Fallback: if the result is too short or doesn't seem like a complete sentence,
        # try using spaCy's sentence segmentation as a backup
        if len(result) < len(vocative) + 10:  # Result is suspiciously short
            nlp = self._get_nlp()
            if nlp:
                doc = nlp(text)
                for sent in doc.sents:
                    # Check if the vocative position is within this sentence's span
                    sent_start = sent.start_char if hasattr(sent, 'start_char') else 0
                    sent_end = sent.end_char if hasattr(sent, 'end_char') else len(text)
                    if sent_start <= vocative_pos < sent_end:
                        result = sent.text.strip()
                        break

        self.logger.debug(
            "_extract_sentence_with_vocative_at_position returning | result=%s",
            result[:100] + "..." if len(result) > 100 else result,
        )
        return result if result else None

    def _classify_candidate_with_llm(self, sentence: str, candidate: str) -> dict[str, str]:
        """
        Classify a vocative candidate using LLM.
        Returns a dict with "classification" ("VOCATIVE" or "OTHER") and "justification" (explanation).
        Returns {"classification": "UNKNOWN", "justification": "<reason>"} if the LLM is unavailable or cannot return a valid response.
        """
        self.logger.debug(
            "_classify_candidate_with_llm called | candidate=%s | sentence_length=%d",
            candidate,
            len(sentence) if sentence else 0,
        )
        if not sentence or not candidate:
            self.logger.warning("_classify_candidate_with_llm returning | result=UNKNOWN (empty input)")
            return {"classification": "UNKNOWN", "justification": "Empty input provided"}

        prompt = (
            "You are analyzing a podcast transcript to determine if a name is being used as a direct address (vocative) "
            "or serving some other linguistic function.\n\n"
            f"Sentence containing the candidate: {sentence}\n"
            f"Candidate name: {candidate}\n\n"
            "A vocative is when someone directly addresses another person by name (e.g., 'Josh, what do you think?' or "
            "'What do you think, Josh?').\n"
            "Other linguistic functions include: mentioning a person in third person (e.g., 'I saw Josh yesterday'), "
            "referring to a person as part of a larger phrase (e.g., 'We uploaded a video, a podcast, EGS, to Gemini'), "
            "or any other non-vocative usage.\n\n"
            "Return ONLY valid JSON: {\"classification\": \"VOCATIVE\" or \"OTHER\", \"justification\": \"<explanation>\"}.\n"
            "Use \"VOCATIVE\" if the candidate is being used as a direct address (vocative).\n"
            "Use \"OTHER\" if the candidate is serving some other linguistic function.\n"
            "Provide a brief justification explaining your classification decision in the \"justification\" field."
        )

        try:
            response = self.client.generate(prompt)
            self.logger.debug("_classify_candidate_with_llm LLM response | response=%s", response[:200])
            
            # Parse JSON response
            data = json.loads(response.strip())
            if isinstance(data, dict):
                classification = data.get("classification", "UNKNOWN")
                justification = data.get("justification", "")
                if classification.upper() in ("VOCATIVE", "OTHER"):
                    result = {
                        "classification": classification.upper(),
                        "justification": str(justification).strip() if justification else ""
                    }
                    self.logger.debug(
                        "_classify_candidate_with_llm returning | classification=%s | justification=%s",
                        result["classification"],
                        result["justification"][:100] if result["justification"] else "",
                    )
                    return result
            
            self.logger.warning(
                "_classify_candidate_with_llm returning | result=UNKNOWN (invalid response format)"
            )
            return {"classification": "UNKNOWN", "justification": "Invalid response format from LLM"}
        except json.JSONDecodeError as exc:
            self.logger.warning(
                "_classify_candidate_with_llm returning | result=UNKNOWN (JSON decode error: %s)",
                exc,
            )
            return {"classification": "UNKNOWN", "justification": f"JSON decode error: {exc}"}
        except requests.exceptions.RequestException as exc:
            # Handle HTTP errors (404, connection errors, etc.) gracefully
            # These are expected when Ollama is not running or endpoint is unavailable
            self.logger.debug(
                "_classify_candidate_with_llm returning | result=UNKNOWN (LLM service unavailable: %s)",
                exc,
            )
            return {"classification": "UNKNOWN", "justification": f"LLM service unavailable: {exc}"}
        except Exception as exc:
            # For unexpected errors, log with traceback
            self.logger.warning(
                "_classify_candidate_with_llm returning | result=UNKNOWN (LLM call failed: %s)",
                exc,
                exc_info=exc,
            )
            return {"classification": "UNKNOWN", "justification": f"LLM call failed: {exc}"}

    def _get_nlp(self) -> Optional[Any]:
        """Lazy-load the SpaCy model."""
        self.logger.debug(
            "_get_nlp called | spacy_model=%s | nlp_loaded=%s",
            self.config.spacy_model,
            self._nlp is not None,
        )
        if self._nlp is not None:
            self.logger.debug("_get_nlp returning | result=<nlp_model> (cached)")
            return self._nlp

        try:
            self._nlp = spacy.load(self.config.spacy_model)
            self.logger.debug("_get_nlp returning | result=<nlp_model> (loaded)")
            return self._nlp
        except OSError:
            self.logger.warning(
                "spaCy model %s not found; vocative detection will be limited. Install with: python -m spacy download %s",
                self.config.spacy_model,
                self.config.spacy_model,
            )
            self._nlp = None
            self.logger.debug("_get_nlp returning | result=None (OSError)")
            return None
        except Exception as exc:
            self.logger.warning("Failed to load spaCy model: %s", exc, exc_info=exc)
            self._nlp = None
            self.logger.debug("_get_nlp returning | result=None (Exception)")
            return None

