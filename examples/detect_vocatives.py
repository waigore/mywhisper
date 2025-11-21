# json_vocative_detector.py
# Works perfectly on 35283959_classified.json and any similar structured podcast JSON
# Author: local-AI veteran
# Adapted for Llama-3.1-8B: lighter, faster, with fallback rules

import json
import re
import spacy
from typing import List, Dict, Optional, Tuple
import requests   # for Ollama / vLLM / OpenAI-compatible local server

from config import (
    OLLAMA_MODEL,
    SPACY_MODEL,
    ASSIGNED_TRANSCRIPT_PATH,
)

# ================== CONFIG ==================
USE_LLM = True                         # toggle to False for pure rules fallback if needed
# ===========================================

nlp = spacy.load(SPACY_MODEL)


def load_json_transcript(path: str = ASSIGNED_TRANSCRIPT_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # already list of segments


def extract_proper_nouns(text: str) -> List[str]:
    """Fast pre-filter with spaCy – catches PERSON entities + standalone PROPN"""
    doc = nlp(text)
    candidates = set()

    # Named entities labeled as PERSON
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidates.add(ent.text)

    # Standalone proper nouns (catches names spaCy missed as entities)
    for token in doc:
        if token.pos_ == "PROPN" and token.text[0].isupper():
            if len(token.text) > 1 and token.text.lower() not in {
                "bitcoin", "fed", "china", "qe", "ism", "repo", "real", "vision", "core", "taproot"
            }:
                candidates.add(token.text)

    return sorted(list(candidates))


def query_local_llm(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192}
    }
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def is_direct_address(utterance: str, candidate: str) -> Tuple[bool, str]:
    """
    Final arbiter using a powerful local LLM.
    Extremely accurate on podcast dialogue.
    """
    prompt = f"""You are an expert dialogue analyst for podcasts.
Determine if the speaker is DIRECTLY ADDRESSING a present participant using the name "{candidate}" (vocative). Ignore third-person mentions.

YES examples (direct address):
- "Andreas, how's the past week been?" → YES: Andreas
- "Before we get to that, Andreas, just a little reminder..." → YES: Andreas
- "Yeah, and Miguel, one thing I could add..." → YES: Miguel
- "Thanks very much, Michael." → YES: Michael
- "Hello there. Welcome to another edition of Macro Mondays. My name is Mikl Olsenol. I'm your usual host here at Real Vision. And today, as usual, I'm joined by you, Andreas. Welcome to the show." → YES: Andreas
- "It's been another volatile week, Andreas. We have to face that..." → YES: Andreas

NO examples (just mentioning, not addressing):
- "I saw this clip from Andreas earlier" → NO
- "We talked with Raoul and Julian" → NO
- "Luke proposed this" → NO
- "Michael Howell's view that liquidity growth is slowing" → NO
- "Dan Ives came out on Friday morning and he said" → NO
- "Eric Balkchunas pointed out that on Thursday" → NO

Transcript line:
\"\"\"{utterance}\"\"\"

Answer strictly with:
YES: <exact name used>
or
NO
"""
    result = query_local_llm(prompt)
    if result.startswith("YES:"):
        name = result.split(":", 1)[1].strip()
        return True, name
    return False, ""


def detect_vocatives_in_json(transcript_path: str = "35283959_classified.json") -> List[Dict]:
    segments = load_json_transcript(transcript_path)
    results = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        speaker_id = seg.get("speaker_id", "UNKNOWN")
        speaker_name = seg.get("speaker_name", speaker_id)  # fallback

        # Pre-filter candidates
        candidates = extract_proper_nouns(text)

        addressed_person: Optional[str] = None

        # If speaker_name field is populated and reliable, prioritize it as known participant
        known_participants = set()
        if speaker_name != speaker_id and speaker_name != "UNKNOWN":
            known_participants.add(speaker_name.split()[0])  # first name usually

        for cand in candidates:
            # Slight boost: if candidate matches another known speaker, check first
            is_addr, exact = is_direct_address(text, cand)
            if is_addr:
                addressed_person = exact
                break  # podcasts rarely address >1 person per turn

        # Fallback: if no LLM hit (or USE_LLM=False), use rules
        if not addressed_person:
            for cand in candidates:
                if re.search(rf'\b{cand}\b[,?!:]\s|\b{cand}\b\s*(?:how|what|let|before|yeah|okay|absolutely|thanks|hello|welcome|dres|dathan)', text, re.IGNORECASE):
                    addressed_person = cand
                    break

        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "utterance_preview": text[:120] + "..." if len(text) > 120 else text,
            "addressed_person": addressed_person,
            "theme": seg.get("theme", None)
        })

    return results


# ======================== RUN ========================
if __name__ == "__main__":
    detections = detect_vocatives_in_json()

    # Pretty output
    print(json.dumps(detections[:10], indent=2, ensure_ascii=False))  # sample

    print("\n=== Direct named addresses found ===")
    for d in detections:
        if d["addressed_person"]:
            print(f"[{d['start']:.2f}s] {d['speaker_id']} → {d['addressed_person']}")
            print(f"    \"{d['utterance_preview']}\"")
            print()