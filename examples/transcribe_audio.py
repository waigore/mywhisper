import json
import os
from typing import List, Tuple

import soundfile as sf
import torch
import torchaudio
from pywhispercpp.model import Model

from config import (
    AUDIO_SOURCE_PATH,
    EXTRACTED_AUDIO_PATH,
    MODEL_PATH,
    TRANSCRIBE_COMPLETE_FILE,
    TRANSCRIBE_DURATION_SEC,
    TRANSCRIBE_START_SEC,
    TRANSCRIBE_TARGET_SAMPLE_RATE,
    WHISPER_TIME_FACTOR,
    WHISPER_TRANSCRIPT_PATH,
)


def ensure_directories(*paths: str) -> None:
    for path in paths:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def extract_audio_chunk(
    source_path: str,
    start_sec: int,
    duration_sec: int,
    target_sample_rate: int,
    output_path: str,
) -> Tuple[torch.Tensor, int, str]:
    print(f"Extracting audio: {source_path} [{start_sec}s - {start_sec + duration_sec}s]")
    audio, samplerate = torchaudio.load(source_path)
    audio = audio[:, start_sec * samplerate:start_sec * samplerate + duration_sec * samplerate]

    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if samplerate != target_sample_rate:
        print(f"Resampling audio to {target_sample_rate} Hz...")
        resample = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=target_sample_rate)
        audio = resample(audio)
        samplerate = target_sample_rate

    audio = audio.squeeze().contiguous()
    audio_np = audio.cpu().numpy().astype("float32", copy=False)

    ensure_directories(output_path)
    sf.write(output_path, audio_np, samplerate)
    print(f"Saved extracted audio to {output_path}")

    return audio, samplerate, output_path


def load_full_audio(
    source_path: str,
    target_sample_rate: int,
) -> Tuple[torch.Tensor, int]:
    print(f"Loading full audio: {source_path}")
    audio, samplerate = torchaudio.load(source_path)

    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if samplerate != target_sample_rate:
        print(f"Resampling audio to {target_sample_rate} Hz...")
        resample = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=target_sample_rate)
        audio = resample(audio)
        samplerate = target_sample_rate

    audio = audio.squeeze().contiguous()
    return audio, samplerate


def _normalize_whisper_timestamp(value: float) -> float:
    """Convert Whisper.cpp centisecond timestamps to seconds."""
    return value / WHISPER_TIME_FACTOR


def transcribe_audio(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    whisper_model: Model,
    transcript_output_path: str,
) -> List[dict]:
    print(f"Transcribing audio ({sample_rate} Hz)...")
    audio_np = audio_tensor.cpu().numpy().astype("float32", copy=False)
    whisper_segments = whisper_model.transcribe(audio_np, language="en")

    normalized_segments: List[dict] = []
    for seg in whisper_segments:
        start_seconds = _normalize_whisper_timestamp(seg.t0)
        end_seconds = _normalize_whisper_timestamp(seg.t1)
        normalized_segments.append(
            {
                "start": start_seconds,
                "end": end_seconds,
                "text": seg.text.strip(),
            }
        )

    ensure_directories(transcript_output_path)
    with open(transcript_output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_segments, f, indent=4)
    print(f"Wrote transcript to {transcript_output_path}")

    return normalized_segments


def main() -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading Whisper model...")
    whisper_model = Model(str(MODEL_PATH), language="en")

    source_path = str(AUDIO_SOURCE_PATH)
    if TRANSCRIBE_COMPLETE_FILE:
        audio_tensor, sample_rate = load_full_audio(
            source_path,
            TRANSCRIBE_TARGET_SAMPLE_RATE,
        )
    else:
        audio_tensor, sample_rate, _ = extract_audio_chunk(
            source_path,
            TRANSCRIBE_START_SEC,
            TRANSCRIBE_DURATION_SEC,
            TRANSCRIBE_TARGET_SAMPLE_RATE,
            str(EXTRACTED_AUDIO_PATH),
        )
    segments = transcribe_audio(
        audio_tensor,
        sample_rate,
        whisper_model,
        str(WHISPER_TRANSCRIPT_PATH),
    )

    print("\n--- Whisper Transcript ---")
    for seg in segments:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s]: {seg['text']}")


if __name__ == "__main__":
    main()

