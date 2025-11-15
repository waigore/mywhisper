# --------------------------------------------------------------
#  PYANNOTE COMMUNITY DIARIZATION SCRIPT (FULL AUDIO)
# --------------------------------------------------------------
import torch
import torchaudio
from typing import Dict, Union

from huggingface_hub import login
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation
from tqdm.auto import tqdm

from config import (
    DIARIZE_AUDIO_PATH,
    DIARIZE_MODEL,
    DIARIZE_NUM_SPEAKERS,
    DIARIZE_OUTPUT_RTTM,
    DIARIZE_RTTM_DIR,
    HF_TOKEN,
)

DIARIZE_RTTM_DIR.mkdir(parents=True, exist_ok=True)
login(token=HF_TOKEN)


class TqdmProgressHook(ProgressHook):
    """
    Thin wrapper around pyannote's default progress hook so we can surface
    the internal pipeline steps through tqdm instead of rich Progress.
    """

    def __init__(self) -> None:
        super().__init__(hidden=True)
        self._bars: Dict[str, "tqdm"] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()
        return False

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        # fall back to a single step when pyannote does not provide totals
        total = total or completed or 1
        completed = completed or total

        bar = self._bars.get(step_name)
        if bar is None:
            bar = tqdm(total=total, desc=f"{step_name}", unit="step", leave=True)
            self._bars[step_name] = bar
        else:
            if bar.total != total:
                bar.total = total

        delta = completed - bar.n
        if delta > 0:
            bar.update(delta)

        if completed >= total:
            bar.close()
            self._bars.pop(step_name, None)


# ------------------- PIPELINE -------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

pipeline = Pipeline.from_pretrained(
    DIARIZE_MODEL,
    token=HF_TOKEN,
)
print("Pipeline loaded")
print(f"Using device: {device}")
pipeline.to(device)

pipeline_kwargs: Dict[str, int] = {}
if DIARIZE_NUM_SPEAKERS:
    pipeline_kwargs["num_speakers"] = DIARIZE_NUM_SPEAKERS


def load_waveform(path: str, target_sample_rate: int = 16000) -> Dict[str, Union[torch.Tensor, int]]:
    waveform, sample_rate = torchaudio.load(path)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return {
        "waveform": waveform.contiguous(),
        "sample_rate": target_sample_rate,
    }

# ------------------- RUN FULL AUDIO -------------------
with TqdmProgressHook() as hook:
    diarization = pipeline(
        load_waveform(str(DIARIZE_AUDIO_PATH)),
        hook=hook,
        **pipeline_kwargs,
    )
seg_ann = (
    diarization.speaker_diarization
    if hasattr(diarization, "speaker_diarization")
    else diarization
)

global_diar = Annotation()
tracks = list(seg_ann.itertracks(yield_label=True))
for turn, _, label in tqdm(tracks, desc="Aggregating segments", unit="seg"):
    global_diar[turn] = label

# ------------------- SAVE -------------------
with open(DIARIZE_OUTPUT_RTTM, "w") as f:
    global_diar.write_rttm(f)

print("Done →", DIARIZE_OUTPUT_RTTM)