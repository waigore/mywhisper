# --------------------------------------------------------------
#  FULL REUSABLE EMBEDDINGS SCRIPT
# --------------------------------------------------------------
import os, joblib, torch, numpy as np, torchaudio
from typing import Tuple, List, Dict
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from pyannote.audio import Pipeline, Inference
from pyannote.core import Annotation, Segment
from huggingface_hub import login

from config import (
    DIARIZE_AUDIO_PATH,
    DIARIZE_CHUNK_DIR,
    DIARIZE_CHUNK_MINUTES,
    DIARIZE_CLUSTER_PKL,
    DIARIZE_NUM_SPEAKERS,
    DIARIZE_OUTPUT_RTTM,
    DIARIZE_OVERLAP_SECONDS,
    DIARIZE_RTTM_DIR,
    HF_TOKEN,
)

DIARIZE_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
DIARIZE_RTTM_DIR.mkdir(parents=True, exist_ok=True)
login(token=HF_TOKEN)

# ------------------- 1. SPLIT -------------------
def split_into_chunks(audio_path, chunk_min, overlap_sec):
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = chunk_min*60*1000
    overlap_ms = overlap_sec*1000
    chunks = []; start_ms = 0; idx = 0
    while start_ms < len(audio):
        end_ms = min(start_ms + chunk_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        out = DIARIZE_CHUNK_DIR/f"chunk_{idx:03d}.wav"
        chunk.export(out, format="wav")
        chunks.append({"file":out, "global_start":start_ms/1000.0,
                       "global_end":end_ms/1000.0, "idx":idx})
        start_ms += chunk_ms - overlap_ms
        idx += 1
    return chunks
chunks = split_into_chunks(str(DIARIZE_AUDIO_PATH), DIARIZE_CHUNK_MINUTES, DIARIZE_OVERLAP_SECONDS)

# ------------------- 2. MODELS -------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                   use_auth_token=HF_TOKEN)
pipeline.to(device)
embedding_inference = Inference("pyannote/embedding",
                                device=device,
                                use_auth_token=HF_TOKEN,
                                window="whole")
MIN_EMB_DURATION = getattr(embedding_inference, "duration", None) or 2.0
_duration_cache: Dict[str, float] = {}


def get_duration_seconds(wav: str) -> float:
    wav = str(wav)
    if wav not in _duration_cache:
        info = torchaudio.info(wav)
        _duration_cache[wav] = info.num_frames / info.sample_rate
    return _duration_cache[wav]


def ensure_min_duration(seg: Segment, min_duration: float, file_duration: float) -> Segment:
    if seg.duration >= min_duration:
        return seg
    center = seg.middle
    half = min_duration / 2.0
    start = max(0.0, center - half)
    end = min(file_duration, center + half)
    if end - start < min_duration:
        if start == 0.0:
            end = min(file_duration, min_duration)
        elif end == file_duration:
            start = max(0.0, file_duration - min_duration)
        else:
            remaining = min_duration - (end - start)
            start = max(0.0, start - remaining / 2.0)
            end = min(file_duration, end + remaining / 2.0)
    return Segment(start, end)

# ------------------- 3. REFERENCE -------------------
ref = chunks[0]
ref_diar = pipeline(str(ref["file"]), num_speakers=DIARIZE_NUM_SPEAKERS)

def count_speakers(ann: Annotation) -> int:
    return len({label for _, _, label in ann.itertracks(yield_label=True)})

TARGET_NUM_SPEAKERS = DIARIZE_NUM_SPEAKERS or count_speakers(ref_diar)
if not TARGET_NUM_SPEAKERS:
    raise ValueError("Unable to determine number of speakers from reference chunk.")


def extract_embs(wav: str, ann: Annotation) -> Tuple[np.ndarray, List[Segment]]:
    embeddings, segments = [], []
    file_duration = get_duration_seconds(wav)
    for seg, _, _ in ann.itertracks(yield_label=True):
        padded = ensure_min_duration(seg, MIN_EMB_DURATION, file_duration)
        emb = embedding_inference.crop(wav, padded)
        if isinstance(emb, tuple):
            emb = emb[0]
        embeddings.append(np.asarray(emb).squeeze())
        segments.append(seg)
    if not embeddings:
        raise ValueError("No segments found to extract embeddings.")
    return np.stack(embeddings), segments

ref_embs, _ = extract_embs(str(ref["file"]), ref_diar)

clusterer = AgglomerativeClustering(n_clusters=TARGET_NUM_SPEAKERS,
                                    metric='cosine', linkage='average')
clusterer.fit(ref_embs)
joblib.dump({"clusterer":clusterer,
             "ref_labels":clusterer.labels_,
             "ref_embs":ref_embs,
             "num_speakers":TARGET_NUM_SPEAKERS}, DIARIZE_CLUSTER_PKL)

# ------------------- 4. PROCESS ALL -------------------
saved = joblib.load(DIARIZE_CLUSTER_PKL)
clusterer = saved["clusterer"]
ref_labels = saved["ref_labels"]
ref_embs = saved["ref_embs"]
TARGET_NUM_SPEAKERS = saved["num_speakers"]
unique_labels = np.unique(ref_labels)
centroids = np.stack([ref_embs[ref_labels == label].mean(0) for label in unique_labels])

global_diar = Annotation()

for c in tqdm(chunks, desc="Chunk"):
    wav = str(c["file"])
    gs = c["global_start"]
    seg_ann = pipeline(wav, num_speakers=DIARIZE_NUM_SPEAKERS or TARGET_NUM_SPEAKERS)
    embs, turns = extract_embs(wav, seg_ann)

    dists = cosine_distances(embs, centroids)
    spk_ids = dists.argmin(axis=1)

    for turn, sid in zip(turns, spk_ids):
        gseg = Segment(turn.start+gs, turn.end+gs)
        global_diar[gseg] = f"SPEAKER_{sid:02d}"

# ------------------- 5. SAVE -------------------
with open(DIARIZE_OUTPUT_RTTM, "w") as f: global_diar.write_rttm(f)
print("Done →", DIARIZE_OUTPUT_RTTM)