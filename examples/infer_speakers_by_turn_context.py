from collections import defaultdict
import re
import math

# Same extract_vocative and transcription

# Score matrix: vocative -> speaker -> score
scores = defaultdict(lambda: defaultdict(float))

for i, segment in enumerate(transcription):
    voc = extract_vocative(segment["text"])
    if voc and i + 1 < len(transcription):
        next_spk = transcription[i + 1]["speaker"]
        scores[voc][next_spk] += 1.0  # Boost for next speaker
    if voc and i > 0:
        prev_spk = transcription[i - 1]["speaker"]
        scores[voc][prev_spk] += 0.5  # Lesser boost for previous (e.g., response to address)

# For each voc, normalize scores to probabilities (confidence)
assignments = {}
for voc, spk_scores in scores.items():
    total = sum(spk_scores.values())
    if total > 0:
        probs = {spk: score / total for spk, score in spk_scores.items()}
        best_spk = max(probs, key=probs.get)
        conf = probs[best_spk]
        # Apply softmax for better distribution if needed
        exp_scores = {spk: math.exp(score) for spk, score in spk_scores.items()}
        softmax_total = sum(exp_scores.values())
        softmax_conf = exp_scores[best_spk] / softmax_total
        assignments[voc] = (best_spk, softmax_conf)
    else:
        assignments[voc] = (None, 0.0)

print("Turn-Taking Assignments:")
for voc, (spk, conf) in assignments.items():
    print(f"'{voc}' likely refers to {spk} with confidence {conf:.2f}")