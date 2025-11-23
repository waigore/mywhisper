import networkx as nx
from networkx.algorithms.bipartite import maximum_matching
from collections import defaultdict
import re

# Same extract_vocative and transcription as above

# Collect edges: speaker -> vocative with weights
edges = defaultdict(int)
for segment in transcription:
    voc = extract_vocative(segment["text"])
    if voc:
        edges[(segment["speaker"], voc)] += 1

# Create bipartite graph
G = nx.Graph()
speakers = set(seg["speaker"] for seg in transcription)
vocatives = set(v for _, v in edges.keys())
G.add_nodes_from(speakers, bipartite=0)
G.add_nodes_from(vocatives, bipartite=1)
for (spk, voc), weight in edges.items():
    G.add_edge(spk, voc, weight=weight)

# Find maximum matching (assign vocatives to speakers)
matching = maximum_matching(G)  # Note: This is unweighted; for weighted, use min_cost_flow or similar

# Confidence: For each match, compute weight / total weights to that voc
assignments = {}
for spk in speakers:
    if spk in matching:
        voc = matching[spk]
        total_weight = sum(G[spk][n]['weight'] for n in G.neighbors(spk))
        conf = G[spk][voc]['weight'] / total_weight if total_weight > 0 else 0.0
        assignments[voc] = (spk, conf)

print("Graph-Based Assignments:")
for voc, (spk, conf) in assignments.items():
    print(f"'{voc}' matches to {spk} with confidence {conf:.2f}")