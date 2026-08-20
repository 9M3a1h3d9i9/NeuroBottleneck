import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sndlib_parser import parse_sndlib
from topology_optimizer.gomory_hu import GomoryHuAnalyzer
from action_masking.exact_mask import ExactActionMask

G = parse_sndlib("data/sndlib/india35.txt")
edges = [(u, v, d['capacity']) for u, v, d in G.edges(data=True)]

analyzer = GomoryHuAnalyzer()
analyzer.load_topology(edges)
gh_tree = analyzer.build_gomory_hu_tree()

masker = ExactActionMask(G, gh_tree)
mask, all_edges, info = masker.build_mask()

print("Weakest pair and mincut:", info)
print("Total edges:", len(all_edges))
print("Allowed edges:", sum(mask))
print("Allowed edges list:", [e for e, m in zip(all_edges, mask) if m])