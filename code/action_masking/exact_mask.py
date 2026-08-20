import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from topology_optimizer.gomory_hu import GomoryHuAnalyzer
from topology_optimizer.picard_cuts import PicardCutAnalyzer
# File of Code , one level upper than of action_macking folder,
#  must be adding to python Path

class ExactActionMask:
    def __init__(self, graph, gh_tree):
        self.graph = graph
        self.gh_tree = gh_tree

    def get_weakest_pair(self):
        """Find the edge with minimum weight in the Gomory-Hu tree."""
        min_weight = float('inf')
        pair = None
        for u, v, data in self.gh_tree.edges(data=True):
            w = data.get('weight', 0)
            if w < min_weight:
                min_weight = w
                pair = (u, v)
        return pair, min_weight

    def build_mask(self):
        # source, target, mincut = *self.get_weakest_pair(), None
        
        # Actually get_weakest_pair returns (pair, weight); adjust:
        pair, mincut = self.get_weakest_pair()
        source, target = pair

        # Use Picard to find original edges crossing the weakest cut
        picard = PicardCutAnalyzer(self.graph)
        bottleneck_edges, _ = picard.analyze_min_cuts(source, target)

        all_edges = list(self.graph.edges())
        mask = []
        for e in all_edges:
            # Edges are undirected; check both directions
            if e in bottleneck_edges or (e[1], e[0]) in bottleneck_edges:
                mask.append(True)
            else:
                mask.append(False)

        return mask, all_edges, (source, target, mincut)


if __name__ == "__main__":
    G = nx.Graph()
    sample = [
        (1, 2, 40), (1, 3, 20),
        (2, 3, 20), (2, 4, 30),
        (3, 4, 10), (4, 5, 50),
        (3, 5, 10)
    ]
    for u, v, c in sample:
        G.add_edge(u, v, capacity=c)

    # Build GH tree
    analyzer = GomoryHuAnalyzer()
    analyzer.load_topology(sample)
    gh = analyzer.build_gomory_hu_tree()

    # Build action mask
    masker = ExactActionMask(G, gh)
    mask, all_edges, info = masker.build_mask()

    print("Weakest pair and mincut:", info)
    print("All edges:", all_edges)
    print("Mask:", mask)
    print("Allowed edges:", [e for e, m in zip(all_edges, mask) if m])