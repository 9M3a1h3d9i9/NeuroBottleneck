"""ماژول Gomory-Hu: محاسبه درخت، λ_min و CriticalEdges با نگاشت صحیح"""

import networkx as nx
import numpy as np
from typing import Tuple, List, Set


class GHModule:
    def __init__(self, threshold_mult: float = 1.5, cache_enabled: bool = True):
        self.threshold_mult = threshold_mult
        self.cache_enabled = cache_enabled
        self._cache = None

    def compute(self, G: nx.Graph) -> Tuple[float, float, Set[tuple]]:
        """
        خروجی:
            lambda_min: حداقل ظرفیت برش
            lambda_max: حداکثر ظرفیت برش (sum of c_eff)
            critical_edges: مجموعه یال‌های بحرانی (فیزیکی)
        """
        if self.cache_enabled and self._cache is not None:
            return self._cache

        # محاسبه ظرفیت مؤثر
        G_eff = nx.Graph()
        for u, v, d in G.edges(data=True):
            c = d.get('capacity', 1000.0)
            u_util = d.get('utilization', 0.4)
            c_eff = c * (1 - u_util)
            G_eff.add_edge(u, v, capacity=max(c_eff, 1.0))

        # درخت Gomory-Hu
        try:
            gh_tree = nx.gomory_hu_tree(G_eff, capacity='capacity')
        except Exception:
            lambda_min = float(min(d['capacity'] for _, _, d in G_eff.edges(data=True))) if G_eff.number_of_edges() > 0 else 0.0
            return lambda_min, lambda_min * 2, set()

        weights = [gh_tree[u][v]['weight'] for u, v in gh_tree.edges()]
        if not weights:
            return 0.0, 0.0, set()

        lambda_min = float(min(weights))
        lambda_max = float(sum(d['capacity'] for _, _, d in G_eff.edges(data=True)))
        threshold = self.threshold_mult * lambda_min

        # استخراج CriticalEdges با نگاشت صحیح
        critical_edges = set()
        for u, v in gh_tree.edges():
            w = gh_tree[u][v]['weight']
            if w <= threshold:
                gh_copy = gh_tree.copy()
                gh_copy.remove_edge(u, v)
                comps = list(nx.connected_components(gh_copy))
                if len(comps) >= 2:
                    S = comps[0]
                    T = set().union(*comps[1:])
                    for a, b in G.edges():
                        if (a in S and b in T) or (a in T and b in S):
                            critical_edges.add((a, b) if a <= b else (b, a))

        result = (lambda_min, lambda_max, critical_edges)
        if self.cache_enabled:
            self._cache = result
        return result

    def invalidate_cache(self):
        self._cache = None