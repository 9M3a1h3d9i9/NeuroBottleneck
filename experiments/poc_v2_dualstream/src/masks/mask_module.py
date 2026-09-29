"""ماژول ماسک: Hard و Soft"""

import numpy as np


class MaskModule:
    def __init__(self, cfg, num_edges):
        self.cfg = cfg
        self.M = num_edges

    def telecom_mask(self, capacities):
        mask = np.ones(2 * self.M, dtype=np.int8)
        for i, c in enumerate(capacities):
            if c <= self.cfg.c_min + self.cfg.delta_c:
                mask[2 * i] = 0
            if c >= self.cfg.c_max - self.cfg.delta_c:
                mask[2 * i + 1] = 0
        return mask

    def gh_hard_mask(self, capacities, edges, critical_edges):
        mask = self.telecom_mask(capacities)
        for i, (u, v) in enumerate(edges):
            edge_key = (u, v) if u <= v else (v, u)
            if edge_key in critical_edges:
                mask[2 * i] = 0
        return mask

    def get_mask(self, capacities, edges, critical_edges):
        if self.cfg.use_hard_mask:
            return self.gh_hard_mask(capacities, edges, critical_edges)
        return self.telecom_mask(capacities)