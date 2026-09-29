"""تابع پاداش v2.0 با ۵ مؤلفه + Soft Penalty"""

import numpy as np


class RewardV2:
    def __init__(self, cfg, num_edges):
        self.cfg = cfg
        self.M = num_edges

    def compute(self, state_old, state_new, action, critical_edges,
                lambda_min_old, lambda_min_new):
        # 1. R_res: normalized relative resilience
        delta_lambda = lambda_min_new - lambda_min_old
        r_res = delta_lambda / max(abs(lambda_min_old), 100.0)
        r_res = float(np.clip(r_res, -1.0, 1.0))

        # 2. R_thr: throughput ratio
        total_demand = sum(state_new['demand'])
        total_served = sum(min(d, c) for d, c in
                           zip(state_new['demand'], state_new['capacity']))
        r_thr = total_served / max(total_demand, 1.0)

        # 3. R_vio: violation ratio
        violations = sum(1 for u in state_new['utilization'] if u > self.cfg.u_th)
        r_vio = violations / self.M

        # 4. R_cost: action cost
        edge_idx, direction = action
        delta_c = self.cfg.delta_c if direction == 1 else 0.0
        r_cost = delta_c / (self.M * self.cfg.delta_c)

        # 5. R_sinr
        sinr = state_new.get('sinr', [10.0] * self.M)
        sinr_arr = np.array(sinr)
        sinr_min, sinr_max = 0.0, 30.0
        r_sinr = float(np.mean(np.clip(
            (sinr_arr - sinr_min) / (sinr_max - sinr_min), 0, 1)))

        # 6. P_soft
        p_soft = 0.0
        if self.cfg.use_soft_penalty and direction == -1:
            u, v = state_new['edges'][edge_idx]
            edge_key = (u, v) if u <= v else (v, u)
            if edge_key in critical_edges:
                p_soft = 1.0 / self.M

        R = (
            self.cfg.alpha_res * r_res
            + self.cfg.beta_thr * r_thr
            - self.cfg.gamma_vio * r_vio
            - self.cfg.delta_cost * r_cost
            + self.cfg.epsilon_sinr * r_sinr
            - self.cfg.zeta_soft * p_soft
        )

        components = {
            'R_res': r_res, 'R_thr': r_thr, 'R_vio': r_vio,
            'R_cost': r_cost, 'R_sinr': r_sinr, 'P_soft': p_soft,
            'R_total': R, 'violations': violations,
        }
        return R, components