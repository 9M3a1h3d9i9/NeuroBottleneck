"""تابع پاداش v2.0 با ۵ مؤلفه + Soft Penalty"""

import numpy as np

class RewardV2:
    def __init__(self, cfg, num_edges, u_th=0.8, delta_c=200.0):
        """
        Args:
            cfg: RewardConfig
            num_edges: تعداد یال‌ها
            u_th: آستانه اشباع
            delta_c: گام تغییر ظرفیت
        """
        self.cfg = cfg
        self.M = num_edges
        self.u_th = u_th
        self.delta_c = delta_c

    def compute(self, state_old, state_new, action, critical_edges,
                lambda_min_old, lambda_min_new):
        # 1. R_res
        delta_lambda = lambda_min_new - lambda_min_old
        r_res = delta_lambda / max(abs(lambda_min_old), 100.0)
        r_res = float(np.clip(r_res, -1.0, 1.0))

        # 2. R_thr
        total_demand = sum(state_new['demand'])
        total_served = sum(
            min(d, c) for d, c in
            zip(state_new['demand'], state_new['capacity'])
        )
        r_thr = total_served / max(total_demand, 1.0)

        # 3. R_vio
        violations = sum(1 for u in state_new['utilization'] if u > self.u_th)
        r_vio = violations / self.M

        # 4. R_cost (عملا هزینه ساختاری می باشد)
        edge_idx, direction = action
        
        # delta_c_actual = self.delta_c if direction == 1 else 0.0
        # r_cost = delta_c_actual / (self.M * self.delta_c) # این همیشه 0.066 است!

            # Note:
                # Even if you set its weight to 0.20, the maximum penalty 
                # is 0.20 * 0.066 = 0.013, which is negligible compared to
                #  the reward of 0.40 gained by increasing λ min. 
                # DQN will still cheat.

        if direction == 1:
            # هزینه افزایش ظرفیت بر اساس ظرفیت فعلی یال 
            # (یال‌های کم‌ظرفیت گران‌ترند)
            # و اگر یال در CriticalEdges باشد، هزینه دوبرابر است

            u, v = state_new['edges'][edge_idx]
            edge_key = (u, v) if u <= v else (v, u)
            base_cost = self.delta_c / max(state_new['capacity'][edge_idx], 1.0)
            if edge_key in critical_edges:
                base_cost *= 2.0
            r_cost = base_cost
        else:
            r_cost = 0.0

        # Note:
            # Cost of increasing capacity based on the edge's current capacity 
            # (lower-capacity edges are more expensive)
            # and if the edge is in CriticalEdges, the cost is doubled


        # 5. R_sinr
        sinr = state_new.get('sinr', [10.0] * self.M)
        sinr_arr = np.array(sinr)
        r_sinr = float(np.mean(np.clip(sinr_arr / 30.0, 0, 1)))

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
