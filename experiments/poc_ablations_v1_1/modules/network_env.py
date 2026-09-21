# ============================================================================
# محیط NeuroBottleneck - نسخه ۱.۱ (فیزیک صحیح)
# ============================================================================
# این محیط، دقیقاً طبق فرمول‌های سند v1.1 پیاده‌سازی شده است.
# ============================================================================

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces

from .action_masking import DualActionMasker
from .gomory_hu_module import GomoryHuAnalyzer


class NetworkEnvV11(gym.Env):
    """
    محیط NeuroBottleneck v1.1
    
    فرمول‌های کلیدی:
      - U(e) = d(e) / c(e)
      - c_e^eff = c_e * (1 - U_e)
      - Reward = α·R_res + β·R_thr - γ·R_vio - δ·R_cost + ε·R_sinr
    """
    
    metadata = {"render_modes": ["human"]}
    
    # ثابت‌های فیزیکی (طبق سند)
    CAPACITY_MIN = 300.0
    CAPACITY_MAX = 5000.0
    CAPACITY_INIT = 2000.0
    CAPACITY_STEP = 200.0
    
    DEMAND_MIN = 100.0
    DEMAND_MAX = 4500.0
    DEMAND_NOISE = 50.0
    
    UTILIZATION_THRESHOLD = 0.8
    EPISODE_LENGTH = 50
    
    # ضرایب پاداش (طبق سند v1.1)
    ALPHA = 0.30   # تاب‌آوری
    BETA = 0.30    # توان عملیاتی
    GAMMA = 0.20   # تخلفات
    DELTA = 0.10   # هزینه
    EPSILON = 0.10 # SINR
    
    def __init__(self, graph, seed=42):
        super().__init__()
        
        self.graph = graph
        self.edges = list(graph.edges())
        self.n_edges = len(self.edges)
        self.n_nodes = graph.number_of_nodes()
        
        # محیط seed
        np.random.seed(seed)
        
        # === فضای اقدام ===
        # action = 2 * edge_idx + direction (0: کاهش، 1: افزایش)
        self.action_space = spaces.Discrete(2 * self.n_edges)
        
        # === فضای مشاهده ===
        # برای هر یال: [c_norm, d_norm, u, sinr_norm] + [lambda_min_norm]
        self.obs_dim = self.n_edges * 4 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        # === ماسک‌ساز و تحلیل‌گر ===
        self.masker = DualActionMasker(
            capacity_min=self.CAPACITY_MIN,
            capacity_max=self.CAPACITY_MAX,
            capacity_step=self.CAPACITY_STEP,
        )
        self.gh_analyzer = GomoryHuAnalyzer(cache_interval=5)
        
        # === شمارنده‌ها ===
        self.step_count = 0
        self.episode_violations = []
        self.episode_rewards = []
    
    def _get_obs(self):
        """ساخت بردار مشاهده نرمال‌شده"""
        obs = []
        for u, v in self.edges:
            cap = self.graph[u][v].get('capacity', self.CAPACITY_INIT)
            dem = self.graph[u][v].get('demand', 1000.0)
            util = dem / max(cap, 1e-6)
            sinr = self.graph[u][v].get('sinr', 15.0)
            
            # نرمال‌سازی
            obs.append(cap / self.CAPACITY_MAX)
            obs.append(dem / self.DEMAND_MAX)
            obs.append(min(util, 1.0))
            obs.append(np.clip((sinr - 0) / 30, 0, 1))
        
        # λ_min نرمال‌شده
        gh_result = self.gh_analyzer.compute(self.graph, step=self.step_count)
        lambda_min = gh_result['lambda_min']
        lambda_max = max(gh_result['lambda_max'], 1e-6)
        obs.append(np.clip(lambda_min / lambda_max, 0, 1))
        
        return np.array(obs, dtype=np.float32)
    
    def _count_violations(self):
        """تعداد لینک‌های اشباع‌شده"""
        count = 0
        for u, v in self.edges:
            cap = self.graph[u][v].get('capacity', 1.0)
            dem = self.graph[u][v].get('demand', 0.0)
            util = dem / max(cap, 1e-6)
            self.graph[u][v]['utilization'] = util
            if util > self.UTILIZATION_THRESHOLD:
                count += 1
        return count
    
    def _compute_reward(self, lambda_min_old, lambda_min_new, 
                         n_violations, cost):
        """
        تابع پاداش اصلاح‌شده طبق سند v1.1
        R = α·R̃_res + β·R̃_thr - γ·R̃_vio - δ·R̃_cost + ε·R̃_sinr
        """
        lambda_max = max(self.gh_analyzer._cached_lambda_max or 1.0, 1e-6)
        
        # R̃_res: افزایش تاب‌آوری
        R_res = (lambda_min_new - lambda_min_old) / (lambda_max + 1e-6)
        R_res = np.clip(R_res, -1, 1)
        
        # R̃_thr: نسبت تقاضای پذیرفته‌شده
        total_demand = sum(self.graph[u][v]['demand'] for u, v in self.edges)
        total_accepted = sum(min(self.graph[u][v]['demand'], 
                                  self.graph[u][v]['capacity']) 
                             for u, v in self.edges)
        # R_thr = total_accepted / max(total_demand, 1e-6)
        R_thr = np.clip(total_accepted / max(total_demand, 1e-6), 0, 1)

        
        # R̃_vio: نرخ تخلف نرمال‌شده
        R_vio = n_violations / max(self.n_edges, 1)
        
        # R̃_cost: هزینه نرمال‌شده
        R_cost = cost / (self.n_edges * self.CAPACITY_STEP)
        
        # R̃_sinr: میانگین SINR نرمال‌شده
        sinrs = [self.graph[u][v].get('sinr', 15.0) for u, v in self.edges]
        R_sinr = np.mean([np.clip((s - 0) / 30, 0, 1) for s in sinrs])
        
        reward = (self.ALPHA * R_res 
                  + self.BETA * R_thr 
                  - self.GAMMA * R_vio 
                  - self.DELTA * R_cost 
                  + self.EPSILON * R_sinr)
        
        return float(reward)
    
    def get_action_mask(self):
        """API رسمی برای MaskablePPO"""
        return self.masker.get_mask(self.graph, self.edges, step=self.step_count)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # بازنشانی ظرفیت و تقاضا
        for u, v in self.edges:
            cap = self.CAPACITY_INIT  # 2000
            # تقاضا بین ۳۰٪ تا ۷۰٪ ظرفیت
            dem = np.random.uniform(0.3 * cap, 0.7 * cap)  # 600-1400
            self.graph[u][v]['capacity'] = cap
            self.graph[u][v]['demand'] = dem
            self.graph[u][v]['utilization'] = dem / cap
            self.graph[u][v]['sinr'] = np.random.uniform(10, 30)
        
        self.step_count = 0
        self.episode_violations = []
        self.episode_rewards = []
        
        # force بازمحاسبه GH
        self.gh_analyzer.compute(self.graph, step=0, force=True)
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        
        # === دیکد اقدام ===
        edge_idx = action // 2
        direction = action % 2
        u, v = self.edges[edge_idx]
        
        # ذخیره λ_min قبل
        lambda_min_old = self.gh_analyzer.compute(
            self.graph, step=self.step_count - 1
        )['lambda_min']
        
        # === اعمال تغییر ظرفیت ===
        delta = self.CAPACITY_STEP if direction == 1 else -self.CAPACITY_STEP
        new_cap = self.graph[u][v]['capacity'] + delta
        new_cap = np.clip(new_cap, self.CAPACITY_MIN, self.CAPACITY_MAX)
        actual_delta = abs(new_cap - self.graph[u][v]['capacity'])
        self.graph[u][v]['capacity'] = new_cap
        
        # === پویایی تقاضا ===
        for e_u, e_v in self.edges:
            noise = np.random.normal(0, self.DEMAND_NOISE)
            new_dem = self.graph[e_u][e_v]['demand'] + noise
            new_dem = np.clip(new_dem, self.DEMAND_MIN, self.DEMAND_MAX)
            self.graph[e_u][e_v]['demand'] = new_dem
            self.graph[e_u][e_v]['utilization'] = (
                new_dem / max(self.graph[e_u][e_v]['capacity'], 1e-6)
            )
        
        # === محاسبه تخلفات ===
        n_violations = self._count_violations()
        self.episode_violations.append(n_violations)
        
        # === بازمحاسبه GH (با caching) ===
        gh_result_new = self.gh_analyzer.compute(
            self.graph, step=self.step_count, force=True
        )
        lambda_min_new = gh_result_new['lambda_min']
        
        # === محاسبه پاداش ===
        reward = self._compute_reward(
            lambda_min_old, lambda_min_new,
            n_violations, actual_delta
        )
        self.episode_rewards.append(reward)
        
        # === بررسی پایان ===
        terminated = self.step_count >= self.EPISODE_LENGTH
        truncated = False
        
        info = {
            'violations': n_violations,
            'lambda_min': lambda_min_new,
            'reward': reward,
            'step': self.step_count,
        }
        
        return self._get_obs(), reward, terminated, truncated, info