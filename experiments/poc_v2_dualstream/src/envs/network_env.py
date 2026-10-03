"""محیط MDP کامل NeuroBottleneck v2.0 — Dual-Stream"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

from ..models.gh_module import GHModule
from ..rewards.reward_v2 import RewardV2
from ..masks.mask_module import MaskModule
from .topologies import load_topology


class NetworkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_cfg, gh_cfg, reward_cfg,
                 topology_name="abilene", seed=42):
        super().__init__()
        self.cfg = env_cfg
        self.gh_cfg = gh_cfg
        self.reward_cfg = reward_cfg
        self.topology_name = topology_name

        self.G = load_topology(topology_name)
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        self.M = len(self.edges)
        self.N = len(self.nodes)

        self.gh_module = GHModule(threshold_mult=env_cfg.lambda_threshold_mult)
        # self.reward_fn = RewardV2(reward_cfg, self.M)
        self.reward_fn = RewardV2(reward_cfg, self.M, u_th=env_cfg.u_th) # Modified
        
        self.mask_module = MaskModule(reward_cfg, self.M)

        self.action_space = spaces.Discrete(2 * self.M)

        self.node_dim = 3
        self.edge_dim = 3
        self.gh_raw_dim = 4 * self.M
        obs_dim = self.N * self.node_dim + self.M * self.edge_dim + self.gh_raw_dim
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self.edge_index = self._build_edge_index()

        self.capacities = None
        self.demands = None
        self.utilizations = None
        self.sinr = None
        self.step_count = 0
        self.lambda_min = 0.0
        self.lambda_max = 1.0
        self.critical_edges = set()
        self._last_gh_step = -1

    def _build_edge_index(self):
        node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        src, dst = [], []
        for u, v in self.edges:
            i, j = node_to_idx[u], node_to_idx[v]
            src.extend([i, j])
            dst.extend([j, i])
        return np.array([src, dst], dtype=np.int64)

    def _get_node_features(self):
        feats = np.zeros((self.N, self.node_dim), dtype=np.float32)
        for i, n in enumerate(self.nodes):
            deg = self.G.degree(n) / max(self.N, 1)
            feats[i] = [0.5, 0.5, deg]
        return feats

    def _get_edge_features(self):
        feats = np.zeros((self.M, self.edge_dim), dtype=np.float32)
        for i, (u, v) in enumerate(self.edges):
            d = self.G[u][v]
            feats[i] = [
                self.capacities[i] / self.cfg.c_max,
                self.utilizations[i],
                d.get('delay', 10.0) / 100.0,
            ]
        return feats

    def _get_gh_features(self):
        feats = np.zeros((self.M, 4), dtype=np.float32)
        for i, (u, v) in enumerate(self.edges):
            edge_key = (u, v) if u <= v else (v, u)
            is_crit = 1.0 if edge_key in self.critical_edges else 0.0
            feats[i] = [
                self.lambda_min / max(self.cfg.c_max, 1.0),
                self.lambda_max / max(self.cfg.c_max * self.M, 1.0),
                self.lambda_min / max(self.lambda_max, 1.0),
                is_crit,
            ]
        return feats

    def _get_obs(self):
        node_feats = self._get_node_features().flatten()
        edge_feats = self._get_edge_features().flatten()
        gh_feats = self._get_gh_features().flatten()
        obs = np.concatenate([node_feats, edge_feats, gh_feats]).astype(np.float32)
        return obs

    def _update_gh(self, force=False):
        if force or (self.step_count - self._last_gh_step >= self.gh_cfg.update_interval):
            for i, (u, v) in enumerate(self.edges):
                self.G[u][v]['capacity'] = float(self.capacities[i])
                self.G[u][v]['utilization'] = float(self.utilizations[i])

            self.gh_module.invalidate_cache()
            self.lambda_min, self.lambda_max, self.critical_edges = \
                self.gh_module.compute(self.G)
            self._last_gh_step = self.step_count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.capacities = np.full(self.M, self.cfg.c_init, dtype=np.float32)
        self.demands = np.full(self.M, self.cfg.d_init, dtype=np.float32)
        self.utilizations = self.demands / self.capacities
        self.sinr = np.random.uniform(10, 25, self.M).astype(np.float32)

        num_crit = min(3, self.M)
        crit_idx = np.random.choice(self.M, num_crit, replace=False)
        for i in crit_idx:
            self.capacities[i] = np.random.uniform(500, 800)

        self.utilizations = self.demands / self.capacities
        self.step_count = 0
        self._last_gh_step = -1
        self._update_gh(force=True)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        edge_idx = action // 2
        direction = 1 if action % 2 == 1 else -1

        lambda_old = self.lambda_min

        delta = self.cfg.delta_c * direction
        self.capacities[edge_idx] = np.clip(
            self.capacities[edge_idx] + delta, self.cfg.c_min, self.cfg.c_max
        )

        d_target = self.cfg.d_init
        noise = np.random.normal(0, self.cfg.sigma_d, self.M)
        self.demands = np.maximum(
            self.cfg.d_min,
            self.demands + self.cfg.alpha_revert * (d_target - self.demands) + noise
        )

        self.utilizations = np.clip(self.demands / self.capacities, 0.01, 1.0)
        self.sinr = np.clip(self.sinr + np.random.normal(0, 0.5, self.M), 0, 30)

        self._update_gh(force=False)

        state_new = {
            'capacity': self.capacities,
            'demand': self.demands,
            'utilization': self.utilizations,
            'sinr': self.sinr,
            'edges': self.edges,
        }
        R, components = self.reward_fn.compute(
            None, state_new, (edge_idx, direction),
            self.critical_edges, lambda_old, self.lambda_min
        )

        terminated = self.step_count >= self.cfg.episode_len
        truncated = False

        info = {
            'components': components,
            'lambda_min': self.lambda_min,
            'lambda_max': self.lambda_max,
            'critical_edges': self.critical_edges,
            'violations': components['violations'],
            'action': action,
        }

        return self._get_obs(), float(R), terminated, truncated, info

    def get_action_mask(self):
        return self.mask_module.get_mask(
            self.capacities, self.edges, self.critical_edges
        )