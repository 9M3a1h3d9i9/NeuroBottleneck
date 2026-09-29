"""
Dual-Stream Feature Extractor برای SB3
- مسیر ۱: GNN روی گراف
- مسیر ۲: GH features
- خروجی: concatenation
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from .gnn_encoders import build_gnn


class DualStreamExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, node_dim, edge_dim, num_nodes, num_edges,
                 gnn_type="graphsage", gnn_hidden=64, gnn_out=64, gh_out=16,
                 use_gnn=True, use_gh_feature=True, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)
        self.use_gnn = use_gnn
        self.use_gh_feature = use_gh_feature
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # GH features: 4 per edge
        self.gh_raw_dim = 4 * num_edges
        self.gh_mlp = nn.Sequential(
            nn.Linear(self.gh_raw_dim, 64), nn.ReLU(),
            nn.Linear(64, gh_out),
        )
        self.gh_norm = nn.LayerNorm(gh_out)

        if use_gnn:
            self.gnn = build_gnn(gnn_type, node_dim, edge_dim, gnn_hidden, gnn_out)
            gnn_out_dim = gnn_out
        else:
            self.gnn = None
            gnn_out_dim = 0

        gh_dim = gh_out if use_gh_feature else 0

        if gnn_out_dim + gh_dim == 0:
            self.fallback = nn.Sequential(
                nn.Linear(observation_space.shape[0], 128), nn.ReLU(),
                nn.Linear(128, features_dim),
            )
            self.total_dim = features_dim
        else:
            self.total_dim = gnn_out_dim + gh_dim
            self.final_proj = nn.Sequential(
                nn.Linear(self.total_dim, features_dim),
                nn.LayerNorm(features_dim),
            )

    def _unpack_obs(self, obs):
        B = obs.size(0)
        n_dim = self.node_dim
        e_dim = self.edge_dim

        node_end = self.num_nodes * n_dim
        edge_end = node_end + self.num_edges * e_dim
        gh_end = edge_end + self.gh_raw_dim

        node_feats = obs[:, :node_end].reshape(B, self.num_nodes, n_dim)
        edge_feats = obs[:, node_end:edge_end].reshape(B, self.num_edges, e_dim)
        gh_feats = obs[:, edge_end:gh_end]

        return node_feats, edge_feats, gh_feats

    def forward(self, obs):
        B = obs.size(0)
        node_feats, edge_feats, gh_feats = self._unpack_obs(obs)

        if self.gnn is None and not self.use_gh_feature:
            return self.fallback(obs)

        streams = []

        if self.use_gnn and self.gnn is not None:
            z_gnn_list = []
            for b in range(B):
                edge_index = self._get_edge_index(obs.device)
                z = self.gnn(node_feats[b], edge_index, edge_feats[b])
                z_gnn_list.append(z)
            z_gnn = torch.cat(z_gnn_list, dim=0)
            streams.append(z_gnn)

        if self.use_gh_feature:
            z_gh = self.gh_norm(self.gh_mlp(gh_feats))
            streams.append(z_gh)

        z = torch.cat(streams, dim=-1)
        return self.final_proj(z)

    def _get_edge_index(self, device):
        """edge_index پیش‌فرض — برای Abilene"""
        edges = []
        n = self.num_nodes
        for i in range(min(n - 1, self.num_edges)):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        if not edges:
            edges = [[0, 0]]
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()