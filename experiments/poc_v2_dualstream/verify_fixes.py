"""Quick verification test for bug fixes before re-running 3 seeds"""

import torch
import numpy as np
from src.envs.network_env import NetworkEnv
from src.models.dual_stream_extractor import DualStreamExtractor
from configs import EnvConfig, GHConfig, RewardConfig


def verify_fixes():
    print("--- Starting Verification Test ---")

    # 1. Test GH reward leakage (GNN_only environment with use_gh_reward=False)
    env_cfg = EnvConfig(topology="abilene")
    gh_cfg = GHConfig()
    reward_cfg = RewardConfig(use_soft_penalty=True)  # Enable soft penalty to test if it stays zero in GNN_only

    print("\n--- 1. Testing Environment & Reward Leakage ---")
    env = NetworkEnv(env_cfg, gh_cfg, reward_cfg, use_gh_in_obs=False, use_gh_reward=False)
    obs, _ = env.reset(seed=42)

    print(f"Nodes (N): {env.N} | Edges (M): {env.M}")  # Should be 12 and 15
    print(f"use_gh_reward status: {env.use_gh_reward}")

    # Perform a capacity decrease action (to potentially trigger the penalty)
    action = 1  # Decrease capacity of edge 0 (direction -1 in your environment)
    obs, reward, terminated, truncated, info = env.step(action=action)

    p_soft = info['components']['P_soft']
    if p_soft == 0.0:
        print("✅ Success: No GH reward leakage (p_soft in GNN_only is 0.0).")
    else:
        print(f"❌ Error: Reward leakage exists! p_soft = {p_soft}")

    # 2. Test fake GNN graph bug
    print("\n--- 2. Testing GNN Feature Extractor ---")

    # Observation space shape
    obs_space = env.observation_space

    extractor = DualStreamExtractor(
        observation_space=obs_space,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        num_nodes=env.N,
        num_edges=env.M,
        use_gnn=True,
        use_gh_feature=False
    )

    # Extract edge_index (requires device)
    device = torch.device('cpu')
    edge_index = extractor._get_edge_index(device)

    num_edges_in_tensor = edge_index.size(1)
    print(f"Edges sent to GNN (bidirectional): {num_edges_in_tensor}")

    # For an undirected Abilene graph with 15 physical edges, 30 directed edges must be sent to GNN.
    if num_edges_in_tensor == 30:
        print("✅ Success: Abilene graph correctly connected to GNN.")
    else:
        print(f"❌ Error: Graph still has issues! Edges sent to GNN: {num_edges_in_tensor}")


if __name__ == "__main__":
    verify_fixes()