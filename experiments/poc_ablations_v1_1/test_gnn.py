"""تست اجرای GNNFeaturesExtractor"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.sndlib_loader import load_network
from modules.network_env import NetworkEnvV11
from modules.gnn_encoder import GNNFeaturesExtractor
import torch

# بارگذاری
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filepath = os.path.join(base, 'data/raw/sndlib/static/abilene.txt')
graph = load_network(filepath)

# محیط
env = NetworkEnvV11(graph, seed=42)
obs, _ = env.reset()

# Features Extractor
fe = GNNFeaturesExtractor(
    observation_space=env.observation_space,
    features_dim=65,
    n_edges=env.n_edges,
    gnn_hidden=64,
    gnn_out=64,
)

# تست
obs_tensor = torch.tensor(obs).unsqueeze(0)
print(f"Input shape: {obs_tensor.shape}")
print(f"Expected output: (1, 65)")

output = fe(obs_tensor)
print(f"Actual output: {output.shape}")
print(f"Output sample: {output[0, :5]}")
print("✅ GNN Features Extractor works!")
