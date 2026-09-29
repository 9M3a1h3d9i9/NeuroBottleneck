"""انکودرهای GNN: GraphSAGE, GAT, GIN, MLP"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GINConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, out):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.node_proj(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z = global_mean_pool(x, batch=batch)
        return self.norm(z)


class GATEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, out, heads=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.conv1 = GATConv(hidden, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, out, heads=1)
        self.norm = nn.LayerNorm(out)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.node_proj(x))
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z = global_mean_pool(x, batch=batch)
        return self.norm(z)


class GINEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, out):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        mlp1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        mlp2 = nn.Sequential(nn.Linear(hidden, out), nn.ReLU(), nn.Linear(out, out))
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.norm = nn.LayerNorm(out)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.node_proj(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z = global_mean_pool(x, batch=batch)
        return self.norm(z)


class MLPEncoder(nn.Module):
    """Baseline: بدون ساختار گراف"""
    def __init__(self, node_dim, edge_dim, hidden, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out),
        )
        self.norm = nn.LayerNorm(out)

    def forward(self, x, edge_index, edge_attr=None):
        z = self.net(x.mean(dim=0, keepdim=True))
        return self.norm(z)


def build_gnn(gnn_type, node_dim, edge_dim, hidden, out):
    if gnn_type == "graphsage":
        return GraphSAGEEncoder(node_dim, edge_dim, hidden, out)
    if gnn_type == "gat":
        return GATEncoder(node_dim, edge_dim, hidden, out)
    if gnn_type == "gin":
        return GINEncoder(node_dim, edge_dim, hidden, out)
    if gnn_type == "mlp":
        return MLPEncoder(node_dim, edge_dim, hidden, out)
    raise ValueError(f"Unknown GNN: {gnn_type}")