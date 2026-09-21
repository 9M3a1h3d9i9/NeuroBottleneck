# ============================================================================
# GraphSAGE Encoder + Custom Features Extractor برای SB3
# ============================================================================
# این ماژول، دو بخش دارد:
#   1. SimpleGraphSAGE: مدل GraphSAGE ساده
#   2. GNNFeaturesExtractor: تبدیل ویژگی‌های تخت به گراف و اجرای GNN
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np


class SimpleGraphSAGE(nn.Module):
    """مدل GraphSAGE ساده با دو لایه"""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.out_dim = out_channels
    
    def forward(self, x, edge_index, batch=None):
        """
        ورودی:
            x: ویژگی‌های گره‌ها (N, in_channels)
            edge_index: لیست یال‌ها (2, E)
            batch: بردار batch (اختیاری)
        خروجی:
            graph_embedding: بردار گراف (out_channels,)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Features Extractor سفارشی برای SB3
    
    این کلاس، بردار تخت مشاهده را دریافت کرده، به ساختار گراف تبدیل می‌کند،
    GNN را اجرا می‌کند و بردار گراف را برمی‌گرداند.
    
    ساختار بردار مشاهده:
      [c_norm_1, d_norm_1, u_1, sinr_1, ..., c_norm_E, d_norm_E, u_E, sinr_E, λ_min]
    """
    
    def __init__(self, observation_space, features_dim=64, 
                 n_edges=None, gnn_hidden=64, gnn_out=64):
        # محاسبه ابعاد
        obs_dim = observation_space.shape[0]
        if n_edges is None:
            n_edges = (obs_dim - 1) // 4
        
        self.n_edges = n_edges
        self.n_nodes = n_edges + 1  # تقریب
        self.gnn_out = gnn_out
        
        # ابعاد نهایی features (بردار GNN + λ_min)
        super().__init__(observation_space, features_dim=gnn_out + 1)
        
        # === GNN Encoder ===
        # ویژگی‌های گره: میانگین ویژگی‌های یال‌های متصل
        node_in_channels = 4  # c_norm, d_norm, u, sinr
        self.gnn = SimpleGraphSAGE(
            in_channels=node_in_channels,
            hidden_channels=gnn_hidden,
            out_channels=gnn_out,
        )
        
        # === ساخت edge_index ثابت برای گراف Abilene ===
        # (برای گراف‌های دیگر، این باید داینامیک باشد)
        self._build_static_edge_index()
    
    def _build_static_edge_index(self):
        """
        ساخت edge_index ثابت بر اساس ساختار گراف
        در نسخه کامل، این باید از محیط دریافت شود.
        برای Abilene، ساختار ثابت است.
        """
        # یال‌های Abilene (11 node, 15 edges)
        # این باید با محیط هماهنگ باشد
        abilene_edges = [
            (0, 1), (0, 2), (0, 8), (1, 2), (1, 9),
            (2, 3), (2, 4), (2, 5), (3, 5), (3, 10),
            (4, 6), (4, 7), (5, 6), (5, 7), (6, 0)
        ]
        
        edge_src = [u for u, v in abilene_edges]
        edge_dst = [v for u, v in abilene_edges]
        # گراف بدون‌جهت: هر دو جهت
        edge_src_full = edge_src + edge_dst
        edge_dst_full = edge_dst + edge_src
        
        self.register_buffer(
            'edge_index',
            torch.tensor([edge_src_full, edge_dst_full], dtype=torch.long)
        )
        
        # تعداد گره‌ها
        self._n_nodes_abilene = max(max(u for u, v in abilene_edges),
                                     max(v for u, v in abilene_edges)) + 1
    
    def forward(self, observations):
        """
        ورودی: observations (batch, obs_dim)
        خروجی: features (batch, gnn_out + 1)
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # === استخراج اجزا ===
        edge_feats = observations[:, :self.n_edges * 4].view(
            batch_size, self.n_edges, 4
        )
        lambda_min = observations[:, -1:].view(batch_size, 1)
        
        # === ساخت ویژگی‌های گره از ویژگی‌های یال ===
        # میانگین‌گیری از یال‌های متصل به هر گره
        node_feats = self._edge_to_node_features(edge_feats, device)
        
        # === اجرای GNN برای هر نمونه در batch ===
        graph_embeddings = []
        for i in range(batch_size):
            x = node_feats[i]  # (N, 4)
            emb = self.gnn(x, self.edge_index)
            graph_embeddings.append(emb)
        
        graph_emb = torch.cat(graph_embeddings, dim=0)  # (batch, gnn_out)
        
        # === الحاق λ_min ===
        features = torch.cat([graph_emb, lambda_min], dim=1)
        
        return features
    
    def _edge_to_node_features(self, edge_feats, device):
        """
        تبدیل ویژگی‌های یال به ویژگی‌های گره
        edge_feats: (batch, n_edges, 4)
        خروجی: (batch, n_nodes, 4)
        """
        batch_size = edge_feats.shape[0]
        n_nodes = self._n_nodes_abilene
        node_feats = torch.zeros(
            batch_size, n_nodes, 4, device=device
        )
        
        # یال‌های Abilene
        abilene_edges = [
            (0, 1), (0, 2), (0, 8), (1, 2), (1, 9),
            (2, 3), (2, 4), (2, 5), (3, 5), (3, 10),
            (4, 6), (4, 7), (5, 6), (5, 7), (6, 0)
        ]
        
        # میانگین ویژگی‌های یال‌ها روی گره‌ها
        count = torch.zeros(batch_size, n_nodes, 1, device=device)
        
        for idx, (u, v) in enumerate(abilene_edges):
            if idx < edge_feats.shape[1]:
                node_feats[:, u, :] += edge_feats[:, idx, :]
                node_feats[:, v, :] += edge_feats[:, idx, :]
                count[:, u, :] += 1
                count[:, v, :] += 1
        
        # جلوگیری از تقسیم بر صفر
        count = torch.clamp(count, min=1.0)
        node_feats = node_feats / count
        
        return node_feats