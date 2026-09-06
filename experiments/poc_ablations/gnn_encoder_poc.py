# پیاده‌سازی ساده GraphSAGE برای استخراج ویژگی از گراف

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import from_networkx
import numpy as np

class SimpleGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # لایه اول
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # لایه دوم
        x = self.conv2(x, edge_index)
        # میانگین‌گیری روی همه گره‌ها برای تولید یک بردار ثابت
        x = global_mean_pool(x, batch=None)  # batch=None برای یک گراف
        return x

# --------------------------------------------------------------------
# تبدیل گراف NetworkX به فرمت PyTorch Geometric
# --------------------------------------------------------------------
def nx_to_pyg_data(G):
    data = from_networkx(G)
    
    # ساخت ویژگی‌های پایه برای گره‌ها
    # 3 ویژگی: درجه نرمال‌شده، ایندکس نرمال‌شده، و یک مقدار ثابت 1
    num_nodes = G.number_of_nodes()
    node_features = []
    for node in G.nodes():
        deg = G.degree(node)
        node_features.append([
            deg / float(num_nodes),
            node / float(num_nodes),
            1.0
        ])
    
    data.x = torch.tensor(node_features, dtype=torch.float32)
    
    # اگر ویژگی‌های لبه (مثل ظرفیت) را هم بخواهیم، می‌توانیم اضافه کنیم
    # اما در این نسخه ساده، فقط از گره‌ها استفاده می‌کنیم.
    return data