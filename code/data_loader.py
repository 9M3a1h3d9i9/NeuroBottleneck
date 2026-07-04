# The task of this file is
# to generate or upload the network praph and extract the basic mathematical structures
# Like the adjacency matrix and the properties of the nodes.

# Given the nature of bottleneck management,
# the networkx tool is the best option for initial implementations 
# and calculating capacity matrices and the Gomory-Hu Tree.

import networkx as nx
import numpy as np

import torch
from config import NetworkConfig

class ORanDataLoader:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.graph = None

    def generate_synthetic_topology(self) -> nx.Graph:
        """تولید یک گراف تصادفی برای شبیه‌سازی توپولوژی شبکه O-RAN"""
        """Generate a random graph to simulate the O-RAN network topology"""
        # تولید یک گراف تصادفی Erdos-Renyi به عنوان بیس شبکه
        # Produce a random Erdos-Renyi graph as the base network
        G = nx.erdos_renyi_graph(n=self.config.num_nodes, \
                                  p=self.config.edge_probability, seed=42)
        
        # تبدیل به گراف جهت‌دار یا بدون جهت بسته به نیاز و تزریق ظرفیت به لینک‌ها
        # Convert
        for u, v in G.edges():
            capacity = np.random.uniform(self.config.min_capacity, self.config.max_capacity)
            G[u][v]['capacity'] = round(capacity, 2)
            G[u][v]['load'] = 0.0  # ترافیک اولیه جاری در لینک
            
        self.graph = G
        return G

    def compute_gomory_hu_tree(self) -> nx.Graph:
        """محاسبه درخت گوموری-هو برای شناسایی دقیق گلوگاه‌های ترافیکی شبکه"""
        """Compute the Gomory-Hu tree to identify network traffic bottlenecks"""
        if self.graph is None:
            self.generate_synthetic_topology()
            
        # محاسبه درخت بر اساس ظرفیت لینک‌ها (نیاز به گراف بدون جهت دارد)
        # Compute the tree based on link capacities (requires an undirected graph)
        # این درخت به عامل DRL کمک می‌کند ساختار گلوگاه‌ها را بهتر درک کند
        # This tree helps the DRL agent better understand the bottleneck structures
        gh_tree = nx.gomory_hu_tree(self.graph, capacity='capacity')
        return gh_tree

    def get_gnn_inputs(self):
        """تبدیل داده‌های گراف به تنسورهای PyTorch برای تغذیه به GraphSAGE"""
        """Convert graph data to PyTorch tensors for feeding into GraphSAGE"""
        if self.graph is None:
            self.generate_synthetic_topology()
            
        # ۱. ماتریس مجاورت (Edge Index) برای PyTorch Geometric
        # 1. Adjacency matrix (edge index) for PyTorch Geometric
        edges = np.array(self.graph.edges()).T
        edge_index = torch.tensor(edges, dtype=torch.long)
        
        # ۲. ویژگی‌های گره‌ها (مثلاً درجه گره و ترافیک اولیه)
        # Node features (e.g node degree and initial traffic)
        # در فازهای بعدی ویژگی‌های پیچیده‌تری مثل لیتنسی گره‌ها اضافه می‌شود
        # In Next phases, more complex features like node latency will be added
        node_features = []
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            node_features.append([degree, 0.0]) # [Degree, Initial Demand]
            
        x = torch.tensor(node_features, dtype=torch.float)
        
        return x, edge_index

if __name__ == "__main__":
    # یک تست کوچک برای اطمینان از صحت کارکرد ماژول
    # A small test to ensure the module works correctly
    from config import MainConfig
    cfg = MainConfig()
    loader = ORanDataLoader(cfg.network)
    
    g = loader.generate_synthetic_topology()
    print(f"Graph generated successfully with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
    
    x, edge_index = loader.get_gnn_inputs()
    print(f"GNN Feature Tensor Shape: {x.shape}")
    print(f"GNN Edge Index Shape: {edge_index.shape}")


    # ممکن از کامنت هایی که به زبان انگلیسی نوشته شده اند، غلط املایی جزئی داشته باشند
