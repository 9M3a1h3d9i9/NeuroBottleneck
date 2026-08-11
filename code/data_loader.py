# In the name of GOD                    بسم الله الرحمن الرحیم 
# Core Data Ingestion & Routing Pipeline for NeuroBottleneck.
# پایپ‌لاین اصلی ورود داده‌ها و مسیریابی برای نورو-باتلنک

import os 
import re
import glob
import sys 

import networkx as nx
import matplotlib.pyplot as plt

# Ensure the 'code' directory is in the Python search path for modular imports
# این خط مسیر فعلی فایل را به مفسر پایتون می‌شناساند تاایمپورت‌های ماژولار ما با خطا مواجه نشوند
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from topology_optimizer.gomory_hu import GomoryHuAnalyzer
from topology_optimizer.picard_cuts import PicardCutAnalyzer


class ORanDynamicPipeline:
    def __init__(self, topo_file: str, traffic_dir: str):
        # Initialize paths and create an empty base network graph
        self.topo_file = topo_file
        self.traffic_dir = traffic_dir
        self.base_graph = nx.Graph()

    def load_base_topology(self):
        """Parses the static topology file to load physical nodes and links with capacities."""
        if not os.path.exists(self.topo_file):
            raise FileNotFoundError(f"Static topology file not found at {self.topo_file}")

        # Read the entire raw text of the SNDlib topology file
        # خواندن دیتای خام
        with open(self.topo_file, 'r') as f:
            content = f.read()

        # استخراج گره‌های فیزیکی شبکه. این بخش با عبارات منظم 
        # (Regex) 
        # محتوای داخل پرانتز 
        # NODES 
        # را شکار می‌کند
        nodes_match = re.search(r'NODES\s*\((.*?)\n\s*\)', content, re.DOTALL)
        
        if nodes_match:
            for line in nodes_match.group(1).strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    self.base_graph.add_node(line.split()[0]) # Add router to graph

        # Extract links, source/target nodes, and physical capacities
        links_match = re.search(r'LINKS\s*\((.*?)\n\s*\)', content, re.DOTALL)
        if links_match:
            for line in links_match.group(1).strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 6:
                        link_id = parts[0]
                        source = parts[2]
                        target = parts[3]
                        capacity = float(parts[5])
                        # Initialize active load at 0.0 for all edges
                        self.base_graph.add_edge(source, target, id=link_id, capacity=capacity, load=0.0)
        
        print(f"[Base Topology] Loaded {self.base_graph.number_of_nodes()} nodes and {self.base_graph.number_of_edges()} links.")

    def apply_traffic_snapshot(self, snapshot_file: str):
        """Parses a 5-min demand matrix (telemetry) and routes traffic."""
        if not os.path.exists(snapshot_file):
            raise FileNotFoundError(f"Traffic snapshot file not found at {snapshot_file}")

        # Reset loads for the new snapshot to avoid overlapping old data
        for u, v in self.base_graph.edges():
            self.base_graph[u][v]['load'] = 0.0

        with open(snapshot_file, 'r') as f:
            content = f.read()

        # استخراج ماتریس تقاضا که نشان‌دهنده ترافیک و تله‌متری لحظه‌ای بین مبدأ و مقصدهاست
        demands_match = re.search(r'DEMANDS\s*\((.*?)\n\s*\)', content, re.DOTALL)
        if not demands_match:
            print("[Warning] No demands section detected in snapshot.")
            return

        for line in demands_match.group(1).strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 6:
                    source, target = parts[2], parts[3]
                    demand_value = float(parts[5])

                    # Route the demand using the shortest path and add to current link loads
                    # مسیریابی تقاضا ها با استفاده از کوتاهترین مسیر و بار لینک ها
                    if nx.has_path(self.base_graph, source, target):
                        path = nx.shortest_path(self.base_graph, source, target)
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            self.base_graph[u][v]['load'] += demand_value

    def generate_residual_graph(self) -> nx.Graph:
        """Builds the active Residual Capacity Graph (G_residual)."""
        residual_g = nx.Graph()
        for node in self.base_graph.nodes():
            residual_g.add_node(node)

        # محاسبه ظرفیت باقیمانده. این گراف دقیقا همان دیتایی است که به عنوان خوراک به الگوریتم‌های برش و درخت پاس داده می‌شود.
        for u, v, data in self.base_graph.edges(data=True):
            capacity = data['capacity']
            load = data['load']
            # Ensure capacity doesn't go below 0.1 to avoid division by zero or structural errors
            residual_cap = max(0.1, capacity - load) 
            residual_g.add_edge(u, v, capacity=residual_cap)
            
        return residual_g

    def run_modular_analysis(self, source: str, target: str, output_img_path: str):
        """Runs custom Gomory-Hu and Picard modules on G_residual."""
        residual_net = self.generate_residual_graph()

        print("\n[Modular Analysis] Running Gomory-Hu (C-01)...")
        gh_analyzer = GomoryHuAnalyzer()
        # تبدیل گراف نتورک‌ایکس به فرمت لیستی دلخواه کلاس گوموری-هو (تاپل‌های سه‌تایی)
        edges_list = [(u, v, data['capacity']) for u, v, data in residual_net.edges(data=True)]
        gh_analyzer.load_topology(edges_list)
        gh_tree = gh_analyzer.build_gomory_hu_tree()
        gh_b_edge, gh_b_cap = gh_analyzer.find_bottleneck(source, target)
        
        print("\n[Modular Analysis] Running Picard Cuts (C-02)...")

        # پاس دادن مستقیم آبجکت گراف به کلاس پیکارد طبق معماری نوشته شده در فایل آن
        picard_analyzer = PicardCutAnalyzer(residual_net)
        picard_bottlenecks, min_cut_val = picard_analyzer.analyze_min_cuts(source, target)

        print("\n=== Pipeline Diagnostic Results ===")
        print(f"Target Path: {source} -> {target}")
        print(f"Gomory-Hu Bottleneck: {gh_b_edge} with capacity {gh_b_cap:.2f} Mbps")
        print(f"Picard Min-Cut Capacity: {min_cut_val:.2f} Mbps")
        print(f"Picard Identified Bottleneck Links: {picard_bottlenecks}")

        # Visualization setup for matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        pos = nx.spring_layout(self.base_graph, seed=42)
        
        # هایلایت کردن یال‌های گلوگاهی (رنگ قرمز و ضخامت بیشتر) برای شناسایی بصری سریع
        edge_colors = []
        edge_widths = []
        for u, v in self.base_graph.edges():
            if (u, v) in picard_bottlenecks or (v, u) in picard_bottlenecks:
                edge_colors.append('red')
                edge_widths.append(4.5)
            else:
                edge_colors.append('gray')
                edge_widths.append(1.5)

        # Draw base network
        nx.draw(self.base_graph, pos, ax=axes[0], with_labels=True, node_color='skyblue', 
                node_size=600, font_weight='bold', edge_color=edge_colors, width=edge_widths)
        
        # Calculate and show percentage of utilization on edges
        edge_labels = {}
        for u, v, data in self.base_graph.edges(data=True):
            util = (data['load'] / data['capacity']) * 100 if data['capacity'] > 0 else 0
            edge_labels[(u, v)] = f"{util:.0f}%"
            
        nx.draw_networkx_edge_labels(self.base_graph, pos, edge_labels=edge_labels, ax=axes[0], font_size=7)
        axes[0].set_title(f"Abilene Active Load\n[Red Links = Picard Bottlenecks for {source}→{target}]", fontsize=11)

        # Draw Gomory-Hu Tree if available
        if gh_tree:
            pos_gh = nx.circular_layout(gh_tree)
            nx.draw(gh_tree, pos_gh, ax=axes[1], with_labels=True, node_color='lightgreen', 
                    node_size=600, font_weight='bold', edge_color='brown', width=2.5)
            
            # Format and display capacities on the tree edges
            gh_labels = nx.get_edge_attributes(gh_tree, 'weight')
            gh_labels_formatted = {k: f"{v:.0f}M" for k, v in gh_labels.items()}
            nx.draw_networkx_edge_labels(gh_tree, pos_gh, edge_labels=gh_labels_formatted, ax=axes[1], font_size=7)
            axes[1].set_title("Gomory-Hu Bottleneck Tree (Residual Capacities)", fontsize=11)

        # Create directories safely and save the final plot
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=300)
        plt.close()
        print(f"\n[OK] Diagnostic plot saved to: {output_img_path}")


if __name__ == "__main__":
    # Define relative paths based on the project structure
    topo_path = "data/raw/abilene_base_topology.txt"
    traffic_dir = "data/raw/dynamic_traffic"
    output_image = "data/processed/abilene_active_bottlenecks.png"
    
    # Initialize pipeline and build base graph
    pipeline = ORanDynamicPipeline(topo_path, traffic_dir)
    pipeline.load_base_topology()
    
    # Find all traffic snapshot files and pick the first one for testing
    snapshot_files = sorted(glob.glob(os.path.join(traffic_dir, "demandMatrix-*.txt")))
    
    if snapshot_files:
        print(f"\n[Pipeline] Active Ingesting: {os.path.basename(snapshot_files[0])}")
        pipeline.apply_traffic_snapshot(snapshot_files[0])
        
        # اجرای تست ماژولار و ساخت خروجی روی مسیر بین آتلانتا تا سیاتل
        pipeline.run_modular_analysis(
            source="ATLAM5", 
            target="STTLng", 
            output_img_path=output_image
        )
    else:
        print(f"[ERROR] Telemetry ingestion failed. No files detected in {traffic_dir}")