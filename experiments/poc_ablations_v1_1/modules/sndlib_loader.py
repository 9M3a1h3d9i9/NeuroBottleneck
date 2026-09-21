# ============================================================================
# بارگذار جامع شبکه‌های SNDlib (فرمت Native ASCII)
# ============================================================================
# این ماژول، فایل‌های استاندارد SNDlib را پارس کرده و به گراف NetworkX تبدیل
# می‌کند. از فرمت‌های زیر پشتیبانی می‌کند:
#   - فرمت Native ASCII (با بخش‌های NODES, LINKS, DEMANDS)
#   - فرمت ساده متنی (خط اول: N M، سپس M خط یال)
# ============================================================================
"""بارگذار جامع شبکه‌های SNDlib (پشتیبانی از نام گره‌های متنی و عددی)"""

import re
import os
import networkx as nx


class SNDlibLoader:
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"فایل پیدا نشد: {filepath}")
        self.filepath = filepath
        self.graph = None

    def load(self):
        with open(self.filepath, 'r') as f:
            content = f.read()

        if 'NODES' in content and 'LINKS' in content:
            self.graph = self._parse_native(content)
        else:
            self.graph = self._parse_simple(content)

        self._add_default_attributes()

        print(f"[LOADED] {os.path.basename(self.filepath)}: "
              f"{self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return self.graph

    def _parse_native(self, content):
        """پارس فرمت Native SNDlib با پشتیبانی از نام‌های متنی"""
        G = nx.Graph()

        # === بخش NODES ===
        nodes_match = re.search(r'NODES\s*\(\s*([\s\S]*?)\n\s*\)', content)
        node_names = []
        if nodes_match:
            for line in nodes_match.group(1).strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # فرمت: NODE_NAME ( lon lat )  یا  INT_ID ( lon lat )
                m = re.match(r'^(\S+)\s+\(', line)
                if m:
                    name = m.group(1)
                    if name not in node_names:
                        node_names.append(name)

        # نگاشت نام به ایندکس
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        G.add_nodes_from(range(len(node_names)))

        # === بخش LINKS ===
        links_match = re.search(
            r'LINKS\s*\(\s*([\s\S]*?)\n\s*\)\s*\n', content
        )
        if links_match:
            for line in links_match.group(1).strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # فرمت: link_id ( src tgt ) pre_cap ... ( mod_cap mod_cost )
                m = re.search(r'\(\s*(\S+)\s+(\S+)\s*\)', line)
                if m:
                    src_name, tgt_name = m.group(1), m.group(2)
                    if src_name in name_to_idx and tgt_name in name_to_idx:
                        u = name_to_idx[src_name]
                        v = name_to_idx[tgt_name]
                        # آخرین پرانتز با دو عدد اعشاری = ظرفیت ماژول
                        cap_matches = re.findall(
                            r'\(\s*([\d.]+)\s+([\d.]+)\s*\)', line
                        )
                        cap = float(cap_matches[-1][0]) if cap_matches else 10000.0
                        G.add_edge(u, v, capacity=cap)

        # === بخش DEMANDS ===
        demands = []
        demands_match = re.search(r'DEMANDS\s*\(\s*([\s\S]*?)\n\s*\)', content)
        if demands_match:
            for line in demands_match.group(1).strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                m = re.search(r'\(\s*(\S+)\s+(\S+)\s*\)', line)
                if m:
                    src_name, tgt_name = m.group(1), m.group(2)
                    if src_name in name_to_idx and tgt_name in name_to_idx:
                        nums = re.findall(r'[\d.]+', line)
                        if len(nums) >= 4:
                            try:
                                demands.append((
                                    name_to_idx[src_name],
                                    name_to_idx[tgt_name],
                                    float(nums[3])
                                ))
                            except (ValueError, IndexError):
                                pass

        G.graph['demands'] = demands
        G.graph['node_names'] = node_names
        G.graph['name_to_idx'] = name_to_idx
        return G

    def _parse_simple(self, content):
        """پارس فرمت ساده: خط اول N M، سپس M خط یال"""
        G = nx.Graph()
        lines = [l.strip() for l in content.split('\n')
                 if l.strip() and not l.startswith('#')]

        if not lines:
            raise ValueError("فایل خالی است")

        parts = lines[0].split()
        if len(parts) >= 2:
            try:
                n_nodes = int(parts[0])
                n_edges = int(parts[1])
                G.add_nodes_from(range(n_nodes))
                for line in lines[1:1+n_edges]:
                    p = line.split()
                    if len(p) >= 3:
                        G.add_edge(int(p[0]), int(p[1]), capacity=float(p[2]))
            except ValueError:
                pass
        return G

    def _add_default_attributes(self):
        import numpy as np
        if self.graph is None or self.graph.number_of_edges() == 0:
            return
        for u, v in self.graph.edges():
            if 'capacity' not in self.graph[u][v]:
                self.graph[u][v]['capacity'] = 2000.0
            cap = self.graph[u][v]['capacity']
            self.graph[u][v]['demand'] = np.random.uniform(0.4 * cap, 0.85 * cap)
            self.graph[u][v]['utilization'] = self.graph[u][v]['demand'] / cap
            self.graph[u][v]['latency'] = np.random.uniform(5, 25)
            self.graph[u][v]['sinr'] = np.random.uniform(10, 30)


def load_network(filepath):
    return SNDlibLoader(filepath).load()