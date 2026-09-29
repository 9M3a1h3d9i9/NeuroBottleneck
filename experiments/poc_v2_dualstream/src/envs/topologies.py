"""توپولوژی‌های شبکه: Abilene, India35, GéANT, Nobel-EU"""

import networkx as nx
import numpy as np
import os
import xml.etree.ElementTree as ET


def build_abilene() -> nx.Graph:
    """Abilene: 12 nodes, 15 edges"""
    G = nx.Graph()
    nodes = ['NY', 'WA', 'CHI', 'DEN', 'KC', 'LA', 'ATL', 'HO', 'SEA', 'SV', 'IN']
    G.add_nodes_from(range(len(nodes)))
    edges = [
        (0, 1, 2000), (0, 2, 2000), (0, 8, 2000),
        (1, 2, 2000), (1, 9, 2000),
        (2, 3, 2000), (2, 4, 2000), (2, 5, 2000),
        (3, 5, 2000), (3, 10, 2000),
        (4, 6, 2000), (4, 7, 2000),
        (5, 6, 2000), (5, 7, 2000),
        (6, 0, 2000)
    ]
    G.add_weighted_edges_from(edges, weight='capacity')
    for u, v in G.edges():
        G[u][v]['utilization'] = 0.4
        G[u][v]['delay'] = 10
        G[u][v]['demand'] = 400.0
    return G


def load_sndlib(filepath: str) -> nx.Graph:
    """پارسر SNDlib برای فایل‌های XML"""
    tree = ET.parse(filepath)
    root = tree.getroot()
    G = nx.Graph()

    for node in root.findall('.//node'):
        nid = node.get('id')
        G.add_node(nid)

    for link in root.findall('.//link'):
        src = link.get('source')
        dst = link.get('target')
        cap_el = link.find('capacity')
        cap = float(cap_el.text) if cap_el is not None else 1000.0
        G.add_edge(src, dst, capacity=cap, utilization=0.4, delay=10, demand=400.0)

    return G


def load_topology(name: str, data_dir: str = "./data/raw/sndlib") -> nx.Graph:
    """بارگذاری توپولوژی بر اساس نام"""
    if name == "abilene":
        return build_abilene()

    fname_map = {
        "india35": "india35.xml",
        "geant": "geant.xml",
        "nobel": "nobel-eu.xml",
    }
    if name not in fname_map:
        raise ValueError(f"Unknown topology: {name}")

    path = os.path.join(data_dir, fname_map[name])
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Falling back to Abilene.")
        return build_abilene()
    return load_sndlib(path)