import re
from pathlib import Path
import networkx as nx


def parse_sndlib_native(file_path: str = None) -> nx.Graph:
    """
    Parse an SNDlib native format file and return an undirected weighted graph.
    Each link gets a 'capacity' attribute taken from the module capacity.
    """
    if file_path is None:
        base = Path(__file__).resolve().parent.parent  # NeuroBottleneck
        file_path = base / "data" / "sndlib" / "india35.txt" 
    text = Path(file_path).read_text()

    nodes_match = re.search(r'NODES\s*\((.*?)\)\n', text, re.DOTALL)
    links_match = re.search(r'LINKS\s*\((.*?)\)\n', text, re.DOTALL)

    if not nodes_match or not links_match:
        raise ValueError("Could not find NODES or LINKS sections")

    nodes_text = nodes_match.group(1)
    links_text = links_match.group(1)

    G = nx.Graph()

    # Add nodes if present
    for line in nodes_text.strip().splitlines():
        m = re.match(r'\s*(\d+)\s*\(', line)
        if m:
            G.add_node(int(m.group(1)))

    # Parse links
    # Format: <id> ( <source> <target> ) <pre_cap> <pre_cost> <routing_cost> <setup_cost> ( <module_cap> <module_cost> ... )
    for line in links_text.strip().splitlines():
        m = re.match(
            r'\s*(\d+)\s*\(\s*(\d+)\s+(\d+)\s*\)\s+'
            r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*'
            r'\(\s*([\d.]+)',
            line,
        )
        if m:
            link_id = int(m.group(1))
            source = int(m.group(2))
            target = int(m.group(3))
            pre_cap = float(m.group(4))
            module_cap = float(m.group(8))

            capacity = module_cap if pre_cap == 0.0 else pre_cap
            G.add_edge(source, target, capacity=capacity, link_id=link_id)
        else:
            print(f"Warning: could not parse link line: {line.strip()}")

    return G


if __name__ == "__main__":
    G = parse_sndlib_native()
    # G = parse_sndlib_native("/home/mahdics313/NeuroBottleneck/data/sndlib/india35.txt")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Is connected: {nx.is_connected(G)}")