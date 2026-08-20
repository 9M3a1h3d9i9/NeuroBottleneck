import networkx as nx


def parse_sndlib(file_path):
    G = nx.Graph()
    mode = None  # None, 'nodes', 'links'

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('NODES'):
                mode = 'nodes'
                continue

            if line.startswith('LINKS'):
                mode = 'links'
                continue

            if line.startswith('DEMANDS') or line.startswith('ADMISSIBLE_PATHS'):
                mode = None
                continue

            if mode == 'nodes':
                parts = line.split()
                # node line: <node_id> ( <longitude> <latitude> )
                if len(parts) >= 1 and parts[0].isdigit():
                    node_id = int(parts[0])
                    G.add_node(node_id)

            elif mode == 'links':
                parts = line.split()
                # link line format:
                # <id> ( <source> <target> ) <pre_cap> <pre_cost> <routing_cost> <setup_cost> ( <module_cap> <module_cost> )
                # split gives:
                # 0=id, 1='(', 2=source, 3=target, 4=')', 5=pre_cap, 6=pre_cost, 7=routing_cost, 8=setup_cost, 9='(', 10=module_cap
                if len(parts) >= 11 and parts[0].isdigit() and parts[1] == '(' and parts[4] == ')':
                    link_id = int(parts[0])
                    source = int(parts[2])
                    target = int(parts[3])

                    pre_cap = float(parts[5])
                    module_cap = float(parts[10])

                    capacity = pre_cap if pre_cap > 0 else module_cap

                    G.add_edge(source, target, capacity=capacity, link_id=link_id)
                else:
                    print(f"Skipped line (unexpected format): {line}")

    return G


if __name__ == "__main__":
    G = parse_sndlib("data/sndlib/india35.txt")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Is connected: {nx.is_connected(G)}")