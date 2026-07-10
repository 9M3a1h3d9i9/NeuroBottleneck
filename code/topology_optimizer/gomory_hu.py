import networkx as nx

class GomoryHuAnalyzer:
    """
    This class analyzes a telecom network topology and extracts the Gomory-Hu tree
    to find structural bottlenecks (min-cuts).
    """

    def __init__(self):
        # Initialize an empty undirected graph
        self.graph = nx.Graph()
        self.gh_tree = None

    def load_topology(self, edges_with_capacity):
        """
        Loads the network topology from a list of edges.
        edges_with_capacity: list of tuples (node_u, node_v, capacity)
        """
        for u, v, cap in edges_with_capacity:
            # Adding edge with 'capacity' attribute which is required for min-cut
            self.graph.add_edge(u, v, capacity=cap)
        
        print(f"Topology loaded: {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def build_gomory_hu_tree(self):
        """
        Builds the Gomory-Hu tree using NetworkX (which uses Gusfield's approach).
        Returns the tree graph.
        """
        # The gomory_hu_tree function calculates all min-cuts efficiently
        self.gh_tree = nx.gomory_hu_tree(self.graph, capacity='capacity')
        print("Gomory-Hu tree successfully generated.")
        return self.gh_tree

    def find_bottleneck(self, source, target):
        """
        Finds the minimum cut (bottleneck) between a source and a target node.
        """
        if self.gh_tree is None:
            print("Error: Please build the Gomory-Hu tree first!")
            return None, None

        # In a G-H tree, the min-cut between two nodes is simply the edge 
        # with the minimum weight along the shortest path between them.
        path = nx.shortest_path(self.gh_tree, source=source, target=target)
        
        min_capacity = float('inf')
        bottleneck_edge = None

        # Loop through the path to find the weakest link (the bottleneck)
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            cap = self.gh_tree[u][v]['weight']
            
            if cap < min_capacity:
                min_capacity = cap
                bottleneck_edge = (u, v)

        return bottleneck_edge, min_capacity

# ==========================================
# Testing the module with a sample network
# ==========================================
if __name__ == "__main__":
    # Create an instance of our analyzer
    analyzer = GomoryHuAnalyzer()

    # Sample data: (Node A, Node B, Link Capacity in Gbps)
    # This represents a small section of a telecom network
    sample_edges = [
        (1, 2, 40), (1, 3, 20),
        (2, 3, 20), (2, 4, 30),
        (3, 4, 10), (4, 5, 50),
        (3, 5, 10)
    ]

    # Step 1: Load the data
    analyzer.load_topology(sample_edges)

    # Step 2: Build the tree
    analyzer.build_gomory_hu_tree()

    # Step 3: Find bottleneck between Node 1 and Node 5
    src_node = 1
    dst_node = 5
    b_edge, b_cap = analyzer.find_bottleneck(src_node, dst_node)

    print(f"--- Results ---")
    print(f"Traffic from Node {src_node} to Node {dst_node}:")
    print(f"The bottleneck is at edge {b_edge} with capacity {b_cap} Gbps.")