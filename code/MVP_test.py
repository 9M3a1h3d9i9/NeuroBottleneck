# !pip install networkx

import networkx as nx
import random

G = nx.gnm_random_graph(15, 30, seed=42)

for u, v in G.edges():
    G[u][v]['capacity'] = random.randint(10, 100)

gh_tree = nx.gomory_hu_tree(G, capacity='capacity')


print(" Number of nodes: ", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# 
critical_edges = list(gh_tree.edges(data=True))
print("Total edges in Gomory-Hu tree: ", len(critical_edges))

#
print("\n Sample critical edges (u, v, min/cut_capacity): ")
for u, v, data in critical_edges[ :5]:
    print(f"  ({u}, {v}) -> min/cut = {data['weight']}")
    # print(f"Edge: {u}-{v}, data = {data}")


# for u, v, data in gh_tree.edges(data=True):
#     print(f"Edge: {u}-{v}, data keys: {data.keys()}, data: {data}")
#     if u == 0:  # فقط یکی دو تا را چاپ کن
#         break