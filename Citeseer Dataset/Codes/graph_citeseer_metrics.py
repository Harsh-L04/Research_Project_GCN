import torch_geometric.datasets as datasets
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load Citeseer dataset
print("Loading Citeseer dataset...")
dataset = datasets.Planetoid(root='./data', name='Citeseer')
data = dataset[0]
print("Citeseer dataset loaded.\n")

# PyG Data overview
print(f'Nodes: {data.num_nodes}')
print(f'Edges: {data.num_edges}')
print(f'Features per node: {data.num_node_features}')
print(f'Classes: {dataset.num_classes}')
print(f'Isolated nodes: {data.has_isolated_nodes()}')
print(f'Self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print(f'First 5 edges:\n{data.edge_index[:, :5]}\n')

# Masks summary
print(f'Train nodes: {data.train_mask.sum()}')
print(f'Validation nodes: {data.val_mask.sum()}')
print(f'Test nodes: {data.test_mask.sum()}')

# Convert to NetworkX graph
edge_list = data.edge_index.t().tolist()
G = nx.Graph(edge_list)
print("\nConverted to undirected NetworkX graph.\n")

# Graph metrics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)
num_components = nx.number_connected_components(G)
largest_cc = max(nx.connected_components(G), key=len)
largest_cc_size = len(largest_cc)
node_degrees = [deg for _, deg in G.degree()]
avg_degree = np.mean(node_degrees)

print("--- Graph Metrics ---")
print(f'Nodes: {num_nodes}, Edges: {num_edges}')
print(f'Density: {density:.6f}')
print(f'Connected Components: {num_components}')
print(f'Largest Component Size: {largest_cc_size} ({largest_cc_size/num_nodes:.2%})')
print(f'Average Node Degree: {avg_degree:.2f}')

# Degree distribution plot
plt.figure(figsize=(10, 6))
plt.hist(node_degrees, bins=range(max(node_degrees) + 2), color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title('Citeseer Node Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.grid(axis='y', alpha=0.6)
plt.tight_layout()
plt.show()
