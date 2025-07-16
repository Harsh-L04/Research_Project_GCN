import torch
from torch_geometric.data import Data
import torch_geometric.datasets as datasets
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
print("Loading Cora dataset...")
dataset = datasets.Planetoid(root='./data', name='Cora')
data = dataset[0]
print("Cora dataset loaded.\n")

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model = GCN(data.num_node_features, 16, dataset.num_classes)

def predict_labels_for_unlabeled_papers(model, original_data, new_features_list, new_edges_list):
    """
    Predict labels for multiple new papers using a trained GCN.

    Parameters:
    - model: Trained GCN model
    - original_data: PyG Data object (e.g., Cora dataset)
    - new_features_list: List of tensors, each [num_node_features]
    - new_edges_list: List of lists, each containing citation node indices

    Returns:
    - List of predicted classes (int) for the new nodes
    """
    model.eval()
    current_x = original_data.x
    current_edge_index = original_data.edge_index

    new_node_start_idx = original_data.num_nodes
    all_new_edges = []

    # 1. Add new nodes
    for i, new_feat in enumerate(new_features_list):
        current_x = torch.cat([current_x, new_feat.unsqueeze(0)], dim=0)
        new_node_idx = new_node_start_idx + i
        cited_nodes = new_edges_list[i]

        # Add forward and reverse edges
        for cited in cited_nodes:
            all_new_edges.append([new_node_idx, cited])
            all_new_edges.append([cited, new_node_idx])

    # 2. Update edge_index
    if all_new_edges:
        new_edge_tensor = torch.tensor(all_new_edges, dtype=torch.long).t()
        current_edge_index = torch.cat([current_edge_index, new_edge_tensor], dim=1)

    # 3. New graph
    updated_data = Data(x=current_x, edge_index=current_edge_index)

    # 4. Forward pass through GCN
    with torch.no_grad():
        out = model(updated_data.x, updated_data.edge_index)

    # 5. Predict labels for new nodes
    predictions = []
    for i in range(len(new_features_list)):
        node_idx = new_node_start_idx + i
        predictions.append(int(out[node_idx].argmax()))

    return predictions
# Let's say you want to classify 2 new papers

# Paper 1: cites nodes 0 and 10
# Paper 2: cites nodes 3 and 15

# Create fake features of same shape as data.x[0] (or use real vectors if available)
new_feats = [
    torch.randn(data.num_node_features),  # Paper 1
    torch.randn(data.num_node_features)   # Paper 2
]

new_edges = [
    [0, 10],  # Paper 1 citations
    [3, 15]   # Paper 2 citations
]

preds = predict_labels_for_unlabeled_papers(model, data, new_feats, new_edges)

for i, label in enumerate(preds):
    print(f"Predicted label for new paper {i+1}: Class {label}")
