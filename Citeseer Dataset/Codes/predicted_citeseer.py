import torch
from torch_geometric.data import Data
import torch_geometric.datasets as datasets
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

print("Loading Citeseer dataset...")
dataset = datasets.Planetoid(root='./data', name='Citeseer')
data = dataset[0]
print("Citeseer dataset loaded.\n")

# Define GCN model
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

# Instantiate model
model = GCN(data.num_node_features, 16, dataset.num_classes)

# Function to predict new paper labels
def predict_labels_for_unlabeled_papers(model, original_data, new_features_list, new_edges_list):
    model.eval()
    current_x = original_data.x
    current_edge_index = original_data.edge_index

    new_node_start_idx = original_data.num_nodes
    all_new_edges = []

    for i, new_feat in enumerate(new_features_list):
        current_x = torch.cat([current_x, new_feat.unsqueeze(0)], dim=0)
        new_node_idx = new_node_start_idx + i
        cited_nodes = new_edges_list[i]

        for cited in cited_nodes:
            all_new_edges.append([new_node_idx, cited])
            all_new_edges.append([cited, new_node_idx])

    if all_new_edges:
        new_edge_tensor = torch.tensor(all_new_edges, dtype=torch.long).t()
        current_edge_index = torch.cat([current_edge_index, new_edge_tensor], dim=1)

    updated_data = Data(x=current_x, edge_index=current_edge_index)

    with torch.no_grad():
        out = model(updated_data.x, updated_data.edge_index)

    predictions = []
    for i in range(len(new_features_list)):
        node_idx = new_node_start_idx + i
        predictions.append(int(out[node_idx].argmax()))

    return predictions

# Example new papers
new_feats = [
    torch.randn(data.num_node_features),
    torch.randn(data.num_node_features)
]

new_edges = [
    [5, 20],  # Paper 1 cites nodes 5 and 20
    [8, 14]   # Paper 2 cites nodes 8 and 14
]

# Predict
preds = predict_labels_for_unlabeled_papers(model, data, new_feats, new_edges)

for i, label in enumerate(preds):
    print(f"Predicted label for new paper {i+1}: Class {label}")