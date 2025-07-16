import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.datasets as datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Cora dataset
dataset = datasets.Planetoid(root='./data', name='Cora')
data = dataset[0]

# GCN Model Definition
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training and Testing Functions
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    total = data.test_mask.sum()
    return int(correct) / int(total)

# Train GCN
for epoch in range(1, 401):
    loss = train()
    if epoch % 20 == 0 or epoch == 400:
        acc = test()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

final_test_accuracy = test()
print(f'Final GCN Test Accuracy: {final_test_accuracy:.4f}')

# Detailed GCN Metrics
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
true = data.y[data.test_mask].cpu().numpy()
pred = pred[data.test_mask].cpu().numpy()

print("\nClassification Report:")
print(classification_report(true, pred, target_names=[f'Class {i}' for i in range(dataset.num_classes)]))

cm = confusion_matrix(true, pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(dataset.num_classes)], yticklabels=[f'Class {i}' for i in range(dataset.num_classes)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (GCN)')
plt.show()

# t-SNE Visualization (GCN Embeddings)
model.eval()
with torch.no_grad():
    embeddings = F.relu(model.conv1(data.x, data.edge_index))

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
labels = data.y.cpu().numpy()

plt.figure(figsize=(12, 10))
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette='tab10', s=30, alpha=0.7)
plt.title('t-SNE of GCN Node Embeddings')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.grid(True)
plt.show()

# t-SNE Visualization (Original Features)
tsne_orig = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_2d = tsne_orig.fit_transform(data.x.cpu().numpy())

plt.figure(figsize=(12, 10))
sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels, palette='tab10', s=30, alpha=0.7)
plt.title('t-SNE of Original Node Features')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.grid(True)
plt.show()
