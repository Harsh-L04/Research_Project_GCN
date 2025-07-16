# Imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# GCN Definition
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Load Citeseer dataset
print("\n--- Loading Citeseer dataset ---")
citeseer_dataset = Planetoid(root='./data', name='Citeseer')
citeseer_data = citeseer_dataset[0]

# Initialize model, optimizer, loss
citeseer_model = GCN(in_channels=citeseer_data.num_node_features,
                     hidden_channels=16,
                     out_channels=citeseer_dataset.num_classes)
citeseer_optimizer = torch.optim.Adam(citeseer_model.parameters(), lr=0.01)
citeseer_criterion = torch.nn.CrossEntropyLoss()

# Train and Test Functions
def train():
    citeseer_model.train()
    citeseer_optimizer.zero_grad()
    out = citeseer_model(citeseer_data.x, citeseer_data.edge_index)
    loss = citeseer_criterion(out[citeseer_data.train_mask], citeseer_data.y[citeseer_data.train_mask])
    loss.backward()
    citeseer_optimizer.step()
    return loss.item()

def test():
    citeseer_model.eval()
    with torch.no_grad():
        out = citeseer_model(citeseer_data.x, citeseer_data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[citeseer_data.test_mask] == citeseer_data.y[citeseer_data.test_mask]).sum()
    total = citeseer_data.test_mask.sum()
    return int(correct) / int(total), pred

# Training Loop
print("\n--- Training GCN on Citeseer ---")
for epoch in range(1, 401):
    loss = train()
    if epoch % 20 == 0 or epoch == 400:
        acc, _ = test()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

# Final Test Evaluation
final_acc, final_pred = test()
print(f"\n✅ Final Test Accuracy on Citeseer: {final_acc:.4f}")

# Classification Report and Confusion Matrix
true_labels = citeseer_data.y[citeseer_data.test_mask].cpu().numpy()
pred_labels = final_pred[citeseer_data.test_mask].cpu().numpy()

print("\n--- Classification Report ---")
print(classification_report(true_labels, pred_labels, target_names=[f"Class {i}" for i in range(citeseer_dataset.num_classes)]))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Citeseer - GCN)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# Get intermediate layer embeddings
citeseer_model.eval()
with torch.no_grad():
    h1 = citeseer_model.conv1(citeseer_data.x, citeseer_data.edge_index)
    embeddings = F.relu(h1)

embeddings_np = embeddings.cpu().numpy()
labels_np = citeseer_data.y.cpu().numpy()

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_np)

# Plot
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=labels_np,
    palette=sns.color_palette("tab10", n_colors=citeseer_dataset.num_classes),
    legend='full',
    alpha=0.7,
    s=30
)
plt.title('t-SNE Visualization of GCN Node Embeddings (Citeseer)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



# Reduce dimensionality before t-SNE to prevent memory issues
print("\n--- Running t-SNE on Original Features ---")
original_features_np = citeseer_data.x.cpu().numpy()
pca = PCA(n_components=50)  # reduce to 50D first
features_pca = pca.fit_transform(original_features_np)

tsne_orig = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_2d = tsne_orig.fit_transform(features_pca)

labels = citeseer_data.y.cpu().numpy()

plt.figure(figsize=(12, 10))
sns.scatterplot(
    x=features_2d[:, 0],
    y=features_2d[:, 1],
    hue=labels,
    palette=sns.color_palette("tab10", n_colors=citeseer_dataset.num_classes),
    legend='full',
    alpha=0.7,
    s=30
)
plt.title('t-SNE of Original Node Features (Citeseer)')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


