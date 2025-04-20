import torch
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data

class LocalGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=True, dropout=0.1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)  # One feature vector per graph
        return x

class GlobalGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, edge_dim=1, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads,
                               dropout=0.1, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1,
                               concat=True, dropout=0.1, edge_dim=edge_dim)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = x.mean(dim=0)  # Global mean pooling
        return self.classifier(x)