"""
This script defines an Encode-Process-Decode model for a learnable simulator using Graph Neural Networks (GNNs), inspired from https://arxiv.org/pdf/2002.09405.
The model consists of an encoder to transform input features, a processor to perform message passing,
and a decoder to produce the final output. The script includes various helper classes and functions
to build and manage the neural networks.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class edgeToNode(MessagePassing):
    def __init__(self, edge_feat_dim, out_channels):
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, edge_index, edge_attr, num_nodes):
        dummy_x = torch.zeros((num_nodes, 1), device=edge_attr.device)
        return self.propagate(edge_index=edge_index, x=dummy_x, edge_attr=edge_attr)

    def message(self, edge_attr):
        return self.edge_mlp(edge_attr)

    def update(self, aggr_out):
        return aggr_out

class NodeGNN(MessagePassing):
    def __init__(self, in_channels, edge_feat_dim, out_channels):
        super().__init__(aggr='mean')

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_feat_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=1)
        return self.edge_mlp(msg_input)

    def update(self, aggr_out, x):
        return self.lin_self(x) + aggr_out

class EnergyGNN(nn.Module):
    def __init__(self, edge_feature_size,latent_size, hidden_size):
        super().__init__()
        self.edge_to_node = edgeToNode(edge_feature_size, latent_size)
        self.gnn_layer1 = NodeGNN(hidden_size,edge_feature_size ,hidden_size)
        self.gnn_layer2 = NodeGNN(hidden_size,edge_feature_size ,hidden_size)
        self.gnn_layer3 = NodeGNN(hidden_size,edge_feature_size ,hidden_size)
        self.gnn_layer4 = NodeGNN(hidden_size,edge_feature_size ,hidden_size)
        self.gnn_layer5 = NodeGNN(hidden_size,edge_feature_size ,hidden_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, edge_index, edge_attr):
        num_nodes = edge_index.max().item() + 1
        x = self.edge_to_node(edge_index, edge_attr, num_nodes)
        x = self.gnn_layer1(x, edge_index, edge_attr).relu()
        x = self.gnn_layer2(x, edge_index, edge_attr).relu()
        x = self.gnn_layer3(x, edge_index, edge_attr).relu()
        x = self.gnn_layer4(x, edge_index, edge_attr).relu()
        x = self.gnn_layer5(x, edge_index, edge_attr)
        return self.regressor(x)
