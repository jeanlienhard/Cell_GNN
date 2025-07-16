"""
This script defines an Encode-Process-Decode model for a learnable simulator using Graph Neural Networks (GNNs), inspired from https://arxiv.org/pdf/2002.09405.
The model consists of an encoder to transform input features, a processor to perform message passing,
and a decoder to produce the final output. The script includes various helper classes and functions
to build and manage the neural networks.
"""

from typing import Callable
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

def build_mlp(input_size:int,hidden_size: int, num_hidden_layers: int, output_size: int) -> nn.Sequential:
    """
    Builds a Multi-Layer Perceptron (MLP).

    Args:
        input_size (int): Size of the input layer.
        hidden_size (int): Size of the hidden layers.
        num_hidden_layers (int): Number of hidden layers.
        output_size (int): Size of the output layer.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    for _ in range(num_hidden_layers-2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)

class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode function approximator for learnable simulator."""

    def __init__(
        self,
        edge_features_size:int,
        nodes_features_size:int,
        mlp_hidden_size: int,
        mlp_num_hidden_layers: int,
        num_message_passing_steps: int,
        output_size: int,
    ):
        """Inits the model."""
        super().__init__()
        self._edge_features_size = edge_features_size
        self._nodes_features_size = nodes_features_size
        self._latent_size = mlp_hidden_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size

        self._networks_builder()

    def forward(self, input_graph: Data) -> torch.Tensor:
        """Forward pass of the learnable dynamics model."""
        graph_0 = self._encode(input_graph)
        graph_m = self._process(graph_0)
        return self._decode(graph_m)

    def _networks_builder(self):
        """Builds the networks."""

        def build_mlp_with_layer_norm(input_size = self._mlp_hidden_size,output_size = self._latent_size):
            mlp = build_mlp(
                input_size=input_size,
                hidden_size=self._mlp_hidden_size,
                num_hidden_layers=self._mlp_num_hidden_layers,
                output_size=output_size 
            )
            return nn.Sequential(mlp, nn.LayerNorm(output_size))

        self._encoder_network = GraphIndependent(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm,
            edge_features_size = self._edge_features_size,
            nodes_features_size = self._nodes_features_size,
        )

        self._processor_networks = nn.ModuleList([
            GNNModel(hidden_size= self._mlp_hidden_size,num_gcn_layers=self._num_message_passing_steps,edge_feat_dim= self._edge_features_size)
        ])

        self._decoder_network = build_mlp(
            input_size= self._latent_size,
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size
        )

    def _encode(self, input_graph: Data) -> Data:
        """Encodes the input graph features into a encoded graph."""
        encoded_graph = self._encoder_network(input_graph)
        return encoded_graph

    def _process(self, graph_0: Data) -> Data:
        """Processes the latent graph with several steps of message passing."""
        x, edge_index,edge_attr = graph_0.x, graph_0.edge_index,graph_0.edge_attr
        for conv in self._processor_networks:
            x = conv(x=x,edge_index = edge_index,edge_attr = edge_attr)
        graph_0.x = x
        return graph_0

    def _decode(self, graph: Data) -> torch.Tensor:
        """Decodes from the latent graph."""
        return self._decoder_network(graph.x)

class GraphIndependent(nn.Module):
    """Graph Independent Network that independently encodes edge and node features."""
    def __init__(self, edge_model_fn: Callable, node_model_fn: Callable, edge_features_size:int,nodes_features_size:int):
        super().__init__()
        self.edge_model = edge_model_fn(edge_features_size,1)
        self.node_model = node_model_fn(nodes_features_size)

    def forward(self, data: Data) -> Data:
        data.edge_attr = self.edge_model(data.edge_attr)
        data.x = self.node_model(data.x)
        return data

class NodeGNN(MessagePassing):
    def __init__(self, in_channels, edge_feat_dim, out_channels):
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_feat_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=1)
        return self.edge_mlp(msg_input)

    def update(self, aggr_out, x):
        return self.lin_self(x) + aggr_out

class GNNModel(nn.Module):
    """GNN Model with Graph Convolutional Networks (GCNs)"""
    def __init__(self, hidden_size, num_gcn_layers,edge_feat_dim):
        super(GNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(NodeGNN(hidden_size,edge_feat_dim,hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.gcn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        for gcn in self.gcn_layers:            
            x = gcn(x, edge_index,edge_attr)
        return x