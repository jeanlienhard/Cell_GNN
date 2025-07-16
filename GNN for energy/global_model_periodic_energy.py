"""
This script defines a LearnedSimulator class that uses a Graph Neural Network (GNN) to simulate the dynamics of particles.
The class includes methods for preprocessing input data, encoding it into a graph format, decoding the output,
and performing various geometric and physical computations.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from scipy.spatial import Voronoi
import gnn_model_energy
import numpy as np
from shapely.geometry import Polygon, box


class LearnedSimulator_periodic(nn.Module):
    """
    LearnedSimulator class that uses a GNN to simulate the dynamics of cells.
    """

    def __init__(self, num_dimensions, normalization_stats,device,n_cells):
        super(LearnedSimulator_periodic, self).__init__()
        self.device = device
        self.normalization_stats = normalization_stats
        self.graph_network = gnn_model_energy.EnergyGNN(edge_feature_size=1,hidden_size=512,latent_size=512)
        self.senders,self.receivers = self.fully_connected_edges(n_cells)

    def forward(self, position_sequence,n_cells):
        edge_index,edge_attr = self.encoder_preprocessor(position_sequence,n_cells)
        normalized_displacement = self.graph_network(edge_index,edge_attr)
        return normalized_displacement.squeeze(-1)
    
    def transform_velocity_to_principal_axes(self, position, velocity_sequence):
        """
        Transforms the velocity sequence to the principal axes with eigenvector.

        Args:
            position (torch.Tensor): Position tensor.
            velocity_sequence (torch.Tensor): Velocity sequence tensor.

        Returns:
            tuple: Transformed velocity sequence and rotation matrix.
        """
        centered_pos = position - position.mean(dim=0, keepdim=True)
        cov = centered_pos.T @ centered_pos
        cov = cov.float() 
        _, eigvecs = torch.linalg.eigh(cov)
        
        direction = centered_pos[torch.norm(centered_pos, dim=1).argmax()]
        if (eigvecs[:, 0] @ direction) > 0:
            eigvecs[:, 0] *= -1

        if torch.det(eigvecs) > 0:
            eigvecs[:, 1] *= -1

        velocity_rot = velocity_sequence @ eigvecs
        return velocity_rot, eigvecs
    
    def encoder_preprocessor(self, position_sequence,n_cells):
        """
        Preprocesses the input position sequence to create a graph representation.

        Args:
            position_sequence (torch.Tensor): Sequence of positions.
            step (int): Current step of the simulation.

        Returns:
            tuple: Data object containing node and edge features, and rotation matrix.
        """
        
        most_recent_position = position_sequence[:, -1]
        target_indices = np.arange(n_cells)
        """ Edge_index """
        senders, receivers = self.senders,self.receivers#self.fully_connected_edges(most_recent_position.cpu().detach().numpy(),target_indices)
        senders = senders.long()
        receivers = receivers.long()

        """ Edge_features """
        normalized_relative_displacements = (most_recent_position[receivers] - most_recent_position[senders])
        adjusted_displacements = ((normalized_relative_displacements + 1.0) % 2.0) - 1.0
        normalized_relative_distances = torch.norm(adjusted_displacements, dim=-1, keepdim=True)
        edge_index=torch.stack([senders, receivers])
        edge_attr=torch.exp(-2*normalized_relative_distances)
        # print(edge_attr)
        return edge_index, edge_attr


    def compute_connectivity_from_voronoi(self,most_recent_position,target_indices):
        """
        Computes the connectivity (senders and receivers) from the Voronoi diagram.

        Args:
            vor (scipy.spatial.Voronoi): Voronoi diagram object.

        Returns:
            tuple: Senders and receivers tensors.
        """
        n_cells = len(target_indices)
        senders = []
        receivers = []
        edges = set()
        vor = Voronoi(most_recent_position)
        for i, j in vor.ridge_points:
            point_i = most_recent_position[i]
            point_j = most_recent_position[j]
            mean = np.mean(most_recent_position)
            if np.all((point_i >= -2+mean) & (point_i <= 2+mean)) and np.all((point_j >= -2+mean) & (point_j <= 2+mean)):
                i_mod, j_mod = i % n_cells, j % n_cells
                if i_mod != j_mod and i_mod in target_indices and j_mod in target_indices:
                    edges.add(tuple(sorted((i_mod, j_mod))))
        senders, receivers = zip(*[(i, j) for i, j in edges] + [(j, i) for i, j in edges]) if edges else ([], [])
        return torch.tensor(senders).to(self.device), torch.tensor(receivers).to(self.device)

    def fully_connected_edges(self,n):
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                edges.add((i, j))
        senders, receivers = zip(*[(i, j) for i, j in edges] + [(j, i) for i, j in edges]) if edges else ([], [])
        return torch.tensor(senders).to(self.device), torch.tensor(receivers).to(self.device)