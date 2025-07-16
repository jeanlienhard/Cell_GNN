"""
This script defines a LearnedSimulator class that uses a Graph Neural Network (GNN) to simulate the dynamics of particles.
The class includes methods for preprocessing input data, encoding it into a graph format, decoding the output,
and performing various geometric and physical computations.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from scipy.spatial import Voronoi
import gnn_model
import numpy as np
from shapely.geometry import Polygon, box 
from torch_geometric.utils import add_self_loops
from sklearn.neighbors import NearestNeighbors 


class LearnedSimulator_periodic(nn.Module):
    """
    LearnedSimulator class that uses a GNN to simulate the dynamics of cells.
    """

    def __init__(self, num_dimensions, normalization_stats,device):
        super(LearnedSimulator_periodic, self).__init__()
        self.device = device
        self.normalization_stats = normalization_stats
        self.graph_network = gnn_model.EncodeProcessDecode(edge_features_size = 1, nodes_features_size = 22,mlp_hidden_size=256,mlp_num_hidden_layers=2,num_message_passing_steps=15,output_size=num_dimensions)

    def forward(self, position_sequence,n_cells):
        input_graphs_tuple = self.encoder_preprocessor(position_sequence,n_cells)
        normalized_displacement = self.graph_network(input_graphs_tuple)
        next_position = self.decoder_postprocessor(normalized_displacement, position_sequence,n_cells)
        return next_position,normalized_displacement

    def compute_polygon_area_and_perimeter(self,polygon):
        polygon = np.array(polygon)
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        perimeter = np.sum(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1))
        return area, perimeter

    def vornoi_area_and_perimeter(self,vor,target_indices):
        """
        Computes the area and perimeter of Voronoi regions.

        Args:
            vor (scipy.spatial.Voronoi): Voronoi diagram object.

        Returns:
            tuple: Areas and perimeters of the Voronoi regions.
        """
        areas = []
        perimeters = []
        for idx in target_indices:
            region_index = vor.point_region[idx]
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0:
                areas.append(1e-10)
                perimeters.append(1e-10)
                continue
            polygon = [vor.vertices[i] for i in region]
            area, perimeter = self.compute_polygon_area_and_perimeter(polygon)
            areas.append(area)
            perimeters.append(perimeter)          
        return areas,perimeters
    
    def encoder_preprocessor(self, position_sequence,n_cells):
        """
        Preprocesses the input position sequence to create a graph representation.

        Args:
            position_sequence (torch.Tensor): Sequence of positions.
            step (int): Current step of the simulation.

        Returns:
            tuple: Data object containing node and edge features.
        """
        
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = self.time_diff(position_sequence)
        velocity_sequence = velocity_sequence[:n_cells]
        target_indices = np.arange(n_cells)
        """ Edge_index """
        senders, receivers = self.compute_connectivity_knn(most_recent_position.cpu().detach().numpy(),target_indices)
        # senders, receivers = self.compute_connectivity_from_voronoi(most_recent_position.cpu().detach().numpy(),target_indices)
        senders = senders.long()
        receivers = receivers.long()
        """ Nodes Features """
        node_features = []
        # Velocity
        velocity_stats = self.normalization_stats["velocity"]
        velocity_sequence = (velocity_sequence - velocity_stats["mean"]) / velocity_stats["std"]
        velocity_norm = torch.linalg.vector_norm(velocity_sequence, dim = -1) 
        node_features.append(velocity_norm)
        velocity_sequence = velocity_sequence.reshape(velocity_sequence.size(0),-1)
        node_features.append(velocity_sequence)
        
        # area, perimeters
        areas = []
        perimeters = []
        for i in range(position_sequence.shape[1]):
            position = position_sequence[:,i]
            vor = Voronoi(position.cpu().detach().numpy())
            area,perimeter = self.vornoi_area_and_perimeter(vor,target_indices)
            areas.append(area)
            perimeters.append(perimeter)
        
        areas_tensor = torch.tensor(np.array(areas), dtype=torch.float32).to(self.device)
        areas_tensor = areas_tensor.reshape(areas_tensor.size(0),-1)
        areas_tensor = areas_tensor.T
        perimeters_tensor = torch.tensor(np.array(perimeters),dtype=torch.float32).to(self.device)
        perimeters_tensor = perimeters_tensor.reshape(perimeters_tensor.size(0),-1)
        perimeters_tensor = perimeters_tensor.T
        node_features.append(areas_tensor)
        node_features.append(perimeters_tensor)


        """ Edge_features """
        # distance, comparative velocity
        edge_features = []
        normalized_relative_displacements = (most_recent_position[receivers] - most_recent_position[senders])
        adjusted_displacements = ((normalized_relative_displacements + 1.0) % 2.0) - 1.0
        normalized_relative_distances = torch.norm(adjusted_displacements, dim=-1, keepdim=True)
        edge_features.append(torch.exp(-2*normalized_relative_distances))

        edge_index=torch.stack([senders, receivers])
        edge_attr=torch.cat(edge_features, dim=-1)
        return Data(x=torch.cat(node_features, dim=-1), edge_index=edge_index, edge_attr=edge_attr)

    def decoder_postprocessor(self, normalized_acceleration, position_sequence,n_cells):
        """
        Postprocesses the output of the GNN to compute the next position.

        Args:
            normalized_acceleration (torch.Tensor): Normalized acceleration tensor.
            position_sequence (torch.Tensor): Sequence of positions.

        Returns:
            torch.Tensor: Next position tensor.
        """
        acceleration_stats = self.normalization_stats["acceleration"]
        acceleration = normalized_acceleration * acceleration_stats["std"] + acceleration_stats["mean"]
        acceleration = acceleration
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]
        new_velocity= most_recent_velocity + acceleration.repeat(position_sequence.shape[0]//n_cells,1)
        new_position = most_recent_position + new_velocity
        return new_position

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
        for i in target_indices:
            edges.add((i,i))
        senders, receivers = zip(*[(i, j) for i, j in edges] + [(j, i) for i, j in edges]) if edges else ([], [])
        return torch.tensor(senders).to(self.device), torch.tensor(receivers).to(self.device)
    
    def compute_connectivity_knn(self, most_recent_position, target_indices, k=5):
        """
        Computes the connectivity (senders and receivers) using the k-nearest neighbors graph.

        Args:
            most_recent_position (np.ndarray): Positions of the nodes, shape (N, 2).
            target_indices (list or np.ndarray): Indices of the nodes to keep.
            k (int): Number of nearest neighbors to connect.

        Returns:
            tuple: (senders, receivers) as torch tensors.
        """
        n_cells = len(target_indices)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(most_recent_position)
        _, indices = nbrs.kneighbors(most_recent_position)
        edges = set()
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:
                point_i = most_recent_position[i]
                point_j = most_recent_position[j]
                mean = np.mean(most_recent_position)
                if np.all((point_i >= -2+mean) & (point_i <= 2+mean)) and np.all((point_j >= -2+mean) & (point_j <= 2+mean)):
                    i_mod, j_mod = i % n_cells, j % n_cells
                    if i_mod != j_mod and i_mod in target_indices and j_mod in target_indices:
                        edges.add((i_mod, j_mod))
        for i in target_indices:
            edges.add((i,i))
        senders, receivers = zip(*[(i, j) for i, j in edges]) if edges else ([], [])
        return torch.tensor(senders).to(self.device), torch.tensor(receivers).to(self.device)

    def time_diff(self, input_sequence):
        """
        Computes the velocity.

        Args:
            input_sequence (torch.Tensor): Input sequence of positions.

        Returns:
            torch.Tensor: Time difference tensor.
        """
        return input_sequence[:, 1:] - input_sequence[:, :-1]
