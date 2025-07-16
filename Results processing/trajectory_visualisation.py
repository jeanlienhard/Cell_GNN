"""
This script loads cell position data from a CSV file and visualizes their evolution 
over time using Voronoi diagrams. It creates an animated plot showing the Voronoi 
cells of a selected subset of points (cells of interest) at each time step.

The animation is saved as a GIF and displayed at the end.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import pandas as pd
from matplotlib.animation import FuncAnimation


def plot_voronoi_cell(ax, vor, point_index, **kwargs):
    region_index = vor.point_region[point_index]
    region = vor.regions[region_index]
    if not -1 in region and len(region) > 0:  
        polygon = [vor.vertices[i] for i in region]
        ax.fill(*zip(*polygon), **kwargs)

data = pd.read_csv("/home/jeanlienhard/Documents/Cell_GNN/GNN for acceleration/Supervised/Training/acceleration_15Conv_256_5knn/computed_trajectory/computed_positions_13.csv")
steps = data["step"].unique()
fig, ax = plt.subplots(figsize=(6, 6))

n_cells = 10
target_indices = np.arange(n_cells)

def update(frame):
    """
    Update function for the animation.
    """
    ax.clear()
    step = steps[frame]
    data_step = data[data["step"] == step]
    all_points = data_step[['x', 'y']].values

    vor = Voronoi(all_points)

    for idx in target_indices:
        plot_voronoi_cell(ax, vor, idx, alpha=0.5, edgecolor='black', facecolor='lightblue')

    ax.plot(all_points[target_indices, 0], all_points[target_indices, 1], 'ro', markersize=5)
    ax.set_title(f"Voronoï - Step {step}")
    ax.grid(True)

ani = FuncAnimation(
    fig, update,
    frames=len(steps),
    interval=100,
    repeat=False
)
ani.save('/home/jeanlienhard/Documents/Cell_GNN/GNN for acceleration/Supervised/Training/acceleration_15Conv_256_5knn/computed_trajectory/animation_computed_trajectory_13.gif')

plt.show()
