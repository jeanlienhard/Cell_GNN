"""
This script loads two CSV files containing the initial and optimized positions of points.
It then computes and plots the Voronoi diagrams for both configurations, 
highlighting the first 10 cells in each diagram. 
The plots help visualize how the Voronoi cells change after optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import pandas as pd


df_initial = pd.read_csv("/home/jeanlienhard/Documents/Cell_GNN/GNN for energy/initial_positions_periodicity.csv")
df_final = pd.read_csv("/home/jeanlienhard/Documents/Cell_GNN/GNN for energy/optimized_positions_periodicity.csv")

def plot_voronoi(ax, points, title):
    vor = Voronoi(points)
    for i in range(10): 
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[j] for j in region]
            ax.fill(*zip(*polygon), alpha=0.5, edgecolor='black', facecolor='lightblue')
    ax.plot(points[:10,0], points[:10,1], 'ro', markersize=5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)

initial_points = df_initial[['x','y']].values[:10]
final_points = df_final[['x','y']].values[:10]

all_10_points = np.vstack([initial_points, final_points])
x_min, x_max = all_10_points[:,0].min(), all_10_points[:,0].max()
y_min, y_max = all_10_points[:,1].min(), all_10_points[:,1].max()

margin_x = 0.6
margin_y = 0.6
x_min -= margin_x
x_max += margin_x
y_min -= margin_y
y_max += margin_y

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, points, title in zip(axes, 
                             [df_initial[['x','y']].values, df_final[['x','y']].values],
                             ["Initial Configuration", "Final Configuration"]):
    plot_voronoi(ax, points, title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
