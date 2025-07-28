import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import pandas as pd

# Charger les deux fichiers
df_initial = pd.read_csv("/home/jeanlienhard/Documents/Cell_GNN/GNN for energy/GNN_for _energy_target/positions_initiales_periodicite_20.csv")
df_final = pd.read_csv("/home/jeanlienhard/Documents/Cell_GNN/GNN for energy/GNN_for _energy_target/positions_optimisees_periodicite_20.csv")
n_cells = 20
def plot_voronoi(ax, points, title):
    vor = Voronoi(points)
    for i in range(n_cells):  # tracer seulement les cellules des 10 centres
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[j] for j in region]
            ax.fill(*zip(*polygon), alpha=0.5, edgecolor='black', facecolor='lightblue')
    # Marquer les 10 centres
    ax.plot(points[:n_cells,0], points[:n_cells,1], 'ro', markersize=5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plot_voronoi(axes[0], df_initial[['x','y']].values, "Initial Configuration")
plot_voronoi(axes[1], df_final[['x','y']].values, "Final Configuration")

plt.tight_layout()
plt.savefig("comparaison_initiale_finale.png")
plt.show()
