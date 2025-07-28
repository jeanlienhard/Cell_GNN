import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import pandas as pd
from matplotlib.animation import FuncAnimation


# Fonction utilitaire pour tracer une cellule Voronoi
def plot_voronoi_cell(ax, vor, point_index, **kwargs):
    region_index = vor.point_region[point_index]
    region = vor.regions[region_index]
    if not -1 in region and len(region) > 0:  # Cellule fermée
        polygon = [vor.vertices[i] for i in region]
        ax.fill(*zip(*polygon), **kwargs)
# Remplacer par ton propre DataFrame
# Exemple fictif :
data = pd.read_csv("/home/jeanlienhard/Documents/Cell_GNN/GNN for energy/GNN_for _energy_target/trajectories_20.csv")
steps = data["step"].unique()
fig, ax = plt.subplots(figsize=(6, 6))
# Indices des sites qu'on veut afficher (0 à 19)
target_indices = np.arange(20)

def update(frame):
    """
    Update function for the animation.
    """
    ax.clear()
    step = steps[frame]
    data_step = data[data["step"] == step]
    # Récupérer tous les points
    all_points = data_step[['x', 'y']].values

    # Calculer le diagramme de Voronoi global
    vor = Voronoi(all_points)

    # Tracer uniquement les cellules des 20 premiers points
    for idx in target_indices:
        plot_voronoi_cell(ax, vor, idx, alpha=0.5, edgecolor='black', facecolor='lightblue')

    # Mettre en évidence les 20 premiers
    ax.plot(all_points[target_indices, 0], all_points[target_indices, 1], 'ro', markersize=5)
    # ax.set_xlim(-7.0, 7.0)
    # ax.set_ylim(-7.0, 7.0)
    ax.set_title(f"Voronoï - Step {step}")
    ax.grid(True)

ani = FuncAnimation(
    fig, update,
    frames=70,#len(steps),#100,
    interval=100,
    repeat=False
)
ani.save('test_lbfgs.gif')

plt.show()
