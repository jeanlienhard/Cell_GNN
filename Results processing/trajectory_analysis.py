"""
This script analyzes cell position data from a CSV file to compute and visualize 
various physical properties over time, including Voronoi cell areas, perimeters, 
accelerations, and associated energy terms (mechanical + kinetic).

It uses a physical model based on deviations from a target area and basic mechanics 
to estimate per-cell and total energies. The results are plotted to observe 
temporal dynamics and variability across cells.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.spatial import Voronoi

def compute_polygon_area_and_perimeter(polygon):
    polygon = np.array(polygon)
    x = polygon[:, 0]
    y = polygon[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    perimeter = np.sum(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1))
    return area, perimeter

def vornoi_area_and_perimeter(vor, target_indices):
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
        area, perimeter = compute_polygon_area_and_perimeter(polygon)
        areas.append(area)
        perimeters.append(perimeter)
    return np.array(areas), np.array(perimeters)

def compute_acceleration(df, step, delta_t=1.0):
    p_prev = df[df["step"] == step - 2][["x", "y"]].to_numpy()
    p_curr = df[df["step"] == step - 1][["x", "y"]].to_numpy()
    p_next = df[df["step"] == step][["x", "y"]].to_numpy()
    if len(p_prev) != len(p_curr) or len(p_curr) != len(p_next):
        return None
    return (p_next - 2 * p_curr + p_prev) / (delta_t ** 2)

def run_data_analysis(csv_path, n_cells, delta_t=1.0, target_area=0.4, dt=0.05, masse=0.1):
    df = pd.read_csv(csv_path)
    steps = df["step"].unique()
    total_energies = []
    per_cell_energies = []
    per_cell_areas = []
    per_cell_perimeters = []
    per_cell_accelerations = []
    valid_steps = []

    target_indices = np.arange(n_cells)

    for step in steps[2:220]:
        acc = compute_acceleration(df, step, delta_t)
        if acc is None:
            continue
        positions = df[df["step"] == step][["x", "y"]].to_numpy()
        vor = Voronoi(positions)
        areas, perimeters = vornoi_area_and_perimeter(vor, target_indices)
        physics_energy = 0.02 * (areas - target_area)**2 + 1e-5 * (target_area / areas)**2
        kinetic_energy = 0.5 * masse * np.sum((acc[:n_cells]/dt**2)**2, axis=1) * dt**2
        energy = physics_energy + kinetic_energy
        total_energy = np.sum(energy)

        acc_norms = np.linalg.norm(acc[:n_cells], axis=1)
        per_cell_accelerations.append(acc_norms)

        total_energies.append(total_energy)
        per_cell_energies.append(energy)
        per_cell_areas.append(areas)
        per_cell_perimeters.append(perimeters)
        valid_steps.append(step)

    per_cell_energies = np.array(per_cell_energies)
    per_cell_areas = np.array(per_cell_areas)
    per_cell_perimeters = np.array(per_cell_perimeters)
    per_cell_accelerations = np.array(per_cell_accelerations)
    valid_steps = np.array(valid_steps)

    mean_energy = per_cell_energies.mean(axis=1)
    min_energy = per_cell_energies.min(axis=1)
    max_energy = per_cell_energies.max(axis=1)

    mean_area = per_cell_areas.mean(axis=1)
    min_area = per_cell_areas.min(axis=1)
    max_area = per_cell_areas.max(axis=1)

    mean_perim = per_cell_perimeters.mean(axis=1)
    min_perim = per_cell_perimeters.min(axis=1)
    max_perim = per_cell_perimeters.max(axis=1)

    mean_acc = per_cell_accelerations.mean(axis=1)
    min_acc = per_cell_accelerations.min(axis=1)
    max_acc = per_cell_accelerations.max(axis=1)

    fig, axs = plt.subplots(3, 2, figsize=(14, 13))
    axs = axs.flatten()

    # 1. Total Energy
    axs[0].plot(valid_steps, total_energies, color='blue')
    axs[0].set_title("Total Energy Over Time")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Total Energy")
    axs[0].grid(True)

    # 2. Cell Energy
    for i in range(n_cells):
        axs[1].plot(valid_steps, per_cell_energies[:, i], color='gray', alpha=0.3, linewidth=0.8)
    axs[1].plot(valid_steps, mean_energy, color='green', label='Mean Energy')
    axs[1].fill_between(valid_steps, min_energy, max_energy, alpha=0.2, color='green', label='Min–Max Range')
    axs[1].set_title("Cell Energy (All) + Mean ± Min/Max")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Energy")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Cell Area
    for i in range(n_cells):
        axs[2].plot(valid_steps, per_cell_areas[:, i], color='gray', alpha=0.3, linewidth=0.8)
    axs[2].plot(valid_steps, mean_area, color='purple', label='Mean Area')
    axs[2].fill_between(valid_steps, min_area, max_area, alpha=0.2, color='purple', label='Min–Max Range')
    axs[2].set_title("Cell Area (All) + Mean ± Min/Max")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Area")
    axs[2].legend()
    axs[2].grid(True)

    # 4. Cell Perimeter
    for i in range(n_cells):
        axs[3].plot(valid_steps, per_cell_perimeters[:, i], color='gray', alpha=0.3, linewidth=0.8)
    axs[3].plot(valid_steps, mean_perim, color='orange', label='Mean Perimeter')
    axs[3].fill_between(valid_steps, min_perim, max_perim, alpha=0.2, color='orange', label='Min–Max Range')
    axs[3].set_title("Cell Perimeter (All) + Mean ± Min/Max")
    axs[3].set_xlabel("Step")
    axs[3].set_ylabel("Perimeter")
    axs[3].legend()
    axs[3].grid(True)

    # 5. Cell Acceleration
    for i in range(n_cells):
        axs[4].plot(valid_steps, per_cell_accelerations[:, i], color='gray', alpha=0.3, linewidth=0.8)
    axs[4].plot(valid_steps, mean_acc, color='red', label='Mean Acceleration')
    axs[4].fill_between(valid_steps, min_acc, max_acc, alpha=0.2, color='red', label='Min–Max Range')
    axs[4].set_title("Cell Acceleration (All) + Mean ± Min/Max")
    axs[4].set_xlabel("Step")
    axs[4].set_ylabel("Acceleration (norm)")
    axs[4].legend()
    axs[4].grid(True)

    axs[5].axis('off')

    plt.tight_layout()
    plt.show()


run_data_analysis("/home/jeanlienhard/Documents/Cell_GNN/GNN for acceleration/Unsupervised/Training/unsupervised_15Conv_256_voronoi/computed_trajectory/computed_positions_5.csv", 10)
