"""
This script loads 2D position data of multiple cells from a CSV file, 
computes their trajectories and second-order finite difference approximations 
of acceleration, and visualizes both the trajectories and accelerations.
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt


class CellTrajectoryDataset:
    def __init__(self, df_cell, step):
        df_cell = df_cell.sort_values('step')
        positions = []
        accelerations = []

        for step_id in range(step):
            data_step = df_cell[df_cell['step'] == step_id]
            if data_step.empty:
                continue
            pos = torch.tensor(data_step[['x', 'y']].values[0], dtype=torch.float64)
            positions.append(pos)
            if step_id > 7:
                acc = positions[-1] - 2 * positions[-2] + positions[-3]
                accelerations.append(acc)

        self.trajectory = torch.stack(positions)
        self.acceleration = torch.stack(accelerations)


steps = 300
csv_file_path = 'Data/raw_data/positions_1.csv'
n_cells = 10


df = pd.read_csv(csv_file_path)
site_indices = df['site_index'].unique()[:n_cells]

datasets = []
for idx in site_indices:
    df_cell = df[df['site_index'] == idx]
    dataset = CellTrajectoryDataset(df_cell, step=steps)
    datasets.append((idx, dataset))


# Trajectories
fig1, axs1 = plt.subplots(5, 2, figsize=(20, 15))
fig1.suptitle("Trajectories of 10 cells", fontsize=16)

for ax, (idx, dataset) in zip(axs1.flatten(), datasets):
    ax.plot(dataset.trajectory[:, 0], dataset.trajectory[:, 1], marker='o', markersize=2)
    ax.set_title(f'Cell {idx}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Acceleration
fig2, axs2 = plt.subplots(5, 2, figsize=(20, 15))
fig2.suptitle("Acceleration of 10 cells", fontsize=16)

for ax, (idx, dataset) in zip(axs2.flatten(), datasets):
    ax.plot(dataset.acceleration[:, 0], label='a_x')
    ax.plot(dataset.acceleration[:, 1], label='a_y')
    ax.set_title(f'Cell {idx}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Acceleration')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
