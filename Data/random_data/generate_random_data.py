"""
This script generates a dataset of random positions.
The positions are generated randomly within a specified domain.
"""

import csv
import numpy as np

n_points = 10
n_steps = 220


x_offset = 2.0
y_offset = 2.0


all_rows = []


for step in range(n_steps):
    site_index = 0
    x_values = np.random.uniform(-1, 1, n_points)
    y_values = np.random.uniform(-1, 1, n_points)

    
    for i in range(n_points):
        x = x_values[i]
        y = y_values[i]
        all_rows.append([step, site_index, x, y, 0])
        site_index += 1

    for gx in range(-2, 3):
        for gy in range(-2, 3):
            if gx != 0 or gy != 0:
                for i in range(n_points):
                    x = x_values[i] + gx * x_offset
                    y = y_values[i] + gy * y_offset
                    all_rows.append([step, site_index, x, y, 0])
                    site_index += 1

# Sauvegarde dans un CSV
with open("Data/random_data/random_positions_5.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "site_index", "x", "y", "is_removed"])
    writer.writerows(all_rows)
