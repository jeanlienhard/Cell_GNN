"""
This script analyzes Voronoi cells from a CSV file containing point positions 
at a given time step. It computes the area and perimeter of each Voronoi cell, 
prints the sum of the areas of the first 10 cells, and shows a histogram 
of these areas to visualize their distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

class VoronoiAnalyzer:
    def compute_polygon_area_and_perimeter(self, polygon):
        polygon = np.array(polygon)
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        perimeter = np.sum(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1))
        return area, perimeter

    def voronoi_area_and_perimeter(self, vor, target_indices):
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
        return areas, perimeters

def analyze_voronoi_from_csv(csv_path, step_index):
    df = pd.read_csv(csv_path)
    df_steps = df[df['step']== step_index]
    points = df_steps[['x','y']].to_numpy()
    print(df_steps.head())
    vor = Voronoi(points)
    analyzer = VoronoiAnalyzer()
    target_indices = np.arange(len(points))
    areas, perimeters = analyzer.voronoi_area_and_perimeter(vor, target_indices)

    print(f"Total area (sum of first 10 cells): {np.sum(areas[:10])}")
    plt.figure(figsize=(6,4))
    plt.hist(areas[:10], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Area distribution')
    plt.xlabel('Area')
    plt.ylabel('Number of cells')
    plt.grid(True)
    plt.show()

    return areas, perimeters

analyze_voronoi_from_csv(
    "/home/jeanlienhard/Documents/Cell_GNN/GNN for acceleration/Unsupervised/Training/unsupervised_15Conv_256_voronoi/computed_trajectory/computed_positions_12.csv", 
    step_index=1000
)
