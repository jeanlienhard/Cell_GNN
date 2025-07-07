"""
This script is mainly used to display the distribution of cell areas  
at a given time step. It builds the Voronoi diagram from positions  
read from a CSV file, clips it to a bounding box, and computes  
the area and perimeter of each cell.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import torch

class VoronoiAnalyzer:
    def voronoi_finite_polygons_2d(self, vor, radius=100):
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        all_ridges = {}

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if all(v >= 0 for v in region):
                new_regions.append(region)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = vor.points[p2] - vor.points[p1] 
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) 

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = [v for _, v in sorted(zip(angles, new_region))]

            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)

    def voronoi_area_and_perimeter(self, vor):
        regions, vertices = self.voronoi_finite_polygons_2d(vor)
        bbox = [-1, 1, -1, 1]
        clipping_box = box(bbox[0], bbox[2], bbox[1], bbox[3])

        areas = []
        perimeters = []

        for region in regions:
            if not region or any(v >= len(vertices) or v < 0 for v in region):
                continue

            polygon = vertices[region]
            poly_shape = Polygon(polygon)
            clipped_polygon = poly_shape.intersection(clipping_box)

            if not clipped_polygon.is_empty:
                areas.append(clipped_polygon.area)
                perimeters.append(clipped_polygon.length)
            else:
                areas.append(1e-10)
                perimeters.append(1e-10)
        return areas, perimeters


def analyze_voronoi_from_csv(csv_path, step_index):
    df = pd.read_csv(csv_path)
    df_step = df[df['step'] == step_index]
    points = df_step[['x', 'y']].to_numpy()

    vor = Voronoi(points)
    analyzer = VoronoiAnalyzer()
    areas, perimeters = analyzer.voronoi_area_and_perimeter(vor)

    plt.hist(areas[:10], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Area distribution (step={step_index})')
    plt.xlabel('Aire')
    plt.ylabel('Number of cells')
    plt.grid(True)
    plt.show()

    return areas, perimeters

analyze_voronoi_from_csv("Data/raw_data/positions_1.csv",199)
