import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
n_cells = 20
class VoronoiAnalyzer:
    def compute_polygon_area_and_perimeter(self, polygon):
        polygon = np.array(polygon)
        # Aire (Green's theorem)
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # Périmètre
        perimeter = np.sum(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1))
        return area, perimeter

    def voronoi_area_and_perimeter(self, vor, target_indices):
        """
        Computes the area and perimeter of Voronoi regions for specific points.

        Args:
            vor (scipy.spatial.Voronoi): Voronoi diagram object.
            target_indices (list or array): indices of the points of interest.

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
        return areas, perimeters
target_areas = [0.1352, 0.1069, 0.3314, 0.2104, 0.1979, 0.1146, 0.2658, 0.3365, 0.2016,
        0.1784, 0.1235, 0.2140, 0.2074, 0.1648, 0.1816, 0.1338, 0.2084, 0.2387,
        0.3114, 0.1379],
target_perimeters = [1.3685, 1.2170, 2.1428, 1.7073, 1.6558, 1.2603, 1.9192, 2.1595, 1.6714,
        1.5722, 1.3084, 1.7219, 1.6951, 1.5111, 1.5862, 1.3615, 1.6991, 1.8187,
        2.0771, 1.3823]
def analyze_voronoi_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # On prend seulement les 10 premiers points du step choisi
    points = df[['x','y']].to_numpy()

    vor = Voronoi(points)
    analyzer = VoronoiAnalyzer()
    target_indices = np.arange(len(points))  # ici: 0 à 9
    areas, perimeters = analyzer.voronoi_area_and_perimeter(vor, target_indices)

    print(f"Total area (sum of first 10 cells): {np.sum(perimeters[:n_cells])}")
    plt.figure(figsize=(6,4))
    plt.scatter((np.array(areas[:n_cells])-np.array(target_areas[:n_cells]))/np.array(target_areas[:n_cells]), (np.array(perimeters[:n_cells])-np.array(target_perimeters[:n_cells]))/np.array(target_perimeters[:n_cells]), color='skyblue', label='Prédictions')
    plt.scatter([0.0],[0.0],color='red')
    # Si les targets sont fournis, on les ajoute
    # if target_areas is not None and target_perimeters is not None:
    #     plt.scatter(target_areas[:n_cells], target_perimeters[:n_cells], 
    #                 color='orange', marker='x', label='Targets')
    plt.title(f'relative error')
    plt.xlabel('Area relative error')
    plt.ylabel('Perimeter relative error')
    plt.grid(True)
    plt.show()

    return areas, perimeters

# Exemple d’appel
analyze_voronoi_from_csv(
    "/home/jeanlienhard/Documents/Cell_GNN/GNN for energy/GNN_for _energy_target/positions_optimisees_periodicite_20.csv"
)
