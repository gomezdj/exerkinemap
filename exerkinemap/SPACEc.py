# Cell-Cell Interaction Analysis 
import numpy as np
from scipy.spatial import Delaunay
import scipy

def delaunay_edges(coords):
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = (simplex[i], simplex[j])
            edges.add(tuple(sorted(edge)))
    return list(edges)

def compute_distances(coords, edges):
    distances = []
    for i, j in edges:
        distance = np.linalg.norm(coords[i] - coords[j])
        distances.append(distance)
    return distances

# Loading data
coords = ...  # Replace with actual coordinate data

# Computing edges and distances
edges = delaunay_edges(coords)
distances = compute_distances(coords, edges)

# Distance Calculation
import numpy as np

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
