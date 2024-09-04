import numpy as np

def sample_cirlce(r=1, size=100):
    points = []
    normals = []
    for phi in np.linspace(0, 2*np.pi, size):
        pt = np.array([np.cos(phi), np.sin(phi)]) * r
        points.append(pt)
        normals.append(pt / np.linalg.norm(pt))
    return np.array(points), np.array(normals)

def sample_curved_cirlce(r=1, size=100):
    points = []
    normals = []
    for phi in np.linspace(0, 2*np.pi, size):
        R = r + np.sin(phi * 10) * 0.1
        pt = np.array([np.cos(phi), np.sin(phi)]) * R
        points.append(pt)
    
    num_points = len(points)
    for i in range(num_points):
        # Use forward difference to calculate normal at point i
        i_next = (i + 1) % num_points
        i_prev = (i - 1) % num_points
        normal = np.array([points[i_next][1] - points[i_prev][1], points[i_prev][0] - points[i_next][0]])
        normals.append(normal/np.linalg.norm(normal))
    return np.array(points), np.array(normals)