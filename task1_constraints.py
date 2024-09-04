import numpy as np
from scipy.spatial import KDTree

def sample_constraints(vertices, normals, eps):
    """Sample points near vertices along normals and -normals directions.
        It should work for 2D points.
        It might be implemented in the way that it works both for 2D or 3D points (without using if...)

    Args:
        vertices (np.array, [N, 2 or 3]): 2D/3D coordinates of N points in space
        normals (np.array, [N, 2 or 3]): Normal direction for each vertex (may need to be normalized)
        eps (float): how near should sample points

    Returns:
        new_vert (np.array, [2*N, 2 or 3]): New sampled points
        new_values (np.array, [2*N, 1]): Distance value for each of the sampled point

        vertices SIZE = [N,2] = 40 = [20,2] = 40
        new_vert = [2*N, 2] = 20*2,2 = 80
        new_values = [2*N, 1] = [20*2,1] = 40
    """
    # TODO: Task 1
    # For each vertex:
    #  – Sample a new vertex along eps * normal direction
    #  – Check if the new vertex is the closest one to the given vertex
    #  – If not, set eps = eps/2 and start again
    # Repeat the same steps but for -eps
    # Important: there should be NO cycle (for/while) over number of vertices.

    new_vert = []
    new_values = []

    kdtree = KDTree(vertices)

    for i, (vertex, normal) in enumerate(zip(vertices, normals)):
        new_vertex_plus, new_vertex_minus = find_constraint(kdtree,i, vertex, normal, eps)

        new_vert.append(new_vertex_plus)
        new_vert.append(new_vertex_minus)
        new_values.append(np.linalg.norm(new_vertex_plus - vertex))
        new_values.append(-np.linalg.norm(new_vertex_minus - vertex))

    new_vert = np.array(new_vert)
    new_values = np.array(new_values)

    return new_vert, new_values

def find_constraint(kdtree,i, vertex, normal, eps):
        new_vertex_plus = vertex + eps * normal
        new_vertex_minus = vertex - eps * normal

        _, closest_index_plus = kdtree.query(new_vertex_plus)
        _, closest_index_minus = kdtree.query(new_vertex_minus)
        if closest_index_plus == i and closest_index_minus == i:
            return new_vertex_plus, new_vertex_minus
        
        else:
             return find_constraint(kdtree,i,vertex,normal,eps/2)

    