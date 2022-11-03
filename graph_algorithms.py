"""This module gives functionality needed to generate graphs represented as adjacency matrices and operate on them"""
from linear_algebra_algorithms import *


def calculate_weight(a, b):
    """
    This function calculates weight of edge matching two d-dimensional points

    :param a (numpy.ndarray) d-dimensional point
    :param b (numpy.ndarray) d-dimensional point
    :return calculated weight of an edge between a and b
    """
    return math.exp(-np.linalg.norm(a - b) / 2)


def generate_adjacency_matrix(points):
    """
    Generates adjacency matrix from d-dimensional array of points, s. t. each node represents point and
    between every 2 nodes there is an edge with positive weight defined in assignment

    :param points (numpy.ndarray) array of d-dimensional points
    :return nxn adjacency matrix
    """
    n = len(points)
    # Init matrix nxn with zeros
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = calculate_weight(points[i], points[j])
            adj_matrix[i, j] = w
            adj_matrix[j, i] = w

    return adj_matrix


def compute_diag_degree_matrix(graph):
    """
    Computes diagonal degree matrix by a given graph in adjacency matrix format

    :param graph (numpy.ndarray) adjacency matrix representing a graph generated from array of points
    :return diagonal matrix defined in the assignment
    """
    n = len(graph)
    diag = np.zeros((n, n))
    for i in range(n):
        diag[i, i] = np.sum(graph[i, :])
    return diag


def generate_norm_laplacian_graph(graph):
    """
    Generates normalized graph laplacian by a given graph in adjacency matrix format

    :param graph (numpy.ndarray) adjacency matrix representing a graph generated from array of points
    :return normalized graph laplacian matrix
    """
    D = compute_diag_degree_matrix(graph)
    D[np.diag_indices_from(compute_diag_degree_matrix(graph))] **= -0.5
    return np.eye(len(graph)) - np.matmul(D, np.matmul(graph, D))


def eigen_values_of_points(points, eps):
    """
    Takes an array of d-dimensional points, generates weighted graph and normalized graph laplacian. Computes
    eigenvectors matrix and corresponding eigenvalues matrix of the normalized graph
    laplacian matrix using QR-iteration algorithm

    :param points ()
    :param eps (float) error (tolerance)
    :return eigenvectors matrix and corresponding eigenvalues matrix (as they are defined in the assignment) of
    the normalized graph laplacian matrix of graph generated from points
    """
    W = generate_adjacency_matrix(points)
    # Normalized graph laplacian
    Lnorm = generate_norm_laplacian_graph(W)
    (A, Q) = qr_iteration(Lnorm, eps)
    return A, Q
