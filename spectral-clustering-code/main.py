"""This is the main module of the project that imports all the other necessary modules"""
import graph_algorithms as graph
import linear_algebra_algorithms as alg
from spectral_clustering import spectral_clustering
from kmeans_pp import kmeans_pp
import numpy as np
import input_parser
from sklearn.datasets import make_blobs
import math
from text_output import *
from visual_output import *
import max_capacity_constants as max_cap

# Constants
# Maximum number of iterations in k-means algorithm
MAX_ITER = 300
# Epsilon
EPS = 0.0001


def choose_params(rand, k, n):
    """
    Chooses number of clusters K, number of points N and dimension d based on its command line parameters

    :param rand (bool) whether parameters must be chosen randomly
    :param k (int) umber of clusters from command line
    :param n (int) number of points
    :returns tuple of (K, N, d) - number of clusters chosen, number of points to generate, dimension of each point
    """
    # Choose randomly dimension of points to generate between 2 and 3
    d = np.random.randint(2, 4)
    if rand:
        # Choose random parameters depending on dimension
        if d == 2:
            K = np.random.randint(math.floor(max_cap.MAX_CAPACITY_2D[0] / 2), max_cap.MAX_CAPACITY_2D[0] + 1)
            N = np.random.randint(math.floor(max_cap.MAX_CAPACITY_2D[1] / 2), max_cap.MAX_CAPACITY_2D[1] + 1)
        else:
            K = np.random.randint(math.floor(max_cap.MAX_CAPACITY_3D[0] / 2), max_cap.MAX_CAPACITY_3D[0] + 1)
            N = np.random.randint(math.floor(max_cap.MAX_CAPACITY_3D[1] / 2), max_cap.MAX_CAPACITY_3D[1] + 1)
    else:
        K = k
        N = n
    return K, N, d


def generate_clusters(K, n, d, points, is_random):
    """
    Generates two arrays of clusters using both spectral clustering method and k-means. Decides on their amount k
    and returns all the generated data

    :param K (int) primary number of clusters
    :param n (int) number of points
    :param d (int) dimension of each point
    :param points (numpy.ndarray) array of points
    :param is_random random flag. Whether k should be chosen be eigengap heuristic
    :return a tuple of spectral clustering method result array, kmeans method and number of clusters k
    """
    # Retrieving eigenvalues and eigenvectors of normal laplacian matrix created by points
    (A, Q) = graph.eigen_values_of_points(points, EPS)
    k = alg.get_eigengap(alg.extract_eigenvalues(A)) if is_random else K
    # Find clusters using two algorithms
    spec_clusters = spectral_clustering(A, Q, MAX_ITER, k)
    kmeans_clusters = kmeans_pp(k, n, d, MAX_ITER, points)
    return spec_clusters, kmeans_clusters, k


def main():
    """
    This is the main function that starts the whole project

    """
    # Parse arguments
    args = input_parser.parse_arguments()
    K, n, d = choose_params(args.random, args.k, args.n)
    # Generate points
    points_arr, y = make_blobs(n_samples=n, centers=K, n_features=d)
    spec_clusters, kmeans_clusters, k = generate_clusters(K, n, d, points_arr, args.random)
    # Generate output
    output_text_cluster_data(points_arr, spec_clusters, kmeans_clusters, y, k)
    draw_plot(points_arr, n, k, spec_clusters, kmeans_clusters, y, d, K)


# Start program
main()
