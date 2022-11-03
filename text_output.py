"""This module outputs generated data into text files"""
# Constants
DATA_PATH = "data.txt"
CLUSTERS_PATH = "clusters.txt"


def output_data_file(points, cluster_membership):
    """
    Prints generated points and its clusters into a file specified in DATA_PATH constant

    :param points (numpy.ndarray) array of points
    :param  cluster_membership (numpy.ndarray) numpy array of original clusters matching these
    """
    with open(DATA_PATH, "w") as data_file:
        for i, p in enumerate(points):
            line = ""
            for coord in p:
                # Print each coordinate with precision of 8 digits after floating point
                line += "%.8f" % coord + ","
            line += str(cluster_membership[i])
            data_file.write(line + "\n")


def output_clusters_file(clusters_lst, k):
    """
    Prints cluster data from both spectral and k-means methods into file

    :param clusters_lst (list) list of numpy cluster arrays to print into file specified in constant CLUSTERS_PATH
    :param k (int) size of each cluster
    """
    with open(CLUSTERS_PATH, "w") as clusters_file:
        clusters_file.write(str(k) + "\n")
        # Iterate over all clusters
        for clusters in clusters_lst:
            # Generate lines array for writing into file
            lines = [[] for j in range(k)]
            for point, cluster in enumerate(clusters):
                lines[cluster].append(str(point))
            lines = [",".join(ln) + "\n" for ln in lines]
            clusters_file.writelines(lines)


def output_text_cluster_data(points, spec_clusters, kmeans_clusters, cluster_membership, k):
    """
    Prints generated data into text files.
    Receives array of points to print, spec_clusters and kmeans_clusters are cluster arrays
    cluster_membership - original clusters (centers), k - size of a single cluster

    :param points (numpy.ndarray) array of generated points
    :param spec_clusters (numpy.ndarray) array of clusters for points generated be spectral clustering
    :param kmeans_clusters (numpy.ndarray) array of clusters for points generated be k-means
    :param cluster_membership (numpy.ndarray) array of initially generated
    :param k (numpy.ndarray) number of clusters
    """
    # Output two files
    output_data_file(points, cluster_membership)
    output_clusters_file([spec_clusters, kmeans_clusters], k)
