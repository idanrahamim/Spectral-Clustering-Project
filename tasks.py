"""This module defines tasks for invoke"""
from invoke import task
import max_capacity_constants as max_cap

# Informative message
CAPACITY_MSG = "The following are maximum capacities for both 2 and 3 dimensional points accordingly " \
               "(format: (number of clusters, number of points)): " + str(max_cap.MAX_CAPACITY_2D) + ", " \
               + str(max_cap.MAX_CAPACITY_3D)


@task
def run(c, k="-1", n="-1", Random=True):
    """
    Compiles K-means++ module and runs the project with corresponding arguments k, n and --Random

    :param c command line
    :param k (int) number of clusters
    :param n (int) number of points to generate
    :param Random (bool) whether k should be random
    """
    rand = "--Random" if Random else "--no-Random"
    if Random:
        # If Random is True, k and n can be arbitrary
        n = k = 1
    if not Random and (k == "-1" or n == "-1"):
        print("Missing arguments k and/or n")
        return
    print(CAPACITY_MSG)
    # Setup k-means++ module
    c.run("python setup.py build_ext --inplace")
    # Start the project
    c.run("python main.py {} {} {}".format(k, n, rand))
