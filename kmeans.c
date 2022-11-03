/*
 * This is the source C-API code for python module called mykmeanssp. This module has only one function for outside use
 * called computeCentroids for applying k-means algorithm on some points
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* computeCentroids(PyObject *, PyObject *);
static double **getFinalCentroids(int, int, int, int , double **, double **);


/*
 * Calculates squared euclidean distance between observation at index obs_ind and centroid at index cent_ind
 *
 * double** observations array of observations (points)
 * double** centroids array of centroids
 * int d dimension of each point in observations and centroids arrays
 * int obs_ind index of observation in observations array
 * int cent_ind index of centroid in centroids array
 */
static double distance(double **observations, double **centroids, int d, int obs_ind, int cent_ind){
    double sum = 0;
    // Counter
    int i;
    for (i = 0; i < d; i++) {
        sum += ((observations[obs_ind][i] - centroids[cent_ind][i]) * (observations[obs_ind][i] - centroids[cent_ind][i]));
    }
    return sum;
}


/*
 * Returns index of centroid in centroids array of centroid with the minimal euclidean distance from observation
 * observations[obs_ind]
 *
 * double** observations array of observations
 * double** centroids array of centroids
 * int d dimension of ech point in observations array and centroids array
 * int k number of centroids in centroids array
 * int obs_ind observation index to check
 */
static int getMinCentroid(double **observations, double **centroids, int d, int k, int obs_ind){
    double min_dist;
    int min_centroid_ind;
    int i;
    double dist;
    // Init first minimum distance
    min_dist = distance(observations, centroids, d, obs_ind, 0);
    min_centroid_ind = 0;
    // Check every other centroid fro minimal distance
    for (i = 1; i < k; i++) {
        dist = distance(observations, centroids, d, obs_ind, i);
        if (dist < min_dist){
            min_centroid_ind = i;
            min_dist = dist;
        }
    }
    return min_centroid_ind;
}


/*
 * Computes new clusters based on previous computation. Updates centroids array
 *
 * double **observations n observations (points)
 * double **centroids centroids from last iteration
 * int *centroid_indices indices of previously computed centroids
 * int *num_in_clus number in cluster
 * int d dimension of each point
 * int n number of observations (points)
 */
static void updateClusters(double **observations, double **centroids, int *centroid_indices, int *num_in_clus,
                                                                                                    int d, int n){
    // Counters
    int i, j;
    int centroid_index;
    for (i = 0; i < n; i++) {
        centroid_index = centroid_indices[i];
        for (j = 0; j < d; j++) {
            centroids[centroid_index][j] += observations[i][j] / (double) num_in_clus[centroid_index];
        }
    }
}


/*
 * Compares 2 arrays of centroids. Returns 1 if all coordinates in 2 arrays have the same values, 0 - otherwise
 *
 * double** centroids array of k d-dimensional centroids
 * double** centroids_cpy the other array of k d-dimensional centroids
 * int k number of centroids in each array
 * int d dimension
 */
static int compareArrays(double** centroids, double** centroids_cpy, int k, int d){
    // Counters
    int i, j;
    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            if (centroids[i][j] != centroids_cpy[i][j]){
                return 0;
            }
        }
    }
    return 1;
}


/*
 * Computes k centroids from n d-dimensional points (observations) using k-means algorithm
 *
 * int k number of centroids to generate
 * int n number of observations
 * int d dimension of each observation
 * int max_iter max number of iterations to perform in k-means algorithm
 * double **observations array of n d-dimensional observations
 * double **init_centroids array of k d-dimensional initial centroids
 */
static double **getFinalCentroids(int k, int n, int d, int max_iter, double **observations, double **init_centroids) {
    int min_clus;
    // Reusable counters
	int i, j;
	int iter = 0;
	// Arrays of points and clusters
    double **clusters;
    int *clus_ind_of_vec;
    int *num_in_clus;
    double **clusters_cpy;

    clusters = init_centroids;
    clus_ind_of_vec = (int *) malloc(n * sizeof(int));
    // clus_ind_of_vec[i] == cluster index of the i'th input vector
    if(clus_ind_of_vec == NULL) return NULL;
    num_in_clus = (int *) malloc(k * sizeof(int));
    // num_in_clus[i] == number of elements in cluster i
	if(num_in_clus == NULL) return NULL;
	
    clusters_cpy = (double **) malloc(k * sizeof(double *));
    // Throw exception if allocation has failed
    if(clusters_cpy == NULL) return NULL;
    for (i = 0; i < k; i++) {
        clusters_cpy[i] = (double*) malloc(d * sizeof(double));
		if(clusters_cpy[i] == NULL) return NULL;
    }
    // Iterate max_iter times
    while (iter < max_iter) {
        for (i = 0; i < k; i++) {
            for (j = 0; j < d; j++) {
                clusters_cpy[i][j] = clusters[i][j];
            }
        }
        for(i = 0; i < k; i++){
            num_in_clus[i] = 0;
        }
        // Update cluster information
        for (i = 0; i < n; i++) {
            min_clus = getMinCentroid(observations, clusters, d, k, i);
            // New cluster of each observation
            clus_ind_of_vec[i] = min_clus;
            // New count for each cluster
            num_in_clus[min_clus]++;
        }
        // Zeroing all the clusters so we can sum the new clusters
        for (i = 0; i < k; i++) {
            for (j = 0; j < d; j++) {
                clusters[i][j] = 0;
            }
        }
        updateClusters(observations, clusters, clus_ind_of_vec, num_in_clus, d, n);
        if(compareArrays(clusters, clusters_cpy, k, d))
            break;
        else
            iter++;
    }

    // Free all allocated memory
    free(clus_ind_of_vec);
    free(num_in_clus);
    for(i = 0; i < k; i++){
        free(clusters_cpy[i]);
    }
    free(clusters_cpy);
    return clusters;
}


/*
 * Python function, that must be imported from python module. This function applies k-means algorithm to a set of
 * points (observations) and returns calculated clusters for each point
 *
 * PyObject *self reference to the function itself (treated as python object)
 * PyObject *args a set of arguments is a set of 4 positional arguments: k (int) - number of clusters,
 * n (int) - number of points, d (int) - dimension of each point, max_iter (int) - max number of iterations,
 * init_centroids (list) = initial centroids  python list, observations (list) - a list of points (observations)
 */
static PyObject* compute_centroids(PyObject *self, PyObject *args){
	PyObject *py_cent_list;
	// Unpacked arguments
	int k;
	int n;
	int d;
	int max_iter;
	double **cent_list;
	// Counters
	int i, j;
	// Python objects
	PyObject *centroid;
	PyObject *observ_list;
	PyObject *init_cent_list;
	double **observ_list_c;
	double **init_cent_list_c;
	// Retrieve arguments from the input python object
	PyArg_ParseTuple(args, "iiiiO!O!", &k, &n, &d, &max_iter, &PyList_Type, &observ_list, &PyList_Type, &init_cent_list);
	// Form C-arrays
	observ_list_c=(double **) malloc(n * sizeof(double *));
	// Throw exception if memory allocation has failed
	if(observ_list_c == NULL) return NULL;
	// Turn python objects into C types
	for(i = 0; i < n; i++) {
		double *vector;
		vector= (double *) malloc(d * sizeof(double));
		if(vector == NULL) return NULL;
		observ_list_c[i] = vector;
		for(j = 0; j < d; j++){
			PyObject *list;
			list=PyList_GetItem(observ_list, i);
			vector[j]=PyFloat_AsDouble(PyList_GetItem(list, j));
		}
	}
	init_cent_list_c = (double **) malloc(k * sizeof(double *));
	// If memory allocation has failed
	if(init_cent_list_c == NULL) return NULL;
	for(i = 0; i < k; i++) {
		double *vector;
		vector = (double *) malloc(d * sizeof(double));
		if(vector == NULL) return NULL;
		init_cent_list_c[i] = vector;
		for(j = 0; j < d; j++){
			PyObject *list;
			list=PyList_GetItem(init_cent_list, i);
			vector[j]=PyFloat_AsDouble(PyList_GetItem(list, j));
		}
	}
	// Calculate centroids using k-means algorithm
	cent_list=getFinalCentroids(k, n, d, max_iter, observ_list_c, init_cent_list_c);
	// Throw exception is the list is NULL (i.e. getFinalCentroids threw an exception)
	if(cent_list == NULL) return NULL;
	
	// Turn into python list
	py_cent_list = PyList_New(k);
	for(i = 0; i < k; i++){
		centroid=PyList_New(d);
		for(j=0; j<d; j++){
			PyObject* python_int = Py_BuildValue("d", cent_list[i][j]);
			PyList_SetItem(centroid, j, python_int);
		}
		PyList_SetItem(py_cent_list, i, centroid);
	}
	// Free all allocated memory
	for(i = 0;i < n; i++){
		free(observ_list_c[i]);
	}
	for(i = 0;i < k; i++){
		free(init_cent_list_c[i]);
	}
	free(observ_list_c);
	free(init_cent_list_c);
	return py_cent_list;
}


/*
 * Module and method definitions for C-API compiler
 */
static PyMethodDef _methods[] = {
    {"compute_centroids", compute_centroids, METH_VARARGS, "Computes centroids using k-means algorithm"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "This module implements k-means algorithm",
    -1,
    _methods
};


PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&_moduledef);
}
