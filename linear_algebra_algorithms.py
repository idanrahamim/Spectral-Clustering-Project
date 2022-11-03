"""This module contains linear algebra functions"""
import math
import numpy as np


def gram_shmidt_qr(A):
    """
    Receives matrix A of real numbers and applies Gram-Shmidt QR decomposition algorithm

    :param A (numpy.ndarray) nxn matrix
    :return tuple (Q, R) matrices Q and R of QR decomposition
    """
    n = len(A)
    U = np.copy(A)
    Q = np.zeros((n, n), dtype=np.float32)
    R = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        col = U[:, i]
        rii = np.linalg.norm(col)
        R[i, i] = rii
        Q[:, i] = col / rii
        # Update the rest of the matrix
        if i != n-1:
            R[i, i + 1:] = np.matmul(Q[:, i], U[:, i + 1:])
            U[:, i + 1:] = U[:, i + 1:] - R[i, i + 1:] * Q[:, i:i + 1]
    return Q, R


def qr_iteration(A, epsilon):
    """
    QR-iteration algorithm. Takes symmetric, full rank matrix A and returns orthogonal matrix Qc of A's eigenvectors
    and upper triangle matrix Ac with corresponding eigenvalues on its diagonal.
    This function is a bottleneck of the whole project (takes 85% of runtime)

    :param A (numpy.ndarray) matrix
    :param epsilon (float) error
    :return orthogonal matrix Qc and upper triangle matrix Ac
    """
    n = len(A)
    # Change data type to speed up computations
    A = A.astype('float32')
    # Resulting matrices
    Ac = np.copy(A)
    Qc = np.eye(n, dtype=np.float32)
    for i in range(n):
        (Q, R) = gram_shmidt_qr(Ac)
        Ac = np.matmul(R, Q)
        Qnew = np.matmul(Qc, Q)
        # Exit if algorithm converges
        if np.amax(np.absolute(np.absolute(Qc) - np.absolute(Qnew))) <= epsilon:
            break
        Qc = Qnew
    # If one of the values on main diagonal is close to 0, then turn it to 0
    Ac_diag = Ac.diagonal()
    for i in range(len(Ac_diag)):
        if abs(Ac_diag[i]) < epsilon:
            Ac[i, i] = 0
    return Ac, Qc


def get_eigengap(eivals):
    """
    Receives an array of eigenvalues in ascending order and computes their eigengap

    :param eivals (numpy.ndarray) eigenvalues in ascending order
    :return eigengap of given eigenvalues
    """
    max_gap = eivals[1] - eivals[0]
    k = 1
    # Check n/2 eigenvalues
    for i in range(math.ceil(len(eivals) / 2)):
        if eivals[i+1] - eivals[i] > max_gap:
            max_gap = eivals[i+1] - eivals[i]
            k = i + 1
    return k


def extract_eigenvalues(D):
    """
    Extracts unique eigenvalues from the main diagonal of D

    :param D (numpy.ndarray) matrix
    :return np.ndarray of unique eigenvalues extracted from D in ascending order
    """
    return np.unique(D.diagonal())
