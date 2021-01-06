#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
import cython


cdef extern from "math.h":
    double abs(double delta)
    double fmin(double a, double b)
    double fmax(double a, double b)
    double pow (double base, double exponent)
    double sqrt (double x)


cdef extern from "float.h":
    double DBL_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double segmental_dtw(double[:, :] query, double[:, :] doc):
    '''
    Segmental Locally Normalized DTW
    :param query: ndarray (n_samples, n_features)
       keyword template
    :param doc: ndarray (n_samples, n_features)
       indexed utterance
    :return:
       minimum cost
    '''
    cdef int N = query.shape[0]
    cdef int M = doc.shape[0]
    cdef double[:,:] dist_mat = distance_mat(query, doc)
    cdef double[:,:] acc = initialize_acc(dist_mat)
    cdef int[:,:] l = initialize_l(dist_mat)
    cdef double[:] costs = np.zeros(3)
    cdef int i_penalty
    cdef double dist
    cdef int i, j

    for i in range(1, N):
        for j in range(1, M):
            dist = dist_mat[i, j]
            costs[0] = (acc[i - 1, j - 1] + dist) / (l[i - 1, j - 1] + 1)
            costs[1] = (acc[i - 1, j] + dist) / (l[i - 1, j] + 1)
            costs[2] = (acc[i, j - 1] + dist) / (l[i, j - 1] + 1)
            i_penalty = min_around(costs)
            if i_penalty == 0:
                acc[i, j] = acc[i - 1, j - 1] + dist
                l[i, j] = l[i - 1, j - 1] + 1
            elif i_penalty == 1:
                acc[i, j] = acc[i - 1, j] + dist
                l[i, j] = l[i - 1, j] + 1
            else:
                acc[i, j] = acc[i, j - 1] + dist
                l[i, j] = l[i, j - 1] + 1

    cdef double min_cost = 1.0e8
    cdef double cost

    for j in range(M):
        cost = acc[N - 1, j] / l[N - 1, j]
        if cost < min_cost:
            min_cost = cost

    return min_cost


cdef inline int min_around(double[:] v):
    cdef int m = 0
    cdef int i

    for i in range(1, v.shape[0]):
        if v[i] < v[m]:
            m = i

    return m


cdef double[:,:] initialize_acc(double[:, :] dist_mat):
    cdef int N = dist_mat.shape[0]
    cdef int M = dist_mat.shape[1]
    cdef double[:, :] acc = np.zeros((N, M))
    cdef int i

    for i in range(N):
        acc[i, 0] = np.sum(dist_mat[:i + 1, 0])
    acc[0] = dist_mat[0]

    return acc


cdef int[:, :] initialize_l(double[:, :] dist_mat):
    cdef int N = dist_mat.shape[0]
    cdef int M = dist_mat.shape[1]
    cdef int[:, :] l = np.zeros((N, M), dtype=np.int32)
    cdef int i

    for i in range(N):
        l[i, 0] = i + 1
    l[0, :] = 1

    return l


cdef double[:,:] distance_mat(double[:, :] query, double[:, :] doc):
    cdef int N = query.shape[0]
    cdef int M = doc.shape[0]

    # Calculate distance matrix
    cdef double[:, :] dist_mat = np.zeros((N, M))
    cdef int i, j
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = l1_norm(query[i], doc[j])

    # Min-max normalize
    cdef double dist_min = DBL_MAX
    cdef double dist_max = -DBL_MAX
    for i in range(N):
        dist_min = DBL_MAX
        dist_max = -DBL_MAX
        for j in range(M):
            dist_min = fmin(dist_min, dist_mat[i, j])
            dist_max = fmax(dist_max, dist_mat[i, j])
        for j in range(M):
            dist_mat[i, j] = (dist_mat[i, j] - dist_min) / (dist_max - dist_min)

    return dist_mat


cdef inline double l1_norm(double[:] a, double[:] b):
    cdef double dist = 0.
    cdef int i

    for i in range(a.shape[0]):
        dist += abs(a[i] - b[i])

    return dist


cdef inline double l2_norm(double[:] a, double[:] b):
    cdef double dist = 0.
    cdef int i

    for i in range(a.shape[0]):
        dist += pow(a[i] - b[i], 2.0)

    return sqrt(dist)
