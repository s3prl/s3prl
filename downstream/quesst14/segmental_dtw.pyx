#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
import cython

from libc.math cimport fabs, fmin, fmax, pow, sqrt, log
from libc.float cimport DBL_MAX


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
    cdef double dist
    cdef double[3] costs
    cdef int[3] lengths
    cdef double norm_cost, min_norm_cost
    cdef int i, j, k

    for i in range(1, N):
        for j in range(1, M):
            dist = dist_mat[i, j]
            costs[0] = acc[i - 1, j - 1] + dist
            costs[1] = acc[i - 1, j] + dist
            costs[2] = acc[i, j - 1] + dist
            lengths[0] = l[i - 1, j - 1] + 1
            lengths[1] = l[i - 1, j] + 1
            lengths[2] = l[i, j - 1] + 1

            min_norm_cost = costs[0] / lengths[0]
            k = 0
            norm_cost = costs[1] / lengths[1]
            if norm_cost < min_norm_cost:
                min_norm_cost = norm_cost
                k = 1
            norm_cost = costs[2] / lengths[2]
            if norm_cost < min_norm_cost:
                min_norm_cost = norm_cost
                k = 2

            acc[i, j] = costs[k]
            l[i, j] = lengths[k]

    min_norm_cost = 1.0e8

    for j in range(M):
        norm_cost = acc[N - 1, j] / l[N - 1, j]
        if norm_cost < min_norm_cost:
            min_norm_cost = norm_cost

    return min_norm_cost


cdef double[:,:] initialize_acc(double[:, :] dist_mat):
    cdef int N = dist_mat.shape[0]
    cdef int M = dist_mat.shape[1]
    cdef double[:, :] acc = np.zeros((N, M))
    cdef double prefix_sum = 0.0
    cdef int i

    for i in range(N):
        prefix_sum += dist_mat[i, 0]
        acc[i, 0] = prefix_sum
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
            dist_mat[i, j] = l2_norm(query[i], doc[j])

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
        dist += fabs(a[i] - b[i])

    return dist


cdef inline double l2_norm(double[:] a, double[:] b):
    cdef double dist = 0.
    cdef int i

    for i in range(a.shape[0]):
        dist += pow(a[i] - b[i], 2.0)

    return sqrt(dist)


cdef inline double negative_log_dot(double[:] a, double[:] b):
    cdef double dist = 0.
    cdef int i

    for i in range(a.shape[0]):
        dist += (a[i] * b[i])

    return -log(dist)
