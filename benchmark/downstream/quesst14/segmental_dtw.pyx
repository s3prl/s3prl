#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
import cython


cdef extern from "math.h":
    double log(double theta)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double segmental_dtw(double[:,:] query, double[:,:] search):
    '''
    Segmental Locally Normalized DTW
    :param query: ndarray (n_samples, n_features)
       keyword template
    :param search: ndarray (n_samples, n_features)
       indexed utterance
    :return:
       minimum cost
    '''
    cdef int N = query.shape[0]
    cdef int M = search.shape[0]
    cdef double[:,:] dist_mat = distance_mat(query, search)
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


cdef double[:,:] distance_mat(double[:, :] s, double[:, :] t):
    cdef int N = s.shape[0]
    cdef int M = t.shape[0]

    cdef double[:, :] dist_mat = np.zeros((N, M))
    cdef int i, j
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = distance(s[i], t[j])
    return dist_mat


cdef inline double distance(double[:] s, double[:] t):
    cdef double dot_result = 0.
    cdef int i
    for i in range(s.shape[0]):
        dot_result += s[i]*t[i]
    return -log(dot_result)
