"""
This file contains tests for shapley_moebius_dependent.py.
"""
import numpy as np
import pandas as pd
import chaospy as cp
from shapley_moebius_dependent import shapley_moebius_dependent
from shapley_moebius_dependent import _calc_h_matrix_dependent


def test_shapley_moebius_dependent():
    """Check whether function runs through."""

    k = 3
    n = 100

    def model(x):
        return np.sum(x, axis=1)

    def trafo(x):
        return x

    rank_corr = np.array([[1, -0.5, 0.5], [-0.5, 1, 0], [0.5, 0, 1]])

    random_mode = "Sobol"

    shapley_moebius_dependent(k, n, model, trafo, rank_corr, random_mode)


def test_calc_h_matrix_dependent():

    k = 3
    n = 100

    def model(x):
        return np.sum(x, axis=1)

    def trafo(x):
        return x

    u = cp.create_sobol_samples(n, 2 * k, 123).T

    n_subsets = np.power(2, k) - 1

    rank_corr = np.array([[1, -0.5, 0.5], [-0.5, 1, 0], [0.5, 0, 1]])

    h_matrix, subset_size = _calc_h_matrix_dependent(
        k, n, u, model, trafo, n_subsets, rank_corr
    )
