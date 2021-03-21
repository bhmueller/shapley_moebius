"""
This file contains tests for shapley_moebius_dependent.py.
"""
import numpy as np
import pandas as pd
import chaospy as cp
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from shapley_moebius_dependent import shapley_moebius_dependent
from shapley_moebius_dependent import _calc_h_matrix_dependent
from auxiliary_functions import get_test_values_additive_uniform


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
    """Check that function works for three uniformly distributed inputs."""

    k = 3
    n = 100

    seed = 123

    def model(x):
        return np.sum(x, axis=1)

    def trafo(x):
        return x

    u = cp.create_sobol_samples(n, 2 * k, seed).T

    # random_mode = "Sobol"

    # correlation = "dependent"

    n_subsets = np.power(2, k) - 1

    rank_corr = np.array([[1, -0.5, 0.5], [-0.5, 1, 0], [0.5, 0, 1]])

    # expected_results = get_test_values_additive_uniform(
    #     k, n, seed, correlation, rank_corr, random_mode
    # )

    h_matrix_actual, subset_size_actual = _calc_h_matrix_dependent(
        k, n, u, model, trafo, n_subsets, rank_corr
    )

    # assert_array_almost_equal(h_matrix_actual, expected_results['h_matrix'][0])

    # assert_array_equal(subset_size_actual, expected_results['subset_size'][0])
