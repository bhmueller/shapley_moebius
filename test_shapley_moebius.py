"""
This file contains tests for shapley_moebius.py.

Iooss, Betrand and Clémentine Prieur. 2019.
    Shapley effects for sensitivity analysis with correlated inputs: comparisons with
    Sobol’ indices, numerical estimation and applications. hal-01556303v6
"""
import numpy as np
import chaospy as cp
import operator
import pytest
from numpy.testing import assert_array_compare
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from shapley_moebius import shapley_moebius_independent
from shapley_moebius import _calc_h_matrix_independent
from shapley_moebius import _calc_mob_independent
from auxiliary_functions import ishigami_function
from auxiliary_functions import get_test_values_additive_uniform


def test_h_matrix_independent():
    """The expected result is calculated with the original MATLAB implementation."""

    def model(x):
        return np.sum(x, axis=1)

    def trafo(x):
        return x

    k = 3
    n = 100
    seed = 123

    # np.random.seed(123)
    u = cp.create_sobol_samples(n, 2 * k, seed).T

    # u = np.linspace(0, 14.5, 30)

    x_a = trafo(u[:, 0:k])
    x_b = trafo(u[:, k:])

    n_subsets = np.power(2, k) - 1
    power_sequence = np.power(2, np.arange(k))

    h_matrix_actual, subset_size_actual = _calc_h_matrix_independent(
        n, model, x_a, x_b, n_subsets, power_sequence
    )

    # h_matrix_expected = np.array(
    #     [
    #         [1.16043091, 0.11355591, 2.0, 2.04714966, 6.29016113, 3.125, 8.09402466],
    #         [
    #             2.53569031,
    #             0.79321594,
    #             3.32890625,
    #             3.36791687,
    #             5.90360718,
    #             4.16113281,
    #             6.69682312,
    #         ],
    #     ]
    # )

    # subset_size_expected = np.array([[1, 1, 2, 1, 2, 2, 3]])

    expected_results = get_test_values_additive_uniform(k, n, seed)

    # Check h_matrix. First estimator only.
    assert_array_almost_equal(h_matrix_actual, expected_results["h_matrix"], decimal=6)

    # Check subset_size.
    assert_array_equal(subset_size_actual, expected_results["subset_size"])


def test_mob_independent():
    """Use same setting as for test_h_matrix_independent."""

    k = 3

    # n = 100

    n_subsets = np.power(2, k) - 1

    h_matrix = np.array(
        [
            [1.16043091, 0.11355591, 2.0, 2.04714966, 6.29016113, 3.125, 8.09402466],
            [
                2.53569031,
                0.79321594,
                3.32890625,
                3.36791687,
                5.90360718,
                4.16113281,
                6.69682312,
            ],
        ]
    )

    subset_size = np.array([[1, 1, 2, 1, 2, 2, 3]])

    mob_actual = _calc_mob_independent(n_subsets, h_matrix, subset_size)

    # Get expected mob from Octave. Load mob_results.mat.
    mob_expected = np.array(
        [
            [
                1.16043091,
                0.11355591,
                0.36300659,
                2.04714966,
                1.54129028,
                0.48214722,
                0.0,
            ],
            [2.53569031, 0.79321594, 0.0, 3.36791687, 0.0, 0.0, 0.0],
        ]
    )

    assert_array_almost_equal(mob_actual, mob_expected)


def test_simplest_case():
    """Test entire function for simple case, where data is uniformly distributed on
    [0, 1). Expected values derived by MATLAB implementation.
    """

    def model(x):
        return np.sum(x, axis=1)

    def trafo(x):
        return x

    k = 3
    n = 10

    np.random.seed(123)

    shapley_effects_actual, variance_actual = shapley_moebius_independent(
        k, n, model, trafo
    )

    shapley_effects_normalised = shapley_effects_actual / variance_actual

    # Check whether normalised Shapley effects sum up to one.

    assert_array_equal(np.sum(shapley_effects_normalised, axis=1), np.ones(2))

    # shapley_effects_expected = np.array(
    #     [[3.06472778, 0.95870972, 4.07058716], [2.53569031, 0.79321594, 3.36791687]]
    # )

    # variance_expected = np.array([[8.09402466], [6.69682312]])

    # assert_array_almost_equal(shapley_effects_actual, shapley_effects_expected)

    # assert_array_almost_equal(variance_actual, variance_expected)


# def test_():
#     """This test setting is taken from IP19."""

#     # k = 3
#     # n = 10
#     # model = ishigami_function
#     # trafo = rosenblatt_transformation


def test_ishigami():
    """This test setting is taken from PRB20. Independent inputs. Model: Ishigami
    function.
    """
    # Transform U[0, 1) to U[-pi, pi). Inputs independent.
    def trafo(x):
        x_trafo = (x - 0.5) * 2 * np.pi
        return x_trafo

    k = 3
    n = 10 ** 6

    np.random.seed(123)

    shapley_effects, variance = shapley_moebius_independent(
        k, n, ishigami_function, trafo
    )

    actual = shapley_effects / variance

    # Desired values are taken from PRB20.
    desired = np.array([[0.4358, 0.4424, 0.1218], [0.4358, 0.4424, 0.1218]])

    # Check relative tolerance only.
    assert_allclose(actual, desired, rtol=1e-03)
