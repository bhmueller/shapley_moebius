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


# def test_bitwise_and():
#     k = 4
#     inputs_encoded = np.power(2, np.arange(k))
#     i = 3

#     g = (
#         np.bitwise_and(
#             i + 1,
#             inputs_encoded,
#         )
#         != 0
#     )

#     g_expected = np.array([True, True, False, False])

#     assert_array_equal(g, g_expected)


def test_h_matrix_independent():
    """The expected result is calculated with the original MATLAB implementation."""

    def model(x):
        return np.sum(x)

    def trafo(x):
        return x

    k = 3
    n = 10

    np.random.seed(123)
    u = cp.create_sobol_samples(n, 2 * k, 123).T

    x_a = trafo(u[:, 0:k])
    x_b = trafo(u[:, k:])

    n_subsets = np.power(2, k) - 1
    power_sequence = np.power(2, np.arange(k))

    h_matrix_actual, subset_size_actual = _calc_h_matrix_independent(
        n, model, x_a, x_b, n_subsets, power_sequence
    )

    # Get values in MATLAB: data = scipy.io.loadmat('wallah.mat')
    # save filename.mat file -v7; where file is the object, i.e. MATLAB array.

    h_matrix_expected = np.array(
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

    subset_size_expected = np.array([[1, 1, 2, 1, 2, 2, 3]])
    # zero_array = np.zeros((2, n_subsets))

    # Check h_matrix.
    assert_array_almost_equal(h_matrix_actual, h_matrix_expected, decimal=6)

    # Check subset_size.
    assert_array_equal(subset_size_actual, subset_size_expected)
    # with pytest.raises(AssertionError):
    #     assert_array_compare(operator.__ne__, h_matrix_expected, zero_array)


def test_mob_independent():
    """Use same setting as for test_h_matrix_independent."""

    k = 3

    # n = 10

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


# def test_find_sel():
#     """Check, whether 2nd for loop spits out the correct bits."""

#     expected = np.array(
#         [
#             [1, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0],
#             [1, 1, 1, 0, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0, 0],
#             [1, 0, 0, 1, 1, 0, 0],
#             [0, 1, 0, 1, 0, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1],
#         ]
#     )

#     actual = np.array([])

#     assert_array_equal(actual, expected)


# # def test_mob():
# #     print(1)


# # def test_sum_mob():
# #     print(1)


# def test_simplest_case():
#     """Just check that shapley_moebius runs through."""

#     def model(x):
#         return np.sum(x)

#     def trafo(x):
#         return x

#     k = 3
#     n = 10

#     shapley_effects, variance = shapley_moebius_independent(k, n, model, trafo)


# def test_():
#     """This test setting is taken from IP19."""

#     # k = 3
#     # n = 10
#     # model = ishigami_function
#     # trafo = rosenblatt_transformation


# def test_ishigami():
#     """This test setting is taken from PRB20."""
#     # Transform U(0, 1) to U(-pi, pi). Inputs independent.
#     def trafo(x):
#         x_trafo = (x - 0.5) * 2 * np.pi
#         return x_trafo

#     k = 3
#     n = 10 ** 4

#     shapley_effects, variance = shapley_moebius_independent(
#         k, n, ishigami_function, trafo
#     )

#     # Desired values are taken from PRB20.
#     desired = np.array([0.4358, 0.4424, 0.1218])

#     # Check relative tolerance only.
#     assert_allclose(shapley_effects, desired, rtol=1e-04)
