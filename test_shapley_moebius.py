"""
This file contains tests for shapley_moebius.py.
"""
import numpy as np
import chaospy as cp
import operator
from numpy.testing import assert_array_compare
from shapley_moebius import shapley_moebius
from shapley_moebius import _calc_h_matrix
from auxiliary_functions import ishigami_function


def model(x):
    return np.sum(x)


def trafo(x):
    return x


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


def test_h_matrix_non_zero():
    """
    This test should pass if all elements in h_matrix are unequal zero.
    """

    def model(x):
        return np.sum(x)

    k = 3
    n = 10

    u = cp.create_sobol_samples(n, 2 * k, 123).T

    x_a = trafo(u[:, 0:k])
    x_b = trafo(u[:, k:])

    n_subsets = np.power(2, k) - 1
    subset_encoder = np.power(2, np.arange(k))

    h_matrix_expected = _calc_h_matrix(n, model, x_a, x_b, n_subsets, subset_encoder)

    zero_array = np.zeros((2, n_subsets))

    assert_array_compare(operator.__ne__, h_matrix_expected, zero_array)


def test_mob():
    print(1)


def test_sum_mob():
    print(1)


def test_ishigami():
    """
    This test setting is taken from Iooss, Betrand and Clémentine Prieur. 2019. Shapley
    effects for sensitivity analysis with correlated inputs: comparisons with Sobol’
    indices, numerical estimation and applications. hal-01556303v6"""

    # k = 3
    # n = 10
    # model = ishigami_function
    # trafo = rosenblatt_transformation
