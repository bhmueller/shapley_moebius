"""
This file contains tests for shapley_moebius.py.
"""
import numpy as np
from numpy.testing import assert_array_equal
from shapley_moebius import shapley_moebius
from auxiliary_functions import ishigami_function


def test_bitwise_and():
    k = 4
    inputs_encoded = np.power(2, np.arange(k))
    i = 3

    g = (
        np.bitwise_and(
            i + 1,
            inputs_encoded,
        )
        != 0
    )

    g_expected = np.array([True, True, False, False])

    assert_array_equal(g, g_expected)


def test_h_matrix_non_zero():
    """
    This test should pass if all elements in h_matrix are unequal zero.
    """
    print(1)
    #

    # zero_array

    # assert_raises(AssertionError, assert_array_equal, h_matrix_expected, zero_array)


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
