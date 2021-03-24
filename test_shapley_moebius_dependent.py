"""
This file contains tests for shapley_moebius_dependent.py.
"""
import numpy as np
import pandas as pd
import chaospy as cp
from functools import partial
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from shapley_moebius_dependent import shapley_moebius_dependent
from shapley_moebius_dependent import _calc_h_matrix_dependent
from auxiliary_functions import get_test_values_additive_uniform
from auxiliary_functions import linear_model
from auxiliary_functions import trafo_normal


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

    h_matrix_actual, subset_size_actual, evals = _calc_h_matrix_dependent(
        k, n, u, model, trafo, n_subsets, rank_corr
    )

    # assert_array_almost_equal(h_matrix_actual, expected_results['h_matrix'][0])

    # assert_array_equal(subset_size_actual, expected_results['subset_size'][0])


def test_linear_two_inputs():
    """Test case taken from IP19, section 3.2. A linear model with two correlated
    Gaussian inputs is considered."""

    beta_1 = 1.3
    beta_2 = 1.5
    beta = np.array([beta_1, beta_2])
    var_1 = 16.0
    var_2 = 4.0
    var = np.array([var_1, var_2])
    # mu has no effect on Shapley effects in this setting.
    mu = np.array([0.0, 0.0])
    rho = 0.3

    # Calculate analytical Shapley effects.
    component_1 = beta_1 ** 2 * var_1
    component_2 = beta_2 ** 2 * var_2
    covariance = rho * np.sqrt(var_1) * np.sqrt(var_2)
    var_y = component_1 + 2 * covariance * beta_1 * beta_2 + component_2
    share = 0.5 * (rho ** 2)
    true_shapley_1 = (
        component_1 * (1 - share) + covariance * beta_1 * beta_2 + component_2 * share
    ) / var_y
    true_shapley_2 = (
        component_2 * (1 - share) + covariance * beta_1 * beta_2 + component_1 * share
    ) / var_y

    shapley_expected = np.array([[true_shapley_1, true_shapley_2]])

    model = partial(linear_model, beta=beta)
    trafo = partial(trafo_normal, mu=mu, var=var)

    k = 2
    n = 10 ** 7
    rank_corr = np.array([[1, rho], [rho, 1]])
    random_mode = "Sobol"

    shapley_effects, variance, evals = shapley_moebius_dependent(
        k, n, model, trafo, rank_corr, random_mode
    )
    shapley_actual = shapley_effects / variance

    assert_array_almost_equal(shapley_actual, shapley_expected)


def test_linear_three_inputs():
    """Test case taken from IP19, section 3.3. A linear model with three correlated
    Gaussian inputs is considered."""

    beta_1 = 1.3
    beta_2 = 1.5
    beta_3 = 2.5
    beta = np.array([beta_1, beta_2, beta_3])
    var_1 = 16
    var_2 = 4
    var_3 = 9
    var = np.array([var_1, var_2, var_3])
    # Again, mu does not affect analytical Shapley effects.
    mu = np.array([0.0, 0.0, 0.0])
    rho = 0.3

    component_1 = beta_1 ** 2 * var_1
    component_2 = beta_2 ** 2 * var_2
    component_3 = beta_3 ** 2 * var_3
    covariance = rho * np.sqrt(var_2) * np.sqrt(var_3)
    var_y = component_1 + component_2 + component_3 + 2 * covariance * beta_2 * beta_3
    share = 0.5 * (rho ** 2)
    true_shapley_1 = (component_1) / var_y
    true_shapley_2 = (
        component_2 + covariance * beta_2 * beta_3 + share * (component_3 - component_2)
    ) / var_y
    true_shapley_3 = (
        component_3 + covariance * beta_2 * beta_3 + share * (component_2 - component_3)
    ) / var_y

    shapley_expected = np.array([[true_shapley_1, true_shapley_2, true_shapley_3]])

    model = partial(linear_model, beta=beta)
    trafo = partial(trafo_normal, mu=mu, var=var)

    k = 3
    n = 10 ** 7
    rank_corr = np.array([[1, 0, 0], [0, 1, rho], [0, rho, 1]])
    random_mode = "Sobol"

    shapley_effects, variance, evals = shapley_moebius_dependent(
        k, n, model, trafo, rank_corr, random_mode
    )
    shapley_actual = shapley_effects / variance

    assert_array_almost_equal(shapley_actual, shapley_expected)


def test_additive():
    """Test case taken from IP19, section 3.5. An additive model with an interaction
    with three correlated Gaussian inputs is considered."""
    print("hi")
