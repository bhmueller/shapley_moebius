"""
This file contains tests for shapley_moebius_dependent.py.

Test cases referenced in the following by IP19 were taken from:
Iooss, Betrand and Clémentine Prieur. 2019.
    Shapley effects for sensitivity analysis with correlated inputs: comparisons with
    Sobol’ indices, numerical estimation and applications. hal-01556303v6
"""
import scipy.io
import numpy as np
import chaospy as cp
from functools import partial
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_allclose
from shapley_moebius_dependent import shapley_moebius_dependent
from shapley_moebius_dependent import _calc_h_matrix_dependent
from shapley_moebius_dependent import _sample_data
from auxiliary import get_test_values_sample_data
from auxiliary import linear_model
from auxiliary import trafo_normal
from auxiliary import additive_model


get_expected_values = False


def test_shapley_moebius_dependent():
    """Check whether function runs through for very simple case."""

    k = 3
    n = 100

    def model(x):
        return np.sum(x, axis=1)

    def trafo(x):
        return x

    rank_corr = np.array([[1, -0.5, 0.5], [-0.5, 1, 0], [0.5, 0, 1]])

    random_mode = "Sobol"

    shapley_moebius_dependent(k, n, model, trafo, rank_corr, random_mode)


def test_sample_data():
    """Test for data transformed to a certain non-standard normal distribution."""

    k = 3
    n = 10
    seed = 123
    u = cp.create_sobol_samples(n, 2 * k, seed).T

    mu = np.zeros(k)
    var_1 = 16
    var_2 = 4
    var_3 = 9
    var = np.array([var_1, var_2, var_3])

    rank_corr = np.array([[1, -0.5, 0.5], [-0.5, 1, 0], [0.5, 0, 1]])
    s_matrix = np.linalg.cholesky(rank_corr).T

    trafo = partial(trafo_normal, mu=mu, var=var)

    if get_expected_values is True:
        expected_results = get_test_values_sample_data(k, n, seed, rank_corr)
    else:
        expected_results = {
            "x_a": scipy.io.loadmat("data/x_a_results.mat")["xa"],
            "x_b": scipy.io.loadmat("data/x_b_results.mat")["xb"],
            "n_a": scipy.io.loadmat("data/n_a_results.mat")["na"],
            "c_b": scipy.io.loadmat("data/c_b_results.mat")["cb"],
        }

    n_a, c_b, x_a, x_b = _sample_data(s_matrix, u, k, trafo)

    assert_array_almost_equal(n_a, expected_results["n_a"])

    assert_array_almost_equal(c_b, expected_results["c_b"])

    assert_array_almost_equal(x_a, expected_results["x_a"])

    assert_array_almost_equal(x_b, expected_results["x_b"])


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

    n_subsets = np.power(2, k) - 1

    rank_corr = np.array([[1, -0.5, 0.5], [-0.5, 1, 0], [0.5, 0, 1]])

    h_matrix_actual, subset_size_actual, evals = _calc_h_matrix_dependent(
        k, n, u, model, trafo, n_subsets, rank_corr
    )


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
    n = 10 ** 6
    rank_corr = np.array([[1, rho], [rho, 1]])
    random_mode = "Sobol"

    shapley_effects, variance, evals = shapley_moebius_dependent(
        k, n, model, trafo, rank_corr, random_mode
    )
    shapley_actual = shapley_effects / variance

    assert_allclose(shapley_actual, shapley_expected, rtol=0.0075)


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
    n = 10 ** 6
    rank_corr = np.array([[1, 0, 0], [0, 1, rho], [0, rho, 1]])
    random_mode = "Sobol"

    shapley_effects, variance, evals = shapley_moebius_dependent(
        k, n, model, trafo, rank_corr, random_mode
    )
    shapley_actual = shapley_effects / variance

    assert_allclose(shapley_actual, shapley_expected, rtol=0.016)


def test_additive():
    """Test case taken from IP19, section 3.5. An additive model with an interaction
    with three correlated Gaussian inputs is considered."""

    var_1 = 1
    var_2 = 1
    var_3 = 1
    var = np.array([var_1, var_2, var_3])
    # In IP19 mu is explicitly set to zero vector.
    mu = np.array([0.0, 0.0, 0.0])
    rho = 0.3
    # Variance obtained analytically by myself.
    var_y = var_1 + var_2 * var_3

    true_shapley_1 = (
        (var_1 * (1 - ((rho ** 2) / 2))) + (((var_2 * var_3) * (rho ** 2)) / 6)
    ) / var_y
    true_shapley_2 = (((var_2 * var_3) * (3 + (rho ** 2))) / 6) / var_y
    true_shapley_3 = (
        ((var_1 * (rho ** 2)) / 2) + (((var_2 * var_3) * (3 - (2 * (rho ** 2)))) / 6)
    ) / var_y

    shapley_expected = np.array([[true_shapley_1, true_shapley_2, true_shapley_3]])

    trafo = partial(trafo_normal, mu=mu, var=var)

    k = 3
    n = 10 ** 6
    rank_corr = np.array([[1, 0, rho], [0, 1, 0], [rho, 0, 1]])
    random_mode = "Sobol"

    shapley_effects, variance, evals = shapley_moebius_dependent(
        k, n, additive_model, trafo, rank_corr, random_mode
    )
    shapley_actual = shapley_effects / variance

    # Test non-normalised Shapley effects.
    assert_array_almost_equal(shapley_effects, shapley_expected * var_y)

    assert_array_almost_equal(shapley_actual, shapley_expected)
