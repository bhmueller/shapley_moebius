"""
This module implements algorithm 5.1 for dependent inputs from Plischke, Elmar, Giovanni
Rabitti, Emanuele Borgonovo. 2020. Computing Shapley Effects for Sensitivity Analysis. arXiv.
"""
import numpy as np
import chaospy as cp
from scipy.stats import norm
from shapley_moebius_independent import _calc_mob
from shapley_moebius_independent import _calc_shapley_effects


def shapley_moebius_dependent(k, n, model, trafo, rank_corr, random_mode):
    """Estimate Shapley effects under input dependence via Möbius inverse.

    Parameters
    ----------
    k: int
        Number of inputs.
    n: int
        Number of basic sample blocks.
    model: function
        Function that maps realisations of inputs to output(s).
    trafo: function
        Function that transforms uniformly distributed variables into
        desired marginal distribution, e.g. inverse Rosenblatt transformation.
    rank_corr: nd.array
        (Pearson) rank correlation matrix of inputs for dependent sampling. C needs to
        be positive-definite for the Cholesky decomposition to work.
    random_mode: str
        Choose how initial uniformly distributed samples are drawn: 'random' for pseudo-
        random (MC) sampling and 'Sobol' for quasi-random (QMC) sampling.

    Returns:
    -------
    shapley_effects: nd.array
        Array containing the estimated non-normalised Shapley effects.
    variance: float
        Total variance of model output.
    evals: int
        Number of model evaluations.
    """

    if random_mode.lower() == "random":
        u = np.random.rand(n, 2 * k)

    elif random_mode.lower() == "sobol":
        u = cp.create_sobol_samples(n, 2 * k, 123).T

    else:
        print("Error. Please specify random_mode by either random or Sobol")

    n_subsets = np.power(2, k) - 1
    # power_sequence = np.power(2, k) - 1

    h_matrix, subset_size, evals = _calc_h_matrix_dependent(
        k, n, u, model, trafo, n_subsets, rank_corr
    )

    # Get Shapley effects: As in fct. with indep. inputs.

    mob = _calc_mob(n_subsets, h_matrix, subset_size)

    shapley_effects, variance = _calc_shapley_effects(k, n_subsets, mob, h_matrix)

    return shapley_effects, variance, evals


def _sample_data(s_matrix, u, k, trafo):
    n_a = norm.ppf(u[:, 0:k])  # Normal a sample.
    # WIP: Correct?
    x_a = trafo(norm.cdf(np.dot(n_a, s_matrix)))
    c_b = np.dot(norm.ppf(u[:, k:]), s_matrix)  # Correlated b sample.
    x_b = trafo(norm.cdf(c_b))

    return n_a, c_b, x_a, x_b


def _calc_h_matrix_dependent(k, n, u, model, trafo, n_subsets, rank_corr):

    # WIP: Transposed or not? It worked with the non-transposed Chol matrix.
    s_matrix = np.linalg.cholesky(rank_corr).T

    n_a, c_b, x_a, x_b = _sample_data(s_matrix, u, k, trafo)

    y_a = model(x_a)
    y_b = model(x_b)

    # WIP: Check whether need length n_subsets - 1 instead.
    h_matrix = np.zeros((1, n_subsets))
    subset_size = np.zeros((1, n_subsets))
    input_indices = np.arange(n_subsets) + 1

    g = (((input_indices[:, None] & (1 << np.arange(k)))) > 0).astype(bool)

    # Count the number of model evaluations.
    evals = 2

    for i in np.arange(n_subsets):

        # WIP: Check whether + 1 is needed. I think it's not.
        current_subset_size = np.sum(g[i])
        subset_size[:, i] = current_subset_size

        g_current = np.nonzero(g[i])[0]
        g_complement = np.where(g[i] == 0)[0]

        element_1 = rank_corr[g_current, :][:, g_current]
        element_2 = rank_corr[g_current, :][:, g_complement]
        element_3 = rank_corr[g_complement, :][:, g_current]
        element_4 = rank_corr[g_complement, :][:, g_complement]

        # Construct new array.
        new_1 = np.hstack((element_1, element_2))
        new_2 = np.hstack((element_3, element_4))
        helper_matrix = np.vstack((new_1, new_2))

        d_matrix = np.linalg.cholesky(helper_matrix).T

        # WIP: Is index correct? Do we need current_subset_size + 1 in the second and
        # third case?
        d_11 = d_matrix[current_subset_size:, current_subset_size:]
        d_22 = d_matrix[0:current_subset_size, 0:current_subset_size]
        d_21 = d_matrix[0:current_subset_size, current_subset_size:]

        c_i = c_b.copy()

        c_i[:, g_complement] = np.dot(n_a[:, g_complement], d_11) + np.dot(
            c_b[:, g_current], np.linalg.lstsq(d_22, d_21, rcond=None)[0]
        )

        x_i = trafo(norm.cdf(c_i))

        y_i = model(x_i)

        evals = evals + 1

        h_matrix[:, i] = np.dot(y_b.T, (y_i - y_a)) / n

    return h_matrix, subset_size, evals
