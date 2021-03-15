"""
This module implements algorithm 5.1 from Plischke, Elmar, Giovanni Rabitti, Emanuele
Borgonovo. 2020. Computing Shapley Effects for Sensitivity Analysis. arXiv.
"""
import numpy as np
import chaospy as cp


def shapley_moebius_independent(k, n, model, trafo):
    """Estimate Shapley effects under input independence via Möbius inverse. Shapley
    effects are computed by using two distinct estimators.

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

    Returns:
    -------
    shapley_effects: nd.array
        Array containing the estimated non-normalised Shapley effects.
    variance: nd.array
        Total variance of model output.
    """

    # dim = 2 * k, since want two data sets.
    u = cp.create_sobol_samples(n, 2 * k, 123).T
    # u has n rows and 2 * k columns.

    x_a = trafo(u[:, 0:k])
    x_b = trafo(u[:, k:])

    n_subsets = np.power(2, k) - 1  # l: number of non-vanishing input subsets.
    # Want a 2 power d sequence for binary coding, d = 1, ..., k.
    power_sequence = np.power(2, np.arange(k))

    h_matrix, subset_size = _calc_h_matrix_independent(
        n, model, x_a, x_b, n_subsets, power_sequence
    )

    mob = _calc_mob_independent(n_subsets, h_matrix, subset_size)

    shapley_effects = np.ones((2, k))

    for i in np.arange(k):
        shapley_effects[:, i] = np.sum(
            mob[
                :,
                np.nonzero(np.bitwise_and(np.arange(n_subsets) + 1, np.power(2, i))),
            ][:, 0, :],
            axis=1,
        )

    variance = np.array([[h_matrix[0, -1]], [h_matrix[1, -1]]])

    return shapley_effects, variance


def _calc_h_matrix_independent(n, model, x_a, x_b, n_subsets, power_sequence):

    y_a = model(x_a)
    y_b = model(x_b)

    h_matrix = np.zeros((2, n_subsets))  # H
    subset_size = np.zeros((1, n_subsets))  # sz

    for i in np.arange(n_subsets):
        # g is set of inputs considered, binary encoded.
        g = (
            np.bitwise_and(
                i + 1,
                power_sequence,
            )
            != 0
        )
        subset_size[:, i] = np.sum(
            g
        )  # Sum over boolean: True/False interpreted as 1/0.

        x_i = x_a.copy()  # x_i is always set to x_a again.
        x_i[:, g] = x_b[:, g].copy()  # Some are replaced by values from x_b.
        y_i = model(x_i)
        # Fill columns of h_matrix one by one.
        h_matrix[:, i] = [
            np.mean(np.power((y_i - y_a), 2)) / 2,
            np.dot(y_b.T, (y_i - y_a)) / n,
        ]

    return h_matrix, subset_size


def _calc_mob_independent(n_subsets, h_matrix, subset_size):
    mob = np.zeros((2, n_subsets))
    # sel = 1

    sel = np.zeros(n_subsets, dtype=np.int8)

    for i in np.arange(n_subsets):

        sel[0 : i + 1] = np.bitwise_xor(
            sel[0 : i + 1], np.concatenate((np.ones(1, dtype=np.int8), sel[0:i]))
        )

        # np.nonzero: get indices of nonzero elements in array.
        # ii reads the indices where sel not zero: from 0 1 0 1 -> 1 3 (zero-indexing)
        ii = np.nonzero(sel)

        mob[:, i] = (
            np.dot(
                h_matrix[:, ii], np.power(-1, subset_size[0][i] + subset_size[0][ii].T)
            )[:, 0]
        ) / subset_size[0][i]

        # sel = np.bitwise_xor([1, sel], [sel, 0])

    return mob
