"""
This module implements algorithm 5.1 from Plischke, Elmar, Giovanni Rabitti, Emanuele
Borgonovo. 2020. Computing Shapley Effects for Sensitivity Analysis. arXiv.
"""
import numpy as np
import chaospy as cp


def shapley_moebius(k, n, model, trafo):
    """
    Estimate Shapley effects via Möbius inverse.

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
        Array containing the estimated Shapley effects.
    variance: float
        Total variance of model output.
    """

    # Better use chaospy.create_sobol_samples(order, dim, seed=1)
    # order: no. of unique samples, dim: no. of spacial dim.s, i.e. no. of inputs.
    u = cp.create_sobol_samples(n, 2 * k, 123)  # dim = 2 * k, since want two data sets.
    # u = np.random.uniform(0, 1, n)
    # u[1, :] = []  # Delete row 1 in matrix u.
    # u has n rows and 2 * k columns.
    x_a = trafo(u[:, 0 : k - 1])  # First k elements.
    x_b = trafo(u[:, k:])  # Last k elements.
    y_a = model(x_a)
    y_b = model(x_b)

    n_subsets = np.power(2, k) - 1  # l: number of non-vanishing input subsets.
    h_matrix = np.zeros((2, n_subsets))  # H
    subset_size = np.zeros((1, n_subsets))  # sz

    # Iterate over all subsets.
    for i in np.arange(n_subsets):
        # g is set of inputs considered.
        g = np.bitwise_and(
            i,
            np.power(2, np.arange(0, k - 1)) != 0,
        )  # Here sth. missing.
        subset_size[i] = np.sum(g)
        x_i = x_a  # x_i is always set to x_a again?
        x_i[:, g] = x_b[:, g]
        y_i = model(x_i)
        # Fill columns of h_matrix one by one.
        h_matrix[:, i] = [np.mean(np.power((y_i - y_a), 2)), y_b.T * (y_i - y_a) / n]

    mob = np.zeros((2, n_subsets))
    sel = 1

    for i in np.arange(n_subsets):
        ii = np.nonzero(sel)
        mob[:, i] = (
            h_matrix[:, ii] * np.power(-1, subset_size[i] + subset_size[ii].T)
        ) / subset_size[i]
        sel = np.bitwise_xor([1, sel], [sel, 0])

    shapley_effects = np.ones((2, k))

    for i in np.arange(k):
        # sum() in Matlab sums elements in each row and returns a column vector.
        shapley_effects[:, i] = np.sum(
            mob[
                :,
                np.where[np.bitwise_and(np.arange(n_subsets), np.power(2, i - 1)) != 0],
            ],
            1,
        )
    # In Numpy column is axis = 1. In Matlab, dim = 2 is a column; dimension in Matlab:
    # 1 = row, 2 = column.

    variance = h_matrix[:, -1]

    return shapley_effects, variance
