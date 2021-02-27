"""
This module implements algorithm 5.1 from Plischke, Elmar, Giovanni Rabitti, Emanuele
Borgonovo. 2020. Computing Shapley Effects for Sensitivity Analysis. arXiv.
"""
import numpy as np
import chaospy as cp


def model(x):
    return np.sum(x)


def trafo(x):
    return x


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
    u = cp.create_sobol_samples(
        n, 2 * k, 123
    ).T  # dim = 2 * k, since want two data sets.
    # assert_array_equal(u.shape, np.array([n, 2 * k]))
    # u = np.random.uniform(0, 1, n)
    # u[1, :] = []  # Delete row 1 in matrix u.
    # u has n rows and 2 * k columns.
    x_a = trafo(u[:, 0:k])  # First k elements.
    x_b = trafo(u[:, k:])  # Last k elements.
    y_a = model(x_a)
    y_b = model(x_b)

    n_subsets = np.power(2, k) - 1  # l: number of non-vanishing input subsets.
    inputs_encoded = np.power(2, np.arange(k))
    h_matrix = np.zeros((2, n_subsets))  # H
    subset_size = np.zeros((1, n_subsets))  # sz

    # Iterate over all subsets.
    for i in np.arange(n_subsets):
        # g is set of inputs considered, binary encoded.
        g = (
            np.bitwise_and(
                i + 1,
                inputs_encoded,
            )
            != 0
        )
        subset_size[:, i] = np.sum(
            g
        )  # Sum over boolean: True/False interpreted as 1/0.
        x_i = x_a.copy()  # x_i is always set to x_a again.
        # g_0 = np.where(g == False)
        x_i[:, g] = x_b[:, g].copy()  # Some are replaced by values from x_b.
        y_i = model(x_i)
        # Fill columns of h_matrix one by one.
        h_matrix[:, i] = [np.mean(np.power((y_i - y_a), 2)), (y_b.T * (y_i - y_a)) / n]

    mob = np.zeros((2, n_subsets))
    sel = 1

    for i in np.arange(n_subsets):
        ii = np.nonzero(sel)  # First sel = 1 (see above).
        mob[:, i] = (
            h_matrix[:, ii] * np.power(-1, subset_size[i] + subset_size[ii].T)
        ) / subset_size[i]
        sel = np.bitwise_xor([1, sel], [sel, 0])

    shapley_effects = np.ones((2, k))

    for i in np.arange(k):
        # sum(dim=2) in Matlab sums elements in each row and returns a column vector.
        shapley_effects[:, i] = np.sum(
            mob[
                :,  # Below line is faulty.
                np.where[np.bitwise_and(np.arange(n_subsets), np.power(2, i - 1)) != 0],
            ],
            1,
        )
    # In Numpy column is axis = 1. In Matlab, dim = 2 is a column; dimension in Matlab:
    # 1 = row, 2 = column.

    variance = h_matrix[:, -1]

    return shapley_effects, variance
