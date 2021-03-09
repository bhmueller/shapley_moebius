"""
This module contains functions needed for the tests of shapley_moebius.py.
"""

import numpy as np
import chaospy as cp
import scipy.io
from scipy.stats import norm

# from oct2py import octave as oct


def ishigami_function(x):
    return np.sin(x[:, 0]) * (1 + 0.1 * (x[:, 2] ** 4)) + 7 * (np.sin(x[:, 1]) ** 2)


# def get_test_values_additive_uniform(k, n):
#     np.random.seed(123)
#     u = cp.create_sobol_samples(n, 2 * k, 123).T
#     scipy.io.savemat('n_inputs.mat', {'k': k})
#     scipy.io.savemat('n_samples.mat', {'n': n})
#     scipy.io.savemat('uniform_data.mat', {'u': u})
#     # Change directory.
#     oct.eval("cd C:/Users/admin/shapley_moebius")
#     # Run .m file.
#     oct.eval("shapmob_testing")
#     # oct.eval("save -v7 myworkspace.mat")

#     expected_results = {
#         'subset_size': scipy.io.loadmat('subset_size_results.mat')['sz'],
#         'h_matrix': scipy.io.loadmat('h_matrix_results.mat')['H'],
#         'mob': scipy.io.loadmat('mob_results.mat')['mob'],
#         'variance': scipy.io.loadmat('variance_results.mat')['V'],
#         'shapley_effects': scipy.io.loadmat('shapley_effects_results.mat')['Shap'],}
#     return expected_results


def transformation_mvnorm(u):
    """Transform uniformly distributed variables into a multivariate normal
    distribution.

    Parameters
    ----------
    u : nd.array
        Matrix of uniformly distributed variables.

    Returns
    -------
    x : nd.array
        Matrix of normally distributed variables.

    """

    # Convert to standard normal draws.
    # stnorm_data = norm.ppf(u)
