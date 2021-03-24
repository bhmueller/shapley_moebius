"""
This module contains functions needed for the tests of shapley_moebius.py.
"""

import numpy as np
import chaospy as cp
import scipy.io
from scipy.stats import norm

from oct2py import octave as oct


def ishigami_function(x):
    return np.sin(x[:, 0]) * (1 + 0.1 * (x[:, 2] ** 4)) + 7 * (np.sin(x[:, 1]) ** 2)


def linear_model(x, beta):
    return x.dot(beta)


def trafo_normal(z, mu, var):
    sigma = np.sqrt(var)
    x = mu + z * sigma
    return x


def get_test_values_additive_uniform(k, n, seed, correlation, rank_corr, random_mode):

    u = cp.create_sobol_samples(n, 2 * k, seed).T
    scipy.io.savemat("data/n_inputs.mat", {"k": k})
    scipy.io.savemat("data/n_samples.mat", {"n": n})
    scipy.io.savemat("data/uniform_data.mat", {"u": u})
    # Change directory.
    oct.eval("cd C:/Users/admin/shapley_moebius/data")
    # Run .m file.
    if correlation == "independent":
        oct.eval("get_expected_values_independent")
    elif correlation == "dependent":
        scipy.io.savemat("data/random_mode.mat", {"rmode": random_mode})
        scipy.io.savemat("data/rank_corr.mat", {"C": rank_corr})
        oct.eval("get_expected_values_dependent")
    else:
        print("Please specify correlation.")
    # oct.eval("save -v7 myworkspace.mat")

    expected_results = {
        "subset_size": scipy.io.loadmat("data/subset_size_results.mat")["sz"],
        "h_matrix": scipy.io.loadmat("data/h_matrix_results.mat")["H"],
        "mob": scipy.io.loadmat("data/mob_results.mat")["mob"],
        "variance": scipy.io.loadmat("data/variance_results.mat")["V"],
        "shapley_effects": scipy.io.loadmat("data/shapley_effects_results.mat")["Shap"],
    }
    return expected_results


def get_test_values_sample_data(k, n, seed, rank_corr):

    u = cp.create_sobol_samples(n, 2 * k, seed).T
    scipy.io.savemat("data/n_inputs.mat", {"k": k})
    scipy.io.savemat("data/n_samples.mat", {"n": n})
    scipy.io.savemat("data/u_sample.mat", {"u": u})
    scipy.io.savemat("data/rank_corr.mat", {"C": rank_corr})
    # Change directory.
    oct.eval("cd C:/Users/admin/shapley_moebius/data")
    # Run .m file.
    oct.eval("get_sample_data")

    expected_results = {
        "x_a": scipy.io.loadmat("data/x_a_results.mat")["xa"],
        "x_b": scipy.io.loadmat("data/x_b_results.mat")["xb"],
        "n_a": scipy.io.loadmat("data/n_a_results.mat")["na"],
        "c_b": scipy.io.loadmat("data/c_b_results.mat")["cb"],
    }
    return expected_results


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
