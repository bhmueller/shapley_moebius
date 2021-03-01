"""
This module contains functions needed for the tests of shapley_moebius.py.
"""

import numpy as np
from scipy.stats import norm


def ishigami_function(x):
    return np.sin(x[:, 1]) * (1 + 0.1 * (x[:, 3] ** 4)) + 7 * (np.sin(x[:, 2]) ** 2)


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
