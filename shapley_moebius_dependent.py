"""
This module implements algorithm 5.1 for dependent inputs from Plischke, Elmar, Giovanni
Rabitti, Emanuele Borgonovo. 2020. Computing Shapley Effects for Sensitivity Analysis. arXiv.
"""
import numpy as np
import chaospy as cp
from scipy.stats import norm


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
        Rank correlation matrix of inputs for dependent sampling.
    random_mode: str
        Choose how initial uniformly distributed samples are drawn: 'random' for pseudo-
        random (MC) sampling and 'Sobol' for quasi-random (QMC) sampling.

    Returns:
    -------
    shapley_effects: nd.array
        Array containing the estimated non-normalised Shapley effects.
    variance: float
        Total variance of model output.
    """

    if random_mode.lower() == "random":
        u = np.random.rand(n, 2 * k)

    elif random_mode.lower() == "sobol":
        u = cp.create_sobol_samples(n, 2 * k, 123).T

    else:
        print("Error. Please specify random_mode by either random or Sobol")

    return u
