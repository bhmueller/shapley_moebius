"""
This module contains functions needed for the tests of shapley_moebius.py.
"""

import numpy as np


def _ishigami_function(x):
    return np.sin(x[:, 1]) * (1 + 0.1 * (x[:, 3] ** 4)) + 7 * (np.sin(x[:, 2]) ** 2)
