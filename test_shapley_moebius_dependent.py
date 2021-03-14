"""
This file contains tests for shapley_moebius_dependent.py.
"""
import numpy as np
import pandas as pd
from shapley_moebius_dependent import shapley_moebius_dependent


def test_shapley_moebius_dependent():
    """Check whether function runs through."""

    def model(x):
        return np.sum(x, axis=1)

    def trafo(x):
        return x
