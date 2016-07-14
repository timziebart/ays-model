
from __future__ import division

import numpy as np


def plants_rhs(xy, t, prod, ax, ay):
    x, y = xy
    dx = x * (np.sqrt(x) * (1 - y) - x) - ax * x
    dy = prod * y * (np.sqrt(y) * (1 - x) - y) - ay * y
    return [dx, dy]


def plants_sunny(p):
    """sunny constraint"""
    return np.sum(p, axis = -1) > 0.65


