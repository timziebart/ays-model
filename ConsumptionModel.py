from __future__ import division

import numpy as np

def consum_rhs(xy, t, u):
    x, y = xy

    dx = x - y
    dy = u
    return [dx, dy]

def consum_rhsPS(xy, t, u):
    x, y = xy

    v = np.zeros_like(x)
    v[:] = u

    dx = x - y
    dy = v
    return [dx, dy]

def consum_sunny(p):
    """sunny constraint"""
    return np.ones(p.shape[:-1], dtype=bool)


