

from __future__ import division

import numpy as np


pi = np.pi


def pendulum_rhs(theta_omega, t, a = None):  # raises an error a not set
    theta, omega = theta_omega
    dtheta = omega
    domega = -np.sin(theta) - a
    return [dtheta, domega]


def pendulum_sunny(p):
    """sunny constraint for gravity Pendulum"""
    return np.abs(p[:, 1])<0.5

