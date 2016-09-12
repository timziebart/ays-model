
from __future__ import division, print_function

import numpy as np
import warnings as warn
import sys

if sys.version_info[0] < 3:
    warn.warn("this code has been tested in Python3 only")


NB_USING_NOPYTHON = True
USING_NUMBA = True
if USING_NUMBA:
    try:
        import numba as nb
    except ImportError:
        warn.warn("couldn't import numba, continuing without", ImportWarning)
        USING_NUMBA = False


if USING_NUMBA:
    jit = nb.jit
else:
    def dummy_decorator_with_args(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        else:
            return dummy_decorator_with_args
    jit = dummy_decorator_with_args


AWS_parameters = {}
AWS_parameters["A_offset"] = 600  # pre-industrial level corresponds to A=0

AWS_parameters["beta"] = 0.03  # 1/yr
AWS_parameters["beta_DG"] = AWS_parameters["beta"] / 2
# AWS_parameters["beta_DG_350"] = AWS_parameters["beta"] / 3
AWS_parameters["epsilon"] = 147.  # USD/GJ
AWS_parameters["rho"] = 2.  # 1
AWS_parameters["phi"] = 47.e9  # GJ/GtC
AWS_parameters["sigma"] = AWS_parameters["sigma_default"] = 1.e12  # GJ
AWS_parameters["sigma_ET"] = AWS_parameters["sigma_default"] * .5**(1/AWS_parameters["rho"])  # GJ
AWS_parameters["tau_A"] = 50.  # yr
AWS_parameters["tau_S"] = 50.  # yr
AWS_parameters["theta"] = AWS_parameters["beta"] / (950 - AWS_parameters["A_offset"])  # 1/(yr GJ)
AWS_parameters["theta_SRM"] = 0.5 * AWS_parameters["theta"]  # 1/(yr GJ)

boundary_parameters = {}
boundary_parameters["A_PB"] = 840 - AWS_parameters["A_offset"]
# boundary_parameters["A_PB_350"] = 735 - AWS_parameters["A_offset"]
boundary_parameters["W_SF"] = 4e13  # year 2000 GWP

grid_parameters = {}

# rescaling parameters
grid_parameters["A_mid"] = boundary_parameters["A_PB"]
# grid_parameters["A_max"] = 500
grid_parameters["W_mid"] = boundary_parameters["W_SF"]
grid_parameters["S_mid"] = 5e10

grid_parameters["n0"] = 40
grid_parameters["grid_type"] = "orthogonal"
border_epsilon = 1e-8
# w, s -> 1 is equiv to W, S -> infty; so keep a small distance to 1 (:
# check everything
# grid_parameters["boundaries"] = [[0, grid_parameters["A_max"]],  # A
grid_parameters["boundaries"] = [[0, 1 - border_epsilon],  # a: rescaled A
                [0, 1 - border_epsilon],  # w: resclaed W
                [0, 1 - border_epsilon]  # s: rescaled S
                ]


def globalize_dictionary(dictionary, module="__main__"):
    if isinstance(module, str):
        module = sys.modules[module]

    for key, val in dictionary.items():
        if hasattr(module, key):
            warn.warn("overwriting global value / attribute '{}' of '{}'".format(key, module.__name__))
        setattr(module, key, val)


# JH: maybe transform the whole to log variables since W,S can go to infinity...
def _AWS_rhs(AWS, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
    A, W, S = AWS
    U = W / epsilon
    F = U / (1 + (S/sigma)**rho)
    R = U - F
    E = F / phi
    Adot = E - A / tau_A
    Wdot = (beta - theta * A) * W
    Sdot = R - S / tau_S
    return Adot, Wdot, Sdot

AWS_rhs = nb.jit(_AWS_rhs, nopython=NB_USING_NOPYTHON)
# AWS_rhs = _AWS_rhs  # used for debugging


@jit(nopython=NB_USING_NOPYTHON)
def AWS_rescaled_rhs(aws, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
    a, w, s = aws
    # A, w, s = Aws
    W = W_mid * w / (1 - w)
    S = S_mid * s / (1 - s)
    A = A_mid * a / (1 - a)

    Adot, Wdot, Sdot = AWS_rhs((A, W, S), t=t, beta=beta, epsilon=epsilon, phi=phi, rho=rho, sigma=sigma, tau_A=tau_A, tau_S=tau_S, theta=theta)

    wdot = Wdot * W_mid / (W_mid + W)**2
    sdot = Sdot * S_mid / (S_mid + S)**2
    adot = Adot * A_mid / (A_mid + A)**2
    return adot, wdot, sdot
    # return Adot, wdot, sdot


@jit(nopython=NB_USING_NOPYTHON)
# def AWS_sunny(Aws):
    # return Aws[:, 0] < A_PB  # planetary boundary
def AWS_sunny(aws):
    return aws[:, 0] < 0.5 # A_PB  # planetary boundary









