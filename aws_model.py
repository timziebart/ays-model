
from __future__ import division, print_function

from aws_general import __version__, __version_info__
import pyviability as pv

import numpy as np
import warnings as warn
import sys

if sys.version_info[0] < 3:
    warn.warn("this code has been tested in Python3 only", category=DeprecationWarning)

assert pv.version_info >= (0, 15), "please get the latest version of pyviability (and don't forget to (re)run the installation"


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




# long name (command option line style) : short name (lower case)
DEFAULT_NAME = "default"
MANAGEMENTS = {
    "degrowth": "dg",
    "solar-radiation": "srm",
    "energy-transformation": "et",
    "carbon-capture-storage": "ccs",
}

def get_management_parameter_dict(management, all_parameters):
    management_dict = dict(all_parameters) # make a copy
    if management == DEFAULT_NAME:
        return management_dict
    ending = "_" + MANAGEMENTS[management].upper()
    changed = False
    for key in management_dict:
        # choose the variables that are changed by the ending
        if key+ending in management_dict:
            changed = True
            management_dict[key] = management_dict[key+ending]
    if not changed:
        raise NameError("didn't find any parameter for management option "\
                        "'{}' (ending '{}')".format(management, ending))
    return management_dict


AWS_parameters = {}
AWS_parameters["A_offset"] = 600  # pre-industrial level corresponds to A=0

AWS_parameters["beta"] = 0.03  # 1/yr
AWS_parameters["beta_DG"] = AWS_parameters["beta"] / 2
# AWS_parameters["beta_DG_350"] = AWS_parameters["beta"] / 3
AWS_parameters["epsilon"] = 147.  # USD/GJ
AWS_parameters["rho"] = 2.  # 1
AWS_parameters["phi"] = 47.e9  # GJ/GtC
AWS_parameters["phi_CCS"] = AWS_parameters["phi"] * 2
AWS_parameters["sigma"] = 4.e12  # GJ
# AWS_parameters["sigma"] = AWS_parameters["sigma_default"] = 4.e12  # GJ
# AWS_parameters["sigma"] = AWS_parameters["sigma_default"] = 1.e12  # GJ # old
AWS_parameters["sigma_ET"] = AWS_parameters["sigma"] * .5**(1/AWS_parameters["rho"])
AWS_parameters["tau_A"] = 50.  # yr
AWS_parameters["tau_S"] = 50.  # yr
AWS_parameters["theta"] = AWS_parameters["beta"] / (950 - AWS_parameters["A_offset"])  # 1/(yr GJ)
AWS_parameters["theta_SRM"] = 0.5 * AWS_parameters["theta"]

boundary_parameters = {}
boundary_parameters["A_PB"] = 945 - AWS_parameters["A_offset"]  # 450ppm
# boundary_parameters["A_PB"] = 840 - AWS_parameters["A_offset"]  # old 400ppm stuff
# boundary_parameters["A_PB_350"] = 735 - AWS_parameters["A_offset"]
boundary_parameters["W_SF"] = 4e13  # USD, year 2000 GWP

grid_parameters = {}

current_state = [240, 7e13, 5e11]

# rescaling parameters
grid_parameters["A_mid"] = current_state[0]
# grid_parameters["A_mid"] = boundary_parameters["A_PB"]
# grid_parameters["A_max"] = 500
grid_parameters["W_mid"] = current_state[1]
# grid_parameters["W_mid"] = boundary_parameters["W_SF"]
grid_parameters["S_mid"] = current_state[2]
# grid_parameters["S_mid"] = 5e10

grid_parameters["n0"] = 40
grid_parameters["grid_type"] = "orthogonal"
border_epsilon = 1e-3
# w, s -> 1 is equiv to W, S -> infty; so keep a small distance to 1 (:
# check everything
# grid_parameters["boundaries"] = [[0, grid_parameters["A_max"]],  # A
# grid_parameters["boundaries"] = [[0, 1 - border_epsilon],  # a: rescaled A
                # [0, 1 - border_epsilon],  # w: resclaed W
                # [0, 1 - border_epsilon]  # s: rescaled S
                # ]
grid_parameters["boundaries"] = np.array([[0, 1],  # a: rescaled A
                [0, 1],  # w: resclaed W
                [0, 1]  # s: rescaled S
                ], dtype=float)
# use the full stuff in the S direction
grid_parameters["boundaries"][:2, 0] = grid_parameters["boundaries"][:2, 0] + border_epsilon
grid_parameters["boundaries"][:2, 1] = grid_parameters["boundaries"][:2, 1] - border_epsilon


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

    s_inv = 1 - s
    s_inv_rho = s_inv ** rho
    K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho )

    a_inv = 1 - a
    w_inv = 1 - w
    W = W_mid * w / w_inv
    A = A_mid * a / a_inv
    adot = K / (phi * epsilon * A_mid) * a_inv * a_inv * W - a * a_inv / tau_A
    wdot = w * w_inv * ( beta - theta * A )
    sdot = (1 - K) * s_inv * s_inv * W / (epsilon * S_mid) - s * s_inv / tau_S

    return adot, wdot, sdot


# @jit(nopython=NB_USING_NOPYTHON)
def AWS_sunny_PB(aws):
    return aws[:, 0] < A_PB / (A_PB + A_mid) # transformed A_PB  # planetary boundary

# @jit(nopython=NB_USING_NOPYTHON)
def AWS_sunny_SF(aws):
    return aws[:, 1] > W_SF / (W_SF + W_mid) # transformed W_SF  # social foundation

# @jit(nopython=NB_USING_NOPYTHON)
def AWS_sunny_PB_SF(aws):
    return np.logical_and(aws[:, 0] < A_PB / (A_PB + A_mid), aws[:, 1] > W_SF / (W_SF + W_mid)) # both transformed









