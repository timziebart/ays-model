
from __future__ import division

import numpy as np

A_offset = 600  # pre-industrial level corresponds to A=0

beta_default = 0.03  # 1/yr
beta_DG = beta_default / 2
epsilon = 147.  # USD/GJ
phi = 47.e9  # GJ/GtC
tau_A = 50.  # yr
theta = beta_default / (950 - A_offset)  # 1/(yr GJ)

A_PB = 840 - A_offset
W_SF = 4e13  # year 2000 GWP

A_max = 500
W_mid = W_SF


def AW_rhs(AW, t=0, beta=None):
    A, W = AW
    Adot = W / (epsilon * phi) - A / tau_A
    Wdot = (beta - theta * A) * W
    return Adot, Wdot


def AW_rescaled_rhs(aw, t=0, beta=None):
    A, w = aw
    W = W_mid * w / (1 - w)
    Adot, Wdot = AW_rhs((A, W), t=t, beta=beta)
    wdot = Wdot * W_mid / (W_mid + W)**2
    return Adot, wdot


def AW_sunny(AW):
    # A is not needed to be rescaled and is the only one used here

    return AW[:, 0] < A_PB
    # return AW[:, 1] > W_SF
    # return (AW[:, 0] < A_PB) & (AW[:, 1] > W_SF)


def AW_rescaled_sunny(aw):
    AW = np.copy(aw)
    AW[:, 1] = W_mid * aw[:, 1] / (1 - aw[:, 1])
    return AW_sunny(AW)











