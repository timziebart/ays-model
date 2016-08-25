

from __future__ import division

import BaseODEs

import PyViability as viab

import numpy as np
import numpy.linalg as la 
import warnings as warn


## import matplotlib as mpl
## mpl.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patch

from scipy.optimize import broyden2 as solveZero

import scipy.integrate as integrate

import sys, os.path

from myPhaseSpaceL import plotPhaseSpace, savePlot

from SODEoneL import SODEone

from PTopologyL import *

import time

import numba as nb


pi = np.pi

stylePoint["markersize"] *= 2


def techChange_rhs(uB_pB, t, rvar, pBmin, pE, delta, smax, sBmax):
    uB, pB = uB_pB
    if sBmax == None:
        p = pE
    else:
        if smax < sBmax * uB:
            p = pE + smax / uB
        else:
            p = sBmax + pE

    duB = rvar * uB * (1 - uB) * (p - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)

    return np.array([duB, dpB])

def techChange_sunny(p):
    """sunny constraint for techChangeModel"""
    return p[:, 0] > 0.325

def techChange_rhsPS(uB_pB, t, rvar, pBmin, pE, delta, smax, sBmax):
    uB, pB = uB_pB
    p = np.zeros_like(pB)
    p[:] = sBmax + pE
    mask = (smax < sBmax * uB)
    p[mask] = (pE + smax / uB[mask])

    duB = rvar * uB * (1 - uB) * (p - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
    return np.array([duB, dpB])

def evol_noManagement(uB_pB, t=0): # rvar, pBmin,
       # pE, sigmaA, uDedge, pA, delta, smax, sBmax):
    uB, pB = uB_pB
    duB = rvar * uB * (1 - uB) * (pE - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
    return np.array([duB, dpB])


def evol_Management1(uB_pB, t=0):
    uB, pB = uB_pB
    duB = rvar * uB * (1 - uB) * ((pA - (pA - pE) * sigmaA) - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
    return np.array([duB, dpB])


def evol_Management2(uB_pB, t=0):
    uB, pB = uB_pB

    if smax < sBmax * uB:
        p = pE + smax / uB
    else:
        p = sBmax + pE

    duB = rvar * uB * (1 - uB) * (p - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
    return np.array([duB, dpB])

def evol_Management2_PS(uB_pB, t=0):
    uB, pB = uB_pB
    p = np.zeros_like(pB)
    p[:] = sBmax + pE
    mask = (smax < sBmax * uB)
    p[mask] = (pE + smax / uB[mask])

    duB = rvar * uB * (1 - uB) * (p - pB)
    dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
    return np.array([duB, dpB])


class TechChangeXY(BaseODEs.BaseODEs):

    """\
Implementation of tech change model
"""


    def __init__(self, management, comment="", **params):
        if management == "default":
            techChange_rhs= evol_noManagement
            rhs_PS = techChange_rhs
            params = {}
        elif management == "one":
            techChange_rhs = evol_Management1
            rhs_PS = techChange_rhs
            params = {}
        elif management == "two":
            techChange_rhs = evol_Management2
            rhs_PS = evol_Management2_PS
            params = {}
        else:
            raise ValueError('You have to define the management options: Write "no" for no management, '
                             '"one" for management option 1 and "two" for managemante option 2')

        BaseODEs.BaseODEs.__init__(self, techChange_rhs, params, comment=comment, rhs_PS = rhs_PS)

    def plotPhaseSpace(self, boundaries, style, alpha=None):
        plotPhaseSpace(self.rhs_PS, boundaries, colorbar=False, style=style, alpha=alpha)


class TechChangeXY2(SODEone):


    def __init__(self, management):
        if management == "no":
             self.evol = self.evol_noManagement
        elif management == "one":
             self.evol = self.evol_Management1
        elif management == "two":
             self.evol = self.evol_Management2
        else:
             raise ValueError('You have to define the management options: Write "no" for no management, '
                                      '"one" for management option 1 and "two" for managemante option 2')

        SODEone.__init__(self, self.evol)
         # set functions for displaying
        self.plotPhaseSpace = lambda boundaries, style, alpha=None: plotPhaseSpace(self.evol, boundaries,
                                                                                    colorbar=False, style=style,
                                                                                    alpha=alpha)

        self.plotTrajectoryPS = lambda style={"color": "blue"}, stop=None: SODEone.plotTrajectoryPS(self, 0, 1,
                                                                                                    style=style,
                                                                                                     stop=stop)
    def evol_noManagement(self, uB_pB, t=0):
        uB, pB = uB_pB
        duB = rvar*uB*(1-uB)*(pE-pB)
        dpB = -(pB-pBmin)*((pB-pBmin)*uB-delta)
        return np.array([duB, dpB])

    def evol_Management1(self, uB_pB, t=0):
        uB, pB = uB_pB
        duB = rvar * uB * (1 - uB) * ((pA-(pA-pE)*sigmaA) - pB)
        dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
        return np.array([duB, dpB])
    def evol_Management2(self, uB_pB, t=0):
        uB, pB = uB_pB
        if pB.size > 1:
            p = np.zeros_like(pB)
            p[:] = sBmax + pE
            mask = (smax < sBmax * uB)
            p[mask] = (pE + smax / uB[mask])
        else:
            if smax < sBmax * uB:
                p = pE + smax / uB
            else:
                p = sBmax + pE

        duB = rvar * uB * (1 - uB) * (p - pB)
        dpB = -(pB - pBmin) * ((pB - pBmin) * uB - delta)
        return np.array([duB, dpB])


def patchit(*traj, **kwargs):
        ax.add_patch(patch.Polygon(np.transpose(np.hstack(traj)) , facecolor = kwargs["color"], **stylePatch))


def pointIt(xy, color):
        plt.plot([xy[0]],[xy[1]], marker = "8", color = color, **stylePoint)


def is_sunnyTC(p):
    """sunny constraint for techChangeModel"""
    return p[:, 0]>uDedge

if __name__ == "__main__":

        args = sys.argv[1:]

        save = "save" in args
        notshow = "notshow" in args

        # global prefix is just the script name without extension
        globalprefix = os.path.splitext(os.path.basename(__file__))[0]

        # prefix/suffix depends on the chosen options
        prefix = ""
        suffix = ""

        # suffix if save
        if save:
                p = args.index("save")
                if p + 1 < len(args):
                        suffix += args[p + 1]

        boundaries = [0, 0, 1, 2]
        xmin, ymin, xmax, ymax = boundaries

        fig2 = plt.figure(figsize=(15, 15))
        ax = fig2.add_subplot(111)

        # default values
        rvar = 1
        pBmin = 0.15
        pE = 0.3
        sigmaA=1.05
        uDedge=0.325
        pA=0.5
        delta=0.025
        smax=0.3
        sBmax=0.5
        globalprefix += "_a=%1.2f_prod=%1.1f_desire=%1.2f_"%(rvar, pBmin, pE)

        # different instances of the model
        moddefTC = TechChangeXY('no')
        mod1TC = TechChangeXY('one')
        mod2TC = TechChangeXY('two')

        plt.xlabel("$u_B$")
        plt.ylabel("$p_B$")
        plt.axis([xmin, xmax, ymin, ymax])


        defaultTC_run = viab.make_run_function2(moddefTC, 1)
        management1TC_run = viab.make_run_function2(mod1TC, 1)
        management2TC_run = viab.make_run_function2(mod2TC, 1)

        viab.MAX_STEP_NUM = 4
        x_num = 80

        x_step, x_half_step, xy = viab.generate_2Dgrid(xmin, xmax, x_num, ymin, ymax)

        viab.x_step = x_step

        state = np.zeros(xy.shape[:-1])

        viab.topology_classification(xy, state, defaultTC_run, management2TC_run, is_sunnyTC)

        moddefTC.plotPhaseSpace(boundaries, style=styleDefault)
        mod2TC.plotPhaseSpace(boundaries, style = styleMod1)


        viab.plot_points(xy, state)
        #plt.ylim([0, 2])
        #plt.xlim([0, 1])

        if save:
            # filename = globalprefix+prefix+suffix+".svg"
            # filename = globalprefix+prefix+suffix+".eps"
            filename = globalprefix+prefix+suffix+".pdf"
            savePlot(filename)

        if not notshow:
            plt.show()
