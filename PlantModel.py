
from __future__ import division

import BaseODEs

import PyViability as viab

import numpy as np
import numpy.linalg as la 
import warnings as warn

#import matplotlib as mpl
#mpl.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patch

from scipy.optimize import broyden2 as solveZero

import sys, os.path

from myPhaseSpaceL import plotPhaseSpace, savePlot

from SODEoneL import SODEone

from PTopologyL import *

import time

import numba as nb


pi = np.pi

stylePoint["markersize"] *= 2


def plant_rhs(xy, t, prod, ax, ay):
    x, y = xy
    dx = x * (np.sqrt(x) * (1 - y) - x) - ax * x
    dy = prod * y * (np.sqrt(y) * (1 - x) - y) - ay * y
    return np.array([dx, dy])


class PlantXY(BaseODEs.BaseODEs):

    def __init__(self, comment = "", **params):
        assert set(["prod", "ax", "ay"]).issubset(params)
        BaseODEs.BaseODEs.__init__(self, plant_rhs, params, comment = comment)

    def steadyState(self):
        q = 2 * self['ay'] / self['prod']
        p = 2 * self['ax']
        return [ ( 0, ((1-p) + np.sqrt(1 - 2*p))/2 ), ( 0, ((1-q) + np.sqrt(1 - 2*q))/2 )]

    def plotPhaseSpace(self, boundaries, style, alpha = None):
        plotPhaseSpace(self.rhs_PS, boundaries, colorbar = False, style = style, alpha = alpha)


class PlantXY2(SODEone):
        """\
Implementation of competing plant growth
"""
        def __init__(self, prod = None, ax = None, ay = None):
                assert not None in [prod, ax, ay]
                SODEone.__init__(self, self.evol)
                self.prod = float(prod)
                self.ax = float(ax)
                self.ay = float(ay)

                # set functions for displaying
                self.plotPhaseSpace = lambda boundaries, style, alpha = None: plotPhaseSpace(self.evol, boundaries, colorbar = False, style = style, alpha = alpha)

                self.plotTrajectoryPS = lambda style = {"color": "blue"}, stop = None: SODEone.plotTrajectoryPS(self, 0, 1, style = style, stop = stop)

        def evol(self, xy, t):
                x, y = xy
                dx = x * (np.sqrt(x) * (1 - y) - x) - self.ax * x
                dy = self.prod * y * (np.sqrt(y) * (1 - x) - y) - self.ay * y
                return np.array([dx, dy])

        def steadyState(self):
                q = 2 * self.ay / self.prod
                p = 2 * self.ax
                #print solveZero(lambda xy: self.evol(xy, 0), [0.4, 0.2], f_tol = 1e-10)
                return [ ( 0, ((1-p) + np.sqrt(1 - 2*p))/2 ), ( 0, ((1-q) + np.sqrt(1 - 2*q))/2 )]


def patchit(*traj, **kwargs):
        ax.add_patch(patch.Polygon(np.transpose(np.hstack(traj)) , facecolor = kwargs["color"], **stylePatch))


def pointIt(xy, color):
        plt.plot([xy[0]],[xy[1]], marker = "8", color = color, **stylePoint)


##############################################################
## from here on is the stuff for the viability calculations
##############################################################

def dummy_constraint(p):
    """used when no constraint is applied"""
    return np.ones(p.shape[:-1]).astype(np.bool)


def is_sunny(p):
    """sunny constraint"""
    return np.sum(p, axis = -1) > xplusy


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

        boundaries = [1e-3, 1e-3, 1, 1]
        xmin, ymin, xmax, ymax = boundaries

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)

        # default values
        a = 0.2 # harvest value
        xplusy = 0.65 # min sum of both for being desirable state
        prod = 2 # factor of productivity for y

        globalprefix += "_a=%1.2f_prod=%1.1f_desire=%1.2f_"%(a, prod, xplusy)

        # different instances of the model
        moddef = PlantXY( prod = prod, ax = a, ay = a )
        mod1 = PlantXY( prod = prod, ax = a/2, ay = a/2 )
        mod2 = PlantXY( prod = prod, ax = 2*a, ay = 0 )


        # dark upstream
        b3 = np.array([[0, xplusy, 0], [xplusy, 0, 0]])
        patchit(b3, color = cDarkUp)

        if "0" in args:
            moddef.plotPhaseSpace(boundaries, style = styleDefault)
            pointIt([0.5, 0], cDefault)
            pointIt([0, 0.75], cDefault)
        if "1" in args:
                prefix += "1"
                mod1.plotPhaseSpace(boundaries, style = styleMod1)
                pointIt([0.79, 0], cMod)
                pointIt([0, 0.86], cMod)
                ax.text(-0.05, 1.03, "a")
        if "2" in args:
                prefix += "2"
                mod2.plotPhaseSpace(boundaries, style = styleMod2)
                pointIt([0, 1], cMod)
                ax.text(-0.05, 1.03, "b")
        
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.axis([xmin, xmax, ymin, ymax])


        #######################
        ## viability stuff starts here
        #######################

        default_run = viab.make_run_function2(moddef, 1)
        management1_run = viab.make_run_function2(mod1, 1)
        management2_run = viab.make_run_function2(mod2, 1)

        MAX_EVOLUTION_NUM = 20
        MAX_STEP_NUM = 100

        xmin, xmax = 0, 1
        ymin, ymax = 0, 1
        x_num = 80

        x_step, x_half_step, xy = viab.generate_2Dgrid(xmin, xmax, x_num, ymin, ymax)

        viab.x_step=x_step

        state = np.zeros(xy.shape[:-1])

        start_time = time.time()

        viab.topology_classification(xy, state, default_run, [management1_run, management2_run], is_sunny)
        time_diff = time.time() - start_time
        print(time_diff)

        viab.plot_points(xy, state)
        moddef.plotPhaseSpace(boundaries, style=styleDefault)
        mod1.plotPhaseSpace(boundaries, style=styleMod1)

        mod2.plotPhaseSpace(boundaries, style=styleMod1)

        if save:
                # filename = globalprefix+prefix+suffix+".svg"
                # filename = globalprefix+prefix+suffix+".eps"
                filename = globalprefix+prefix+suffix+".pdf"
                savePlot(filename)

        if not notshow:
                plt.show()

