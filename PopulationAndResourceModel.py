
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


def PopAndRes_rhs(xy, t, phi, r, gamma, delta, kappa):
    x, y = xy
    dx = delta * x + phi * gamma * x * y
    dy = r * y * (1 - y / kappa) - gamma * x * y
    return np.array([dx, dy])


class PopAndRes(BaseODEs.BaseODEs):

    """\
Implementation of population and resource dynamics model
"""

    def __init__(self, comment="", **params):
        assert set(["delta", "phi", "gamma", "kappa"]).issubset(params)
        BaseODEs.BaseODEs.__init__(self, PopAndRes_rhs, params, comment=comment)

    def plotPhaseSpace(self, boundaries, style, alpha=None):
        plotPhaseSpace(self.rhs_PS, boundaries, colorbar=False, style=style, alpha=alpha)


class PopAndRes2(SODEone):

    """\
Implementation of population and resource dynamics model
"""

    def __init__(self, management):
        if management == "no":
            self.evol = self.evol_noManagement
        elif management == "one":
            self.evol = self.evol_Management1
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
    def evol_noManagement(self, xy, t=0):
        x, y = xy
        dx = delta * x + phi * gamma0 * x * y
        dy = r * y * (1 - y / kappa) - gamma0 * x * y
        return np.array([dx, dy])

    def evol_Management1(self, xy, t=0):
        x, y = xy
        dx = delta * x + phi * gamma1 * x * y
        dy = r * y * (1 - y / kappa) - gamma1 * x * y
        return np.array([dx, dy])


def patchit(*traj, **kwargs):
        ax.add_patch(patch.Polygon(np.transpose(np.hstack(traj)) , facecolor = kwargs["color"], **stylePatch))

def pointIt(xy, color):
        plt.plot([xy[0]],[xy[1]], marker = "8", color = color, **stylePoint)


def is_sunnyPuR(xy):
    """sunny constraint for techChangeModel"""
    return (xy[:, 0]>xMinimal).__and__(xy[:, 1]>yMinimal)

if __name__ == "__main__":

        args = sys.argv[1:]

        save = "save" in args
        notshow = "notshow" in args
        plota = "plota" in args
        plotb = "plotb" in args
        plotc = "plotc" in args
        plotd = "plotd" in args

        plotb=True
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
        if plota:
            #boundaries = [0, 0, 9000, 9000]
            boundaries = [0, 0, 35000, 18000]
            xmin, ymin, xmax, ymax = boundaries

            fig2 = plt.figure(figsize=(15, 15))
            ax = fig2.add_subplot(111)

            # default values plot 1
            xMinimal=1000
            yMinimal=3000
            phi=4
            r=0.04
            gamma0=4*10**(-6)
            gamma1=2.8*10**(-6)
            delta=-0.1
            kappa=12000

            # different instances of the model
            moddefPuR = ProdAndRes('no')
            mod1PuR = ProdAndRes('one')

            defaultPuR_run = viab.make_run_function2(moddefPuR, 10)
            management1PuR_run = viab.make_run_function2(mod1PuR, 10)

        elif plotb:
            boundaries = [0, 0, 9000, 9000]
            #boundaries = [0, 0, 35000, 18000]
            xmin, ymin, xmax, ymax = boundaries

            fig3 = plt.figure(figsize=(15, 15))
            ax = fig3.add_subplot(111)

            #default values plot 2
            xMinimal = 1200
            yMinimal = 2000
            phi = 4
            r = 0.04
            gamma0 = 8 * 10 ** (-6)
            gamma1 = 13.6 * 10 ** (-6)
            delta = -0.15
            kappa = 6000

            # different instances of the model
            moddefPuR = ProdAndRes('no')
            mod1PuR = ProdAndRes('one')

            defaultPuR_run = viab.make_run_function2(moddefPuR, 1)
            management1PuR_run = viab.make_run_function2(mod1PuR, 1)

        elif plotc:
            boundaries = [0, 0, 9000, 9000]
            xmin, ymin, xmax, ymax = boundaries
            # default values plot 3
            xMinimal = 4000
            yMinimal = 3000
            phi = 4
            r = 0.04
            gamma0 = 8 * 10 ** (-6)
            gamma1 = 16* 10 ** (-6)
            delta = -0.15
            kappa = 6000

            fig4 = plt.figure(figsize=(15, 15))
            ax = fig4.add_subplot(111)

            # different instances of the model
            moddefPuR = PopAndRes2('no')
            mod1PuR = PopAndRes2('one')

            defaultPuR_run = viab.make_run_function2(moddefPuR, 1)
            management1PuR_run = viab.make_run_function2(mod1PuR, 1)

        elif plotd:
            boundaries = [0, 0, 9000, 9000]
            xmin, ymin, xmax, ymax = boundaries
            # default values plot 3
            xMinimal = 4000
            yMinimal = 3000
            phi = 4
            r = 0.04
            gamma0 = 8 * 10 ** (-6)
            gamma1 = 11.2 * 10 ** (-6)
            delta = -0.15
            kappa = 6000

            fig4 = plt.figure(figsize=(15, 15))
            ax = fig4.add_subplot(111)

            # different instances of the model
            moddefPuR = ProdAndRes('no')
            mod1PuR = ProdAndRes('one')

            defaultPuR_run = viab.make_run_function2(moddefPuR, 1)
            management1PuR_run = viab.make_run_function2(mod1PuR, 1)

        else:
            print('"plota","plotb", "plotc" or "plotd" as script argument required. "plota" is evaluated by default')
            plota=True
            boundaries = [0, 0, 35000, 18000]
            xmin, ymin, xmax, ymax = boundaries

            fig2 = plt.figure(figsize=(15, 15))
            ax = fig2.add_subplot(111)

            # default values plot 1
            xMinimal = 1000
            yMinimal = 3000
            phi = 4
            r = 0.04
            gamma0 = 4 * 10 ** (-6)
            gamma1 = 2.8 * 10 ** (-6)
            delta = -0.1
            kappa = 12000

            # different instances of the model
            moddefPuR = PopAndRes2('no')
            mod1PuR = PopAndRes2('one')

            defaultPuR_run = viab.make_run_function2(moddefPuR, 10)
            management1PuR_run = viab.make_run_function2(mod1PuR, 10)


        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.axis([xmin, xmax, ymin, ymax])


        viab.MAX_STEP_NUM = 2
        x_num = 80
        y_num = 60
        x_len = xmax - xmin
        y_len = ymax - ymin
        x_step = max(x_len/x_num,y_len/y_num)
        viab.x_step = x_step
        timestep=1
        viab.timestep=timestep

        if plota:
            viab.timestep=timestep*10
           # viab.x_step = x_step*10

        x_half_step = x_step/2
        x = np.linspace(xmin,xmax, x_num + 1)
        x = (x[:-1] + x[1:]) / 2
        y = np.linspace(ymin,ymax, y_num + 1)
        y = (y[:-1] + y[1:]) / 2
        xy = np.asarray(np.meshgrid(x, y))
        del x, y
        xy = np.rollaxis(xy, 0, 3)
        state = np.zeros(xy.shape[:-1])
        np.set_printoptions(threshold=np.nan)

        viab.topology_classification(xy, state, defaultPuR_run, management1PuR_run, is_sunnyPuR)

        viab.plot_points(xy, state)

        moddefPuR.plotPhaseSpace(boundaries, style=styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, style=styleMod1)

        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])


        if save:
            # filename = globalprefix+prefix+suffix+".svg"
            # filename = globalprefix+prefix+suffix+".eps"
            filename = globalprefix+prefix+suffix+".pdf"
            savePlot(filename)

        if not notshow:
            plt.show()
