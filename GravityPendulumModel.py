

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

import sys, os.path

from myPhaseSpaceL import plotPhaseSpace, savePlot

from SODEoneL import SODEone

from PTopologyL import *

## global ax, ay, prod

pi = np.pi

stylePoint["markersize"] *= 2


def gravity_rhs(theta_omega, t, a):
    theta, omega = theta_omega
    dtheta = omega
    domega = -np.sin(theta) - a
    return np.array([dtheta, domega])

class GravPend(BaseODEs.BaseODEs):
    def __init__(self, comment="", **params):
        assert set(["a"]).issubset(params)
        BaseODEs.BaseODEs.__init__(self, gravity_rhs, params, comment=comment)

    def plotPhaseSpace(self, boundaries, style, alpha=None):
        plotPhaseSpace(self.rhs_PS, boundaries, colorbar=False, style=style, alpha=alpha)


class GravPend2(SODEone):

    """\
Implementation of tech change model
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
    def evol_noManagement(self, theta_omega, t=0):
        theta, omega = theta_omega
        dtheta = omega
        domega = np.sin(theta)*(-1)
        return np.array([domega, dtheta])

    def evol_Management1(self, theta_omega, t=0):
        theta, omega = theta_omega
        dtheta = omega
        domega = -np.sin(theta)-a
        return np.array([domega, dtheta])


def patchit(*traj, **kwargs):
        ax.add_patch(patch.Polygon(np.transpose(np.hstack(traj)) , facecolor = kwargs["color"], **stylePatch))

def pointIt(xy, color):
        plt.plot([xy[0]],[xy[1]], marker = "8", color = color, **stylePoint)


def is_sunnyGPM(p):
    """sunny constraint for gravity Pendulum"""
    return np.abs(p[:, 1])<l

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

        boundaries = [0, -2, 8, 1]
        #boundaries = [0.49, 0.130, 0.52, 0.155]
        xmin, ymin, xmax, ymax = boundaries

        fig2 = plt.figure(figsize=(15, 15))
        ax = fig2.add_subplot(111)

        # default values
        a=0.6
        l=0.5
        #globalprefix += "_a=%1.2f_prod=%1.1f_desire=%1.2f_"%(rvar, pBmin, pE)

        # different instances of the model
        moddefTC = GravPend('no')
        mod1TC = GravPend('one')


        plt.xlabel("$\Theta$")
        plt.ylabel("$\omega$")
        plt.axis([xmin, xmax, ymin, ymax])


        #######################
        ## viability stuff starts here
        #######################

        defaultTC_run = viab.make_run_function2(moddefTC, 1)
        management1TC_run = viab.make_run_function2(mod1TC, 1)

        viab.MAX_STEP_NUM = 4
        x_num = 80
        x_len = xmax - xmin
        y_len = ymax - ymin
        x_step = max(x_len,y_len) / x_num
        viab.x_step = x_step
        viab.STEPSIZE = 2 * x_step
        #viab.timestep=1
        #viab.ADD_NEIGHBORS_ONLY_ONCE= True #3 421 997

        #viab.ADD_NEIGHBORS_WITH_MASK = True #48 933 119

        # 46 946 543

        x_half_step = x_step/2
        x = np.linspace(xmin,xmax, x_num + 1)
        x = (x[:-1] + x[1:]) / 2
        y = np.linspace(ymin,ymax, x_num + 1)
        y = (y[:-1] + y[1:]) / 2
        xy = np.asarray(np.meshgrid(x, y))
        print(defaultTC_run)
        del x, y
        xy = np.rollaxis(xy, 0, 3)
        state = np.zeros(xy.shape[:-1])
        np.set_printoptions(threshold=np.nan)
        viab.topology_classification(xy, state, defaultTC_run, management1TC_run, is_sunnyTC)

        moddefTC.plotPhaseSpace(boundaries, style=styleDefault)
        mod1TC.plotPhaseSpace(boundaries, style = styleMod1)


        viab.plot_points(xy, state)
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])

        if save:
            # filename = globalprefix+prefix+suffix+".svg"
            # filename = globalprefix+prefix+suffix+".eps"
            filename = globalprefix+prefix+suffix+".pdf"
            savePlot(filename)

        if not notshow:
            plt.show()
