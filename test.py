
from __future__ import division, print_function


import PyViability as viab
import PlantModel as pm
import TechChangeModel as tcm
import PopulationAndResourceModel as prm
#import GravityPendulumModel as gpm
import PTopologyL as topo

import sys, os, time

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np

def patchit(*traj, **kwargs):
        ax.add_patch(patch.Polygon(np.transpose(np.hstack(traj)) , facecolor = kwargs["color"], **topo.stylePatch))

if __name__ == "__main__":


    args = sys.argv[1:]

    if "plants" in args:
        # test plant model
        xmin, xmax = 0, 1
##         boundaries = [1e-3, 1e-3, 1, 1]
        boundaries = [xmin, xmin, xmax, xmax]
        #boundaries = [0.49, 0.130, 0.52, 0.155]
        xmin, ymin, xmax, ymax = boundaries
        # default values
        a = 0.2 # harvest value
        pm.xplusy = 0.65 # min sum of both for being desirable state
        prod = 2 # factor of productivity for y

        fig = plt.figure(figsize=(15, 15), tight_layout = True)
        ax = fig.add_subplot(111)
        b3 = np.array([[0, pm.xplusy, 0], [pm.xplusy, 0, 0]])
        patchit(b3, color = topo.cDarkUp)

        moddef = pm.PlantXY( prod = prod, ax = a, ay = a , comment = "default")
        mod1 = pm.PlantXY( prod = prod, ax = a/2, ay = a/2, comment = "management 1" )
        mod2 = pm.PlantXY( prod = prod, ax = 2*a, ay = 0, comment = "management 2" )

        moddef.plotPhaseSpace(boundaries, topo.styleDefault)
##         mod1.plotPhaseSpace(boundaries, topo.styleMod1)
        mod2.plotPhaseSpace(boundaries, topo.styleMod2)

        timestep = 1

        x_num = 80
        x_len = xmax - xmin
        x_step = x_len / x_num
        viab.x_step=x_step
        x_half_step = x_step / 2
        x = np.linspace(xmin ,xmax, x_num + 1)
        x = (x[:-1] + x[1:]) / 2
        y = np.linspace(xmin, xmax, x_num + 1)
        y = (y[:-1] + y[1:]) / 2
        xy = np.asarray(np.meshgrid(x, y))
        del x, y
        xy = np.rollaxis(xy, 0, 3)
        state = np.zeros(xy.shape[:-1])

        step_size = 2 * x_step

        default_run = viab.make_run_function(moddef._rhs_fast, moddef._odeint_params, step_size)
        management1_run = viab.make_run_function(mod1._rhs_fast, mod1._odeint_params, step_size)
        management2_run = viab.make_run_function(mod2._rhs_fast, mod2._odeint_params, step_size)
  ##      default_run = viab.make_run_function2(moddef, timestep)
  ##      management1_run = viab.make_run_function2(mod1, timestep)
  ##      management2_run = viab.make_run_function2(mod2, timestep)

        default_evols_list = [default_run]


        states = state
        coordinates = xy
        start_time = time.time()
        viab.topology_classification(xy, state, default_run, [management1_run, management2_run], pm.is_sunny)
##         states[pm.is_sunny(coordinates)] = 1  # initial state for shelter calculation
        # coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions
##         viab.viability_kernel(coordinates, states, [1], 0, 1, 1, default_evols_list)
        time_diff = time.time() - start_time
        print(time_diff)
        viab.plot_points(xy, state)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.gca().set_aspect('equal')
##         plt.gca().set_aspect('equal', adjustable='box')

    elif "techChange" in args:
        boundaries = [0, 0, 1, 2]
        xmin, ymin, xmax, ymax = boundaries

        fig2 = plt.figure(figsize=(15, 15))
        ax = fig2.add_subplot(111)

        # default values
        tcm.rvar = 1
        tcm.pBmin = 0.15
        tcm.pE = 0.3
        tcm.sigmaA = 1.05
        tcm.uDedge = 0.325
        tcm.pA = 0.5
        tcm.delta = 0.025
        tcm.smax = 0.3
        tcm.sBmax = 0.5



        # different instances of the model
        moddefTC = tcm.TechChangeXY('default')
        # mod1TC = tcm.TechChangeXY('one')
        mod2TC = tcm.TechChangeXY('two')

        plt.xlabel("$u_B$")
        plt.ylabel("$p_B$")
        plt.axis([xmin, xmax, ymin, ymax])

        viab.MAX_STEP_NUM = 4
        x_num = 80

        x_step, x_half_step, xy = viab.generate_2Dgrid(xmin, xmax, x_num, ymin, ymax)

        viab.x_step = x_step

        step_size = 2*x_step
        defaultTC_run = viab.make_run_function(moddefTC._rhs_fast, moddefTC._odeint_params, step_size)
        # management1TC_run = viab.make_run_function2(mod1TC, 1)
        management2TC_run = viab.make_run_function(mod2TC._rhs_fast, mod2TC._odeint_params, step_size)

        state = np.zeros(xy.shape[:-1])

        viab.topology_classification(xy, state, defaultTC_run, management2TC_run, tcm.is_sunnyTC)

        moddefTC.plotPhaseSpace(boundaries, topo.styleDefault)
        mod2TC.plotPhaseSpace(boundaries, topo.styleMod1)


        viab.plot_points(xy, state)
        # plt.ylim([0, 2])
        # plt.xlim([0, 1])
    elif "PuR_Plot_a" in args:

        boundaries = [0, 0, 35000, 18000]
        xmin, ymin, xmax, ymax = boundaries

        fig2 = plt.figure(figsize=(15, 15))
        ax = fig2.add_subplot(111)

        # default values for sunny region
        prm.xMinimal = 1000
        prm.yMinimal = 3000

        # generating grid and step size values
        viab.MAX_STEP_NUM = 4
        x_num = 80
        x_step, x_half_step, xy = viab.generate_2Dgrid(xmin, xmax, x_num, ymin, ymax)
        viab.x_step = x_step
        step_size = 2 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 4 * 10 ** (-6), delta = -0.1, kappa = 12000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 2.8 * 10 ** (-6), delta = -0.1, kappa = 12000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, step_size)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, step_size)

        default_evols_list = [defaultPuR_run]
        states = np.zeros(xy.shape[:-1])

        start_time = time.time()
        viab.topology_classification(xy, states, defaultPuR_run, management1PuR_run, prm.is_sunnyPuR)

        time_diff = time.time() - start_time
        print(time_diff)

        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
       # plt.gca().set_aspect('equal')
    ##         plt.gca().set_aspect('equal', adjustable='box')

    elif "PuR_Plot_b" in args:

        boundaries = [0, 0, 9000, 9000]
        xmin, ymin, xmax, ymax = boundaries

        fig3 = plt.figure(figsize=(15, 15))
        ax = fig3.add_subplot(111)

        # default values for sunny region
        prm.xMinimal = 1200
        prm.yMinimal = 2000

        # generating grid and step size values
        viab.MAX_STEP_NUM = 4
        x_num = 80
        x_step, x_half_step, xy = viab.generate_2Dgrid(xmin, xmax, x_num, ymin, ymax)
        viab.x_step = x_step
        step_size = 2 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 13.6 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, step_size)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, step_size)

        default_evols_list = [defaultPuR_run]
        states = np.zeros(xy.shape[:-1])

        start_time = time.time()
        viab.topology_classification(xy, states, defaultPuR_run, management1PuR_run, prm.is_sunnyPuR)

        time_diff = time.time() - start_time
        print(time_diff)

        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        # plt.gca().set_aspect('equal')
        ##         plt.gca().set_aspect('equal', adjustable='box')

    elif "PuR_Plot_c" in args:

        boundaries = [0, 0, 9000, 9000]
        xmin, ymin, xmax, ymax = boundaries

        fig4 = plt.figure(figsize=(15, 15))
        ax = fig4.add_subplot(111)

        # default values for sunny region
        prm.xMinimal = 4000
        prm.yMinimal = 3000

        # generating grid and step size values
        viab.MAX_STEP_NUM = 4
        x_num = 80
        x_step, x_half_step, xy = viab.generate_2Dgrid(xmin, xmax, x_num, ymin, ymax)
        viab.x_step = x_step
        step_size = 2 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 16 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, step_size)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, step_size)

        default_evols_list = [defaultPuR_run]
        states = np.zeros(xy.shape[:-1])

        start_time = time.time()
        viab.topology_classification(xy, states, defaultPuR_run, management1PuR_run, prm.is_sunnyPuR)

        time_diff = time.time() - start_time
        print(time_diff)

        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        # plt.gca().set_aspect('equal')
        ##         plt.gca().set_aspect('equal', adjustable='box')

    elif "PuR_Plot_d" in args:

        boundaries = [0, 0, 9000, 9000]
        xmin, ymin, xmax, ymax = boundaries


        fig4 = plt.figure(figsize=(15, 15))
        ax = fig4.add_subplot(111)

        # default values for sunny region
        prm.xMinimal = 4000
        prm.yMinimal = 3000

        # generating grid and step size values
        viab.MAX_STEP_NUM = 4
        x_num = 80
        x_step, x_half_step, xy = viab.generate_2Dgrid(xmin, xmax, x_num, ymin, ymax)
        viab.x_step = x_step
        step_size = 2 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 11.2 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, step_size)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, step_size)

        default_evols_list = [defaultPuR_run]
        states = np.zeros(xy.shape[:-1])

        start_time = time.time()
        viab.topology_classification(xy, states, defaultPuR_run, management1PuR_run, prm.is_sunnyPuR)

        time_diff = time.time() - start_time
        print(time_diff)

        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        # plt.gca().set_aspect('equal')
        ##         plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

