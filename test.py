
from __future__ import division, print_function


import PyViability as viab
import PlantModel as pm
import TechChangeModel as tcm
import PopulationAndResourceModel as prm
import GravityPendulumModel as gpm
import PTopologyL as topo

import myPhaseSpaceL as mPS
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np

from scipy.integrate import odeint

def patchit(*traj, **kwargs):
        ax.add_patch(patch.Polygon(np.transpose(np.hstack(traj)) , facecolor = kwargs["color"], **topo.stylePatch))

if __name__ == "__main__":


    args = sys.argv[1:]

    if "colortest" in args:
        fig = plt.figure(figsize=(15, 15), tight_layout = True)
        ax = fig.add_subplot(111)
        xmin, xmax = 0, 1

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
        state[:] = 1

        viab.plot_areas(xy, state)

    if "plants" in args:
        # test plant model

        # boundaries of PhaseSpace
        xmin, xmax = 0, 1
        boundaries = [xmin, xmin, xmax, xmax]
        xmin, ymin, xmax, ymax = boundaries

        # default values
        a = 0.2 # harvest value
        pm.xplusy = 0.65 # min sum of both for being desirable state
        prod = 2 # factor of productivity for y

        # plot preparation
        fig = plt.figure(figsize=(15, 15), tight_layout=True)
        ax = fig.add_subplot(111)
        b3 = np.array([[0, pm.xplusy, 0], [pm.xplusy, 0, 0]])
        patchit(b3, color=topo.cDarkUp)

        # steady_states_def = moddef.steadyState()
        # steady_states1= mod1.steadyState()
        # steady_states2 = mod2.steadyState()

        # normalized grid
        xy, scalingfactor, offset, x_step = viab.normalized_grid(boundaries, 80)

        # states for grid
        state = np.zeros(xy.shape[:-1])

        # Some initial states for 80*80 grid to avoid runtime warnings
        init_states = [(4880, -1), (4960, -1), (5040, -1), (61, -5), (62, -5), (63, -5), (0, -4), (1, -4),
                       (40, -4), (41, -4), (42, -4), (43, -4)]
        for i in range(len(init_states)):
            state[init_states[i][0]] = init_states[i][1]

        # Integration length for odeint
        viab.x_step = x_step
        x_half_step = x_step / 2
        viab.STEPSIZE = 1.5 * x_step

        # different instances of the model
        moddef = pm.PlantXY(prod=prod, ax=a, ay=a, comment="default")
        mod1 = pm.PlantXY(prod=prod, ax=a / 2, ay=a / 2, comment="management 1")
        mod2 = pm.PlantXY(prod=prod, ax=2 * a, ay=0, comment="management 2")

        default_run = viab.make_run_function(moddef._rhs_fast, moddef._odeint_params, offset, scalingfactor)
        management1_run = viab.make_run_function(mod1._rhs_fast, mod1._odeint_params, offset, scalingfactor)
        management2_run = viab.make_run_function(mod2._rhs_fast, mod2._odeint_params, offset, scalingfactor)

        default_evols_list = [default_run]

        # scaled sunny function
        sunny = viab.scaled_to_one_sunny(pm.is_sunny, offset, scalingfactor)

        # topology classification via viability algorithm
        start_time = time.time()

        viab.topology_classification(xy, state, [default_run], [management1_run, management2_run], sunny)

        time_diff = time.time() - start_time
        print(time_diff)

        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, state)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        moddef.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1.plotPhaseSpace(boundaries, topo.styleMod1)
        mod2.plotPhaseSpace(boundaries, topo.styleMod2)

        fig = plt.figure(figsize=(15, 15), tight_layout = True)
        viab.plot_areas(xy, state)
        moddef.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1.plotPhaseSpace(boundaries, topo.styleMod1)
        mod2.plotPhaseSpace(boundaries, topo.styleMod2)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.gca().set_aspect('equal', adjustable='box')

    if "techChange" in args:
        # boundaries of PhaseSpace
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

        # normalized grid
        xy, scalingfactor, offset, x_step = viab.normalized_grid(boundaries, 80)

        # Integration length for odeint
        viab.x_step = x_step
        viab.STEPSIZE = 1.5 * x_step

        # different instances of the model
        moddefTC = tcm.TechChangeXY('default')
        mod2TC = tcm.TechChangeXY('two')

        defaultTC_run = viab.make_run_function(moddefTC._rhs_fast, moddefTC._odeint_params, offset, scalingfactor)
        management2TC_run = viab.make_run_function(mod2TC._rhs_fast, mod2TC._odeint_params, offset, scalingfactor)

        # # Example: Plotting the scaled right-hand-side
        # defaultTC_rhs_test = viab.make_run_function(moddefTC.rhs_PS, moddefTC._odeint_params, offset, scalingfactor, returning = "PS_plt_scaled_rhs")
        # mPS.plotPhaseSpace(defaultTC_rhs_test, [0, 0, 1, 1], colorbar=False, style=topo.styleDefault)
        #
        # mod2TC_rhs_test = viab.make_run_function(mod2TC.rhs_PS, mod2TC._odeint_params, offset, scalingfactor,
        #                                             returning="PS_plt_scaled_rhs")
        # mPS.plotPhaseSpace(mod2TC_rhs_test, [0, 0, 1, 1], colorbar=False, style=topo.styleMod1)

        # Generating states for grid points
        state = np.zeros(xy.shape[:-1])

        # Some initial states for 80*80 grid to avoid runtime warnings
        init_states = [(79*7, -1), (79*8, -1), (79*9, -1), (559, -1), (639
                                                                       , -1)]
        for i in range(len(init_states)):
            state[init_states[i][0]] = init_states[i][1]

        # scaled sunny-function
        sunny = viab.scaled_to_one_sunny(tcm.is_sunnyTC, offset, scalingfactor)

        start_time = time.time()

        # topology classification via viability algorithm
        viab.topology_classification(xy, state, [defaultTC_run], [management2TC_run], sunny)

        time_diff = time.time() - start_time
        print(time_diff)

        # backscaling grid
        # xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # Plotting:
        viab.plot_points(xy, state)

        # moddefTC.plotPhaseSpace(boundaries, topo.styleDefault)
        # mod2TC.plotPhaseSpace(boundaries, topo.styleMod1)
#
        # plt.xlim([xmin, xmax])
        # plt.ylim([ymin, ymax])

        plt.xlabel("$u_B$")
        plt.ylabel("$p_B$")

    if "PuR_Plot_a" in args:

        boundaries = [0, 0, 35000, 18000]
        xmin, ymin, xmax, ymax = boundaries

        fig2 = plt.figure(figsize=(15, 15))
        ax = fig2.add_subplot(111)

        # default values for sunny region
        prm.xMinimal = 1000
        prm.yMinimal = 3000

        # generating grid and step size values
        xy, scalingfactor, offset, x_step = viab.normalized_grid(boundaries, 80)
        viab.x_step = x_step
        viab.STEPSIZE = 1.5 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 4 * 10 ** (-6), delta = -0.1, kappa = 12000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 2.8 * 10 ** (-6), delta = -0.1, kappa = 12000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, offset, scalingfactor)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, offset, scalingfactor)

        default_evols_list = [defaultPuR_run]

        # set states and scale sunny region
        states = np.zeros(xy.shape[:-1])
        sunny = viab.scaled_to_one_sunny(prm.is_sunnyPuR, offset, scalingfactor)

        # Viability calculation
        start_time = time.time()

        viab.topology_classification(xy, states, [defaultPuR_run], [management1PuR_run], sunny)

        time_diff = time.time() - start_time
        print(time_diff)

        # plotting
        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    if "PuR_Plot_b" in args:

        boundaries = [0, 0, 9000, 9000]
        xmin, ymin, xmax, ymax = boundaries

        # default values for sunny region
        prm.xMinimal = 1200
        prm.yMinimal = 2000

        # generating grid and step size values
        xy, scalingfactor, offset, x_step = viab.normalized_grid(boundaries, 80)
        viab.x_step = x_step
        viab.STEPSIZE = 1.5 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 13.6 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, offset, scalingfactor)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, offset, scalingfactor)

        default_evols_list = [defaultPuR_run]

        # Some initial states for 80*80 grid to avoid runtime warnings
        states = np.zeros(xy.shape[:-1])
        init_states = [(1934, -6), (3289, -7)]
        for i in range(len(init_states)):
            states[init_states[i][0]] = init_states[i][1]

        # scaled sunny function
        sunny = viab.scaled_to_one_sunny(prm.is_sunnyPuR, offset, scalingfactor)

        # viability calculation
        start_time = time.time()

        viab.topology_classification(xy, states, [defaultPuR_run], [management1PuR_run], sunny)

        time_diff = time.time() - start_time
        print(time_diff)

        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    if "PuR_Plot_c" in args:

        boundaries = [0, 0, 9000, 9000]
        xmin, ymin, xmax, ymax = boundaries

        # default values for sunny region
        prm.xMinimal = 4000
        prm.yMinimal = 3000

        # generating grid and step size values
        xy, scalingfactor, offset, x_step = viab.normalized_grid(boundaries, 80)
        viab.x_step = x_step
        viab.STEPSIZE = 1.5 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 16 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, offset, scalingfactor, remember = True)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, offset, scalingfactor, remember = True)

        default_evols_list = [defaultPuR_run]

        # Some initial states for 80*80 grid to avoid runtime warnings
        states = np.zeros(xy.shape[:-1])

        init_states = [(1613, -10), (3289, -10)]
        for i in range(len(init_states)):
           states[init_states[i][0]] = init_states[i][1]

        # scaled sunny function
        sunny = viab.scaled_to_one_sunny(prm.is_sunnyPuR, offset, scalingfactor)

        # viability calculation
        start_time = time.time()

        viab.topology_classification(xy, states, [defaultPuR_run], [management1PuR_run], sunny)

        time_diff = time.time() - start_time
        print(time_diff)

        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    if "PuR_Plot_d" in args:

        boundaries = [0, 0, 9000, 9000]
        xmin, ymin, xmax, ymax = boundaries

        # default values for sunny region
        prm.xMinimal = 4000
        prm.yMinimal = 3000

        # generating grid and step size values
        xy, scalingfactor, offset, x_step = viab.normalized_grid(boundaries, 80)
        viab.x_step = x_step
        viab.STEPSIZE = 1.5 * x_step

        # different instances of the model
        moddefPuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="default")
        mod1PuR = prm.PopAndRes(phi = 4, r = 0.04, gamma = 11.2 * 10 ** (-6), delta = -0.15, kappa = 6000, comment="management 1")

        defaultPuR_run = viab.make_run_function(moddefPuR._rhs_fast, moddefPuR._odeint_params, offset, scalingfactor)
        management1PuR_run = viab.make_run_function(mod1PuR._rhs_fast, mod1PuR._odeint_params, offset, scalingfactor)

        default_evols_list = [defaultPuR_run]

        # Some initial states for 80*80 grid to avoid runtime warnings
        states = np.zeros(xy.shape[:-1])

        init_states = [(2334, -10), (3289, -10)]
        for i in range(len(init_states)):
            states[init_states[i][0]] = init_states[i][1]

        # scaled sunny function
        sunny = viab.scaled_to_one_sunny(prm.is_sunnyPuR, offset, scalingfactor)

        # viability calculation
        start_time = time.time()

        viab.topology_classification(xy, states, [defaultPuR_run], [management1PuR_run], sunny)

        time_diff = time.time() - start_time
        print(time_diff)

        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, states)

        moddefPuR.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1PuR.plotPhaseSpace(boundaries, topo.styleMod1)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    if "pendulum" in args:
        # test gravity pendulum
        xmin, xmax = 0, 2 * np.pi
        ymin, ymax = -2.2, 1.2
        boundaries = [xmin, ymin, xmax, ymax]

        # default values
        a = 0.6
        gpm.l = 0.5

        # generating grid and step size values
        xy, scalingfactor,  offset, x_step = viab.normalized_grid(boundaries, 80)
        viab.x_step = x_step
        viab.STEPSIZE = 1.5 * x_step

        # different instances of the model
        moddef = gpm.GravPend(a=0, comment="default")
        mod1 = gpm.GravPend(a=a, comment="management 1")

        default_run = viab.make_run_function(moddef._rhs_fast, moddef._odeint_params, offset, scalingfactor)
        management1_run = viab.make_run_function(mod1._rhs_fast, mod1._odeint_params, offset, scalingfactor)

        # states before calculation and scaled sunny function
        state = np.zeros(xy.shape[:-1])
        sunny = viab.scaled_to_one_sunny(gpm.is_sunnyGPM, offset, scalingfactor)

        # viability calculation
        start_time = time.time()

        viab.topology_classification(xy, state, [default_run], [management1_run], sunny, periodic_boundaries = np.array([1, -1]))

        time_diff = time.time() - start_time
        print(time_diff)

        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, state)

        moddef.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1.plotPhaseSpace(boundaries, topo.styleMod2)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])


        fig = plt.figure(figsize=(15, 15), tight_layout=True)

        viab.plot_areas(xy, state)

        moddef.plotPhaseSpace(boundaries, topo.styleDefault)
        mod1.plotPhaseSpace(boundaries, topo.styleMod2)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    if "pendulum-hex" in args:
        # test gravity pendulum
        xmin, xmax = 0, 2 * np.pi
        ymin, ymax = -2.2, 1.2
        boundaries = [[xmin, xmax], [ymin, ymax]]
        PSboundaries = [xmin, ymin, xmax, ymax]

        # default values
        a = 0.6
        gpm.l = 0.5

        # generating grid and step size values
        xy, scalingfactor,  offset, x_step = viab.hexGrid(boundaries, 80, verb = True)

        # states before calculation and scaled sunny function
        state = np.zeros(xy.shape[:-1])
        sunny = viab.scaled_to_one_sunny(gpm.is_sunnyGPM, offset, scalingfactor)

        # different instances of the model
        moddef = gpm.GravPend(a=0, comment="default")
        mod1 = gpm.GravPend(a=a, comment="management 1")

        default_run = viab.make_run_function(moddef._rhs_fast, moddef._odeint_params, offset, scalingfactor)
        default_PS = viab.make_run_function(moddef._rhs_PS, moddef._odeint_params, offset, scalingfactor, returning = "PS")
        management1_run = viab.make_run_function(mod1._rhs_fast, mod1._odeint_params, offset, scalingfactor)
        management1_PS = viab.make_run_function(mod1._rhs_PS, mod1._odeint_params, offset, scalingfactor, returning = "PS")

        # create the figure already so it can be used for the verbosity plots
        fig = plt.figure(figsize=(15, 15), tight_layout=True)
        # viability calculation
        start_time = time.time()

        viab.topology_classification(xy, state, [default_run], [management1_run], sunny, periodic_boundaries = np.array([1, -1]))

        time_diff = time.time() - start_time
        print(time_diff)

        # backscaling
        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, state)
        # mPS.plotPhaseSpace(default_PS, [0, 0, 1, 1], style = topo.styleDefault, colorbar = False)
        # mPS.plotPhaseSpace(management1_PS, [0, 0, 1, 1], style = topo.styleMod1, colorbar = False)
        moddef.plotPhaseSpace(PSboundaries, topo.styleDefault)
        mod1.plotPhaseSpace(PSboundaries, topo.styleMod2)
        # plt.axes().set_aspect("equal")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        fig = plt.figure(figsize=(15, 15), tight_layout=True)
        viab.plot_areas(xy, state)
        # plt.axes().set_aspect("equal")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    if "eddies" in args:
        # test ediies calculation
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        boundaries = [[xmin, xmax], [ymin, ymax]]
        PSboundaries = [xmin, ymin, xmax, ymax]

        def rhs(xy, t):
            x, y = xy
            return [-y, x]
        def rhsPS(xy, t):
            print(xy.shape)
            ret = np.zeros_like(xy)
            ret[0] = - xy[1]
            ret[1] =   xy[0]
            return ret
        def sunny(xy):
            return xy[:,0] > 0


        # generating grid and step size values
        # xy, scalingfactor,  offset, x_step = viab.hexGrid(boundaries, 40, verb = True)
        # viab.STEPSIZE = 1 * x_step
        xy, scalingfactor,  offset, x_step = viab.normalized_grid(PSboundaries, 40)
        viab.STEPSIZE = 1.5 * x_step

        # states before calculation and scaled sunny function
        state = np.zeros(xy.shape[:-1])
        sunny = viab.scaled_to_one_sunny(sunny, offset, scalingfactor)

        # create correctly scaled evolution functions
        default_run = viab.make_run_function(rhs, (), offset, scalingfactor, remember = True)
        default_PS = viab.make_run_function(rhsPS, (), offset, scalingfactor, returning = "PS")

        # create the figure already so it can be used for the verbosity plots
        fig = plt.figure(figsize=(15, 15), tight_layout=True)

        # viability calculation
        start_time = time.time()

        viab.topology_classification(xy, state, [default_run], [], sunny,
                                     compute_eddies = True)

        time_diff = time.time() - start_time
        print(time_diff)

        # backscaling
        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, state)
        mPS.plotPhaseSpace(rhsPS, PSboundaries, style = topo.styleDefault, colorbar = False)
        # mPS.plotPhaseSpace(default_PS, [0, 0, 1, 1], style = topo.styleDefault, colorbar = False)
        plt.axes().set_aspect("equal")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        fig = plt.figure(figsize=(15, 15), tight_layout=True)
        viab.plot_areas(xy, state)
        plt.axes().set_aspect("equal")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    if "eddies-hex" in args:
        # test ediies calculation
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        boundaries = [[xmin, xmax], [ymin, ymax]]
        PSboundaries = [xmin, ymin, xmax, ymax]

        def rhs(xy, t):
            x, y = xy
            return [-y, x]
        def rhsPS(xy, t):
            ret = np.zeros_like(xy)
            ret[0] = - xy[1]
            ret[1] =   xy[0]
            return ret
        def sunny(xy):
            return xy[:,0] > 0


        # generating grid and step size values
        xy, scalingfactor,  offset, x_step = viab.hexGrid(boundaries, 40, verb = True)
        viab.STEPSIZE = 1 * x_step
        # xy, scalingfactor,  offset, x_step = viab.normalized_grid(PSboundaries, 40)
        # viab.STEPSIZE = 1.5 * x_step

        # states before calculation and scaled sunny function
        state = np.zeros(xy.shape[:-1])
        sunny = viab.scaled_to_one_sunny(sunny, offset, scalingfactor)

        # create correctly scaled evolution functions
        default_run = viab.make_run_function(rhs, (), offset, scalingfactor, remember = True)
        default_PS = viab.make_run_function(rhsPS, (), offset, scalingfactor, returning = "PS")

        # create the figure already so it can be used for the verbosity plots
        fig = plt.figure(figsize=(15, 15), tight_layout=True)

        # viability calculation
        start_time = time.time()

        viab.topology_classification(xy, state, [default_run], [], sunny,
                                     compute_eddies = True)

        time_diff = time.time() - start_time
        print(time_diff)

        # backscaling
        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, state)
        mPS.plotPhaseSpace(rhsPS, PSboundaries, style = topo.styleDefault, colorbar = False)
        # mPS.plotPhaseSpace(default_PS, [0, 0, 1, 1], style = topo.styleDefault, colorbar = False)
        plt.axes().set_aspect("equal")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        fig = plt.figure(figsize=(15, 15), tight_layout=True)
        viab.plot_areas(xy, state)
        plt.axes().set_aspect("equal")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    plt.show()

