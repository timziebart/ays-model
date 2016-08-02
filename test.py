
from __future__ import division, print_function


import PyViability as viab
import helper
import PlantModel as pm
import TechChangeModel as tcm
import PopulationAndResourceModel as prm
import GravityPendulumModel as gpm
import ConsumptionModel as cm
import PTopologyL as topo

import myPhaseSpaceL as mPS

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import functools as ft
import numba as nb


def generate_example(default_rhss,
                     management_rhss,
                     sunny_fct,
                     boundaries,
                     default_parameters = [],
                     management_parameters = [],
                     n0=80,
                     grid_type="orthogonal",
                     periodicity=[],
                     backscaling=True,
                     plot_points=True,
                     plot_areas=False,
                     default_rhssPS = None,
                     management_rhssPS = None,
                     ):

    plotPS = lambda rhs, boundaries, style: mPS.plotPhaseSpace(rhs, [boundaries[0][0], boundaries[1][0], boundaries[0][1], boundaries[1][1]], colorbar=False, style=style)

    if not default_parameters:
        default_parameters = [{}] * len(default_rhss)
    if not management_parameters:
        management_parameters = [{}] * len(management_rhss)

    xlim, ylim = boundaries
    if default_rhssPS is None:
        default_rhssPS = default_rhss
    if management_rhssPS is None:
        management_rhssPS = management_rhss

    def example_function():
        grid, scaling_factor,  offset, _ = viab.generate_grid(boundaries,
                                                        n0,
                                                        grid_type,
                                                        periodicity = periodicity) #noqa
        states = np.zeros(grid.shape[:-1])

        default_runs = [viab.make_run_function(nb.jit(rhs), helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor) for rhs, parameters in zip(default_rhss, default_parameters)] #noqa
        management_runs = [viab.make_run_function(nb.jit(rhs), helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor) for rhs, parameters in zip(management_rhss, management_parameters)] #noqa

        sunny = viab.scaled_to_one_sunny(sunny_fct, offset, scaling_factor)

        # adding the figure here already if VERBOSE is set
        # this makes only sense, if backscaling is switched off
        if (not backscaling) and plot_points:
            fig = plt.figure(figsize=(15, 15), tight_layout=True)
            fig.suptitle('example: ' + example, fontsize=20)

        start_time = time.time()
        viab.topology_classification(grid, states, default_runs, management_runs, sunny, periodic_boundaries = periodicity)
        time_diff = time.time() - start_time

        print("run time: {!s} s".format(dt.timedelta(seconds=time_diff)))

        if backscaling:
            grid = viab.backscaling_grid(grid, scaling_factor, offset)

            if plot_points:
                fig = plt.figure(figsize=(15, 15), tight_layout=True)
                fig.suptitle('example: ' + example, fontsize=20)

                viab.plot_points(grid, states)

                [plotPS(ft.partial(rhs, **parameters), boundaries, topo.styleDefault) #noqa
                    for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
                [plotPS(ft.partial(rhs, **parameters), boundaries, style)
                    for rhs, parameters, style in zip(management_rhssPS, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.xlim(xlim)
                plt.ylim(ylim)


            if plot_areas:
                fig = plt.figure(figsize=(15, 15), tight_layout=True)
                fig.suptitle('example: ' + example, fontsize=20)

                viab.plot_areas(grid, states)

                [plotPS(ft.partial(rhs, **parameters), boundaries, topo.styleDefault) #noqa
                    for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
                [plotPS(ft.partial(rhs, **parameters), boundaries, style)
                    for rhs, parameters, style in zip(management_rhssPS, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.xlim(xlim)
                plt.ylim(ylim)

        else:
            plot_limits = [0,1]

            default_PSs = [viab.make_run_function(rhs, helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning="PS") #noqa
                            for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
            management_PSs = [viab.make_run_function(rhs, helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning="PS") #noqa
                                for rhs, parameters in zip(management_rhssPS, management_parameters)] #noqa

            if plot_points:
                # figure already created above
                fig.suptitle('example: ' + example, fontsize=20)

                viab.plot_points(grid, states)

                [plotPS(rhs, [plot_limits]*2, topo.styleDefault) for rhs, parameters in zip(default_PSs, default_parameters)]
                [plotPS(rhs, [plot_limits]*2, style) for rhs, parameters, style in zip(management_PSs, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.xlim(plot_limits)
                plt.ylim(plot_limits)


            if plot_areas:
                fig = plt.figure(figsize=(15, 15), tight_layout=True)
                fig.suptitle('example: ' + example, fontsize=20)

                viab.plot_areas(grid, states)

                [plotPS(rhs, [plot_limits]*2, topo.styleDefault) for rhs, parameters in zip(default_PSs, default_parameters)]
                [plotPS(rhs, [plot_limits]*2, style) for rhs, parameters, style in zip(management_PSs, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.xlim(plot_limits)
                plt.ylim(plot_limits)

    return example_function



EXAMPLES = {
            "pendulum":
                generate_example([gpm.pendulum_rhs],
                                 [gpm.pendulum_rhs],
                                 gpm.pendulum_sunny,
                                 [[0, 2*np.pi],[-2.2,1.2]],
                                 default_parameters=[{"a":0.0}],
                                 management_parameters=[{"a":0.6}],
                                 periodicity=[1, -1],
                                 backscaling=False,
                                 ),
            "pendulum-hex":
                generate_example([gpm.pendulum_rhs],  # hex-grid generation not yet done
                                 [gpm.pendulum_rhs],
                                 gpm.pendulum_sunny,
                                 [[0, 2*np.pi],[-2.2,1.2]],
                                 default_parameters=[{"a":0.0}],
                                 management_parameters=[{"a":0.6}],
                                 periodicity=[1, -1],
                                 grid_type="simplex-based",
                                 ),
            "plants":
                generate_example([pm.plants_rhs],
                                 [pm.plants_rhs]*2,
                                 pm.plants_sunny,
                                 [[0, 1],[0, 1]],
                                 default_parameters=[{"ax":0.2, "ay":0.2, "prod":2}],
                                 management_parameters=[{"ax":0.1, "ay":0.1, "prod":2}, {"ax":2, "ay":0, "prod":2}],
                                 ),
            "plants-hex":
                generate_example([pm.plants_rhs],
                                 [pm.plants_rhs]*2,
                                 pm.plants_sunny,
                                 [[0, 1],[0, 1]],
                                 default_parameters=[{"ax":0.2, "ay":0.2, "prod":2}],
                                 management_parameters=[{"ax":0.1, "ay":0.1, "prod":2}, {"ax":2, "ay":0, "prod":2}],
                                 grid_type="simplex-based",
                                 ),
            "techChange":
                generate_example([tcm.techChange_rhs],
                                 [tcm.techChange_rhs],
                                 tcm.techChange_sunny,
                                 [[0, 1], [0, 2]],
                                 default_parameters=[
                                     dict(rvar = 1, pBmin = 0.15, pE = 0.3, delta = 0.025, smax = 0.3, sBmax = None)],
                                 management_parameters=[
                                     dict(rvar = 1, pBmin = 0.15, pE = 0.3, delta = 0.025, smax = 0.3, sBmax = 0.5)],
                                 management_rhssPS = [tcm.techChange_rhsPS],
                                 ),
            "techChange-hex":
                generate_example([tcm.techChange_rhs],
                                 [tcm.techChange_rhs],
                                 tcm.techChange_sunny,
                                 [[0, 1], [0, 2]],
                                 default_parameters=[
                                     dict(rvar=1, pBmin=0.15, pE=0.3, delta=0.025, smax=0.3, sBmax=None)],
                                 management_parameters=[
                                     dict(rvar=1, pBmin=0.15, pE=0.3, delta=0.025, smax=0.3, sBmax=0.5)],
                                 management_rhssPS=[tcm.techChange_rhsPS],
                                 grid_type="simplex-based"
                                 ),
            "easter-a":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=1000, yMinimal=3000),
                                 [[0, 35000],[0, 18000]],
                                 default_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 4 * 10 ** (-6), delta = -0.1, kappa = 12000)],
                                 management_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 2.8 * 10 ** (-6), delta = -0.1, kappa = 12000)],
                                 backscaling=False,
                                 ),
            "easter-a-hex":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=1000, yMinimal=3000),
                                 [[0, 35000], [0, 18000]],
                                 default_parameters=[
                                     dict(phi=4, r=0.04, gamma=4 * 10 ** (-6), delta=-0.1, kappa=12000)],
                                 management_parameters=[
                                     dict(phi=4, r=0.04, gamma=2.8 * 10 ** (-6), delta=-0.1, kappa=12000)],
                                 grid_type="simplex-based"
                                 ),
            "easter-b":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=1200, yMinimal=2000),
                                 [[0, 9000], [0, 9000]],
                                 default_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 management_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 13.6 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 ),
            "easter-b-hex":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=1200, yMinimal=2000),
                                 [[0, 9000], [0, 9000]],
                                 default_parameters=[
                                     dict(phi=4, r=0.04, gamma=8 * 10 ** (-6), delta=-0.15, kappa=6000)],
                                 management_parameters=[
                                     dict(phi=4, r=0.04, gamma=13.6 * 10 ** (-6), delta=-0.15, kappa=6000)],
                                 grid_type="simplex-based"
                                 ),

            "easter-c":
                generate_example([prm.easter_rhs],
                                [prm.easter_rhs],
                                ft.partial(prm.easter_sunny, xMinimal=4000, yMinimal=3000),
                                [[0, 9000],[0, 9000]],
                                default_parameters=[
                                    dict(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                management_parameters=[
                                    dict(phi = 4, r = 0.04, gamma = 16 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                ),
            "easter-c-hex":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=4000, yMinimal=3000),
                                 [[0, 9000],[0, 9000]],
                                 default_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 management_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 16 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 grid_type="simplex-based"
                                 ),
            "easter-d":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=4000, yMinimal=3000),
                                 [[0, 9000], [0, 9000]],
                                 default_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 management_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 11.2 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 ),
            "easter-d-hex":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=4000, yMinimal=3000),
                                 [[0, 9000], [0, 9000]],
                                 default_parameters=[
                                     dict(phi=4, r=0.04, gamma=8 * 10 ** (-6), delta=-0.15, kappa=6000)],
                                 management_parameters=[
                                     dict(phi=4, r=0.04, gamma=11.2 * 10 ** (-6), delta=-0.15, kappa=6000)],
                                 grid_type="simplex-based"
                                 ),
            "consum":
                generate_example([],
                                 [cm.consum_rhs]*2,
                                 cm.consum_sunny,
                                 [[0, 2], [0, 3]],
                                 default_parameters = [],
                                 management_parameters = [dict(u = -0.5),
                                                       dict(u = 0.5)],
                                 management_rhssPS = [cm.consum_rhsPS]*2,
                                 ),
            "consum-hex":
                generate_example([],
                                 [cm.consum_rhs] * 2,
                                 cm.consum_sunny,
                                 [[0, 2], [0, 3]],
                                 default_parameters=[],
                                 management_parameters=[dict(u=-0.5),
                                                        dict(u=0.5)],
                                 management_rhssPS=[cm.consum_rhsPS] * 2,
                                 grid_type="simplex-based"
                                 ),

}

AVAILABLE_EXAMPLES = sorted(EXAMPLES)

## check that the special input "help" and "all" are not example names
assert not set(["all", "help"]).issubset(AVAILABLE_EXAMPLES)

if __name__ == "__main__":


    args = sys.argv[1:]

    if "help" in args:
        print("available examples are: " + ", ".join(AVAILABLE_EXAMPLES))
        sys.exit(0)

    if "all" in args:
        args = AVAILABLE_EXAMPLES

    assert set(args).issubset(AVAILABLE_EXAMPLES), "You mentioned an example " \
        "that I don't know ..."

    for example in args:
        print("computing example: " + example)
        EXAMPLES[example]()

    plt.show()
    assert False

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
        boundaries = [[xmin, xmax], [ymin, ymax]]
        PSboundaries = [xmin, ymin, xmax, ymax]
        periodicity = [1, -1] # on the x-axis but not the y-axis

        # default values
        a = 0.6
        gpm.l = 0.5

        # generating grid and step size values
        # xy, scalingfactor,  offset, x_step = viab.normalized_grid(boundaries, 80)
        xy, scalingfactor,  offset, x_step = viab.generate_grid(boundaries, 80, "orthogonal", periodicity = periodicity)
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

        viab.topology_classification(xy, state, [default_run], [management1_run], sunny, periodic_boundaries = periodicity)

        time_diff = time.time() - start_time
        print(time_diff)

        xy = viab.backscaling_grid(xy, scalingfactor, offset)

        # plotting
        viab.plot_points(xy, state)

        moddef.plotPhaseSpace(PSboundaries, topo.styleDefault)
        mod1.plotPhaseSpace(PSboundaries, topo.styleMod2)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])


        fig = plt.figure(figsize=(15, 15), tight_layout=True)

        viab.plot_areas(xy, state)

        moddef.plotPhaseSpace(PSboundaries, topo.styleDefault)
        mod1.plotPhaseSpace(PSboundaries, topo.styleMod2)

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

