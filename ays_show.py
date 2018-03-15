#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from ays_general import __version__, __version_info__
import ays_model as aws
import ays_general
from pyviability import helper

import numpy as np

import scipy.integrate as integ
import scipy.optimize as opt

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.ticker as ticker
from matplotlib import animation

import warnings as warn

import heapq as hq
import operator as op

import argparse, argcomplete

import pickle

import functools as ft

DG_BIFURCATION_END = "dg-bifurcation-end"
DG_BIFURCATION_MIDDLE = "dg-bifurcation-middle"
RUN_OPTIONS = [aws.DEFAULT_NAME] + list(aws.MANAGEMENTS) + [DG_BIFURCATION_END, DG_BIFURCATION_MIDDLE]

if __name__ == "__main__":

    # a small hack to make all the parameters available as global variables
    # aws.globalize_dictionary(aws.AWS_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters, module=aws)
    aws.globalize_dictionary(aws.boundary_parameters, module=aws)

    parser = argparse.ArgumentParser(description="sample trajectories of the AWS model")

    parser.add_argument("option", choices=RUN_OPTIONS, default=aws.DEFAULT_NAME, nargs="?",
                        help="choose either the default or one of the management options to show")
    parser.add_argument("-m", "--mode", choices=["all", "lake"], default="all",
                        help="which parts should be sampled (default 'all')")
    parser.add_argument("-n", "--num", type=int, default=400,
            help="number of initial conditions (default: 400)")
    parser.add_argument("--no-boundary", dest="draw_boundary", action="store_false",
                        help="remove the boundary inside the plot")
    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")
    parser.add_argument("--split", action="store_true",
                        help="split the plotting for the different basins of attraction")
    parser.add_argument("-z", "--zero", action="store_true",
            help="compute the zero of the RHS in the S=0 plane")
    parser.add_argument("-d", "--defense", default=0, choices=[0, 1, 2], type=int,
                        help="producing the figures for the defense")
    parser.add_argument("-w", "--with-curve", action="store_true",
                        help="plot the curve of the changing fixpoint for the beta management")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()


    # small hack for now
    args.options =[args.option]

    num = args.num
    aws_0 = np.random.rand(num,3)  # args.mode == "all"
    if args.mode == "lake":
        aws_0[0] = aws_0[0] * aws.A_PB / (aws.A_PB + aws.A_mid)
    else:
        border_epsilon =1e-2
        aws_0 = border_epsilon + (1-2*border_epsilon) * aws_0

    ########################################
    # prepare the integration
    ########################################
    time = np.linspace(0, 300, 1000)

    # parameter_dicts = []
    parameter_lists = []
    for management in args.options:
        if management == DG_BIFURCATION_END:
            parameter_dict = aws.get_management_parameter_dict("degrowth", aws.AYS_parameters)
            parameter_dict["beta"] = 0.035
        elif management == DG_BIFURCATION_MIDDLE:
            parameter_dict = aws.get_management_parameter_dict("degrowth", aws.AYS_parameters)
            parameter_dict["beta"] = 0.027
        else:
            parameter_dict = aws.get_management_parameter_dict(management, aws.AYS_parameters)
        if args.zero:
            x0 = [0.5, 0.5, 0] # a, w, s
            print("fixed point(s) of {}:".format(management))
            # below the '0' is for the time t
            print(opt.fsolve(aws.AYS_rescaled_rhs, x0,
                             args=(0., ) + helper.get_ordered_parameters(aws._AYS_rhs, parameter_dict)))
            print()
        parameter_lists.append(helper.get_ordered_parameters(aws._AYS_rhs, parameter_dict))
    # colors = ["green", "blue", "red"]
    # assert len(parameter_lists) <= len(colors), "need to add colors"


    colortop = "green"
    colorbottom = "black"
    traj_kwargs = dict(
        alpha = 0.3,
        lw=2,
    )


    if args.split:
        fig_green, ax3d_green = ays_general.create_figure(A_mid=aws.A_mid, W_mid=aws.W_mid, S_mid=aws.S_mid, isdefense=args.defense)
        if args.defense != 2:
            ax3d_green.view_init(ays_general.ELEVATION_FLOW, ays_general.AZIMUTH_FLOW)

        fig_black, ax3d_black = ays_general.create_figure(A_mid=aws.A_mid, W_mid=aws.W_mid, S_mid=aws.S_mid, isdefense=args.defense)
        if args.defense != 2:
            ax3d_black.view_init(ays_general.ELEVATION_FLOW, ays_general.AZIMUTH_FLOW)

    else:
        fig, ax3d = ays_general.create_figure(A_mid=aws.A_mid, W_mid=aws.W_mid, S_mid=aws.S_mid, isdefense=args.defense)
        if args.defense != 2:
            ax3d.view_init(ays_general.ELEVATION_FLOW, ays_general.AZIMUTH_FLOW)

    for i in range(num):
        x0 = aws_0[i]
        # management trajectory with degrowth:
        for parameter_list in parameter_lists:
            traj = integ.odeint(aws.AYS_rescaled_rhs, x0, time, args=parameter_list)
            if args.split:
                if traj[-1,2]<0.5:
                    ax3d_black.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                                color=colorbottom, **traj_kwargs)
                else:
                    ax3d_green.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                                color=colortop, **traj_kwargs)
            else:
                ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                            color=colorbottom if traj[-1,2]<0.5 else colortop, **traj_kwargs)

        # below traj was default and traj2 was degrowth
        # if traj2[:,0].max() > aws.A_PB > traj[:,0].max() and traj[-1,2] < 1e10 and traj2[-1,2] > 1e10: # lake candidate!
            # # JH: transform so that W_SF,sigma_default go to 1/2 and infinity goes to 1:
            # ax3d.plot3D(xs=traj[:,0], ys=traj[:,1]/(aws.W_mid+traj[:,1]), zs=traj[:,2]/(aws.S_mid+traj[:,2]),
                        # color="red" if traj[-1,2]<1000 else "blue", alpha=.7)
            # ax3d.plot3D(xs=traj2[:,0], ys=traj2[:,1]/(aws.W_mid+traj2[:,1]), zs=traj2[:,2]/(aws.S_mid+traj2[:,2]),
                        # color="orange" if traj2[-1,2]<1000 else "cyan", alpha=.7)
            # #print(traj2[:,0].max() - traj[:,0].max())

    if args.split:
        fig_2d = plt.figure("bottom-flow", figsize=(7.5, 7.5), tight_layout=True)
        ax_2d = fig_2d.add_subplot(111)
        flow_steps = 20
        X = np.linspace(border_epsilon, 1-border_epsilon, flow_steps)
        Y = np.linspace(border_epsilon, 1-border_epsilon, flow_steps)
        Z = np.array([0])

        XY = np.squeeze(np.meshgrid(X, Y, Z))
        evol = lambda xy: aws._AYS_rescaled_rhs(xy, 0, *parameter_list)
        dX, dY, dZ = evol(XY)  # that is where deriv from Vera is mapped to
        data = [X, Y, dX, dY]
        c = ax_2d.streamplot(*data, color=colorbottom, linewidth=2)
        print(aws.W_SF, aws.W_mid)
        a_PB = aws.A_PB / (aws.A_PB + aws.A_mid)
        y_SF = aws.W_SF / (aws.W_SF + aws.W_mid)
        boundary_traj = np.array([[a_PB,1], [a_PB, y_SF], [0, y_SF]]).T
        print(boundary_traj[0])
        ax_2d.plot(boundary_traj[0], boundary_traj[1], color="gray", lw=5, alpha=0.8)
        y_b = aws.AYS_black_fp_rescaled(*parameter_list)
        print("y_b", y_b)
        if args.with_curve:
            print("doing the curve")
            curve = []
            print("original", parameter_list)
            for beta in np.linspace(0.015, 0.03, 20):
                new_ays_params = dict(aws.AYS_parameters)
                new_ays_params["beta"] = beta
                parameter_dict = aws.get_management_parameter_dict(management, new_ays_params)
                parameter_list = helper.get_ordered_parameters(aws._AYS_rhs, parameter_dict)
                print(beta, parameter_list)
                curve.append(aws.AYS_black_fp_rescaled(*parameter_list))
            curve = np.array(curve)
            ax_2d.plot(curve[0,0], curve[0, 1], color="blue", ms=12, marker="8")
            ax_2d.plot(curve[:, 0], curve[:,1], color="blue", lw=3)
        ax_2d.plot(y_b[0], y_b[1], color=colorbottom, ms=12, marker="8")
        ax_2d.set_xlabel("A [GtC]", fontsize = 25)
        ax_2d.set_ylabel("Y [1e+12 USD/yr]", fontsize = 25)
        ax_2d.tick_params(labelsize=25)
        ax_2d.set_xlim(0,1)
        ax_2d.set_ylim(0,1)
        start, stop = 0, np.infty
        transf = ft.partial(ays_general.compactification, x_mid=aws.A_mid)
        inv_transf = ft.partial(ays_general.inv_compactification, x_mid=aws.A_mid)
        formatters, locators = ays_general.transformed_space(transf, inv_transf, axis_use=True, start=start, stop=stop, num=7,
                                                 isdefense=args.defense)
        ax_2d.xaxis.set_major_locator(ticker.FixedLocator(locators))
        ax_2d.xaxis.set_major_formatter(ticker.FixedFormatter(formatters))

        W_scale = 1e12
        transf = ft.partial(ays_general.compactification, x_mid=aws.W_mid)
        inv_transf = ft.partial(ays_general.inv_compactification, x_mid=aws.W_mid)
        formatters, locators = ays_general.transformed_space(transf, inv_transf, axis_use=True, start=start, stop=stop, num=7, scale=W_scale,
                                                 isdefense=args.defense)
        ax_2d.yaxis.set_major_locator(ticker.FixedLocator(locators))
        ax_2d.yaxis.set_major_formatter(ticker.FixedFormatter(formatters))


    if args.draw_boundary:
        if args.split:
            ays_general.add_boundary(ax3d_green,
                                     sunny_boundaries=["planetary-boundary", "social-foundation"],
                                     **aws.grid_parameters, **aws.boundary_parameters)
            ays_general.add_boundary(ax3d_black,
                                     sunny_boundaries=["planetary-boundary", "social-foundation"],
                                     **aws.grid_parameters, **aws.boundary_parameters)
        else:
            ays_general.add_boundary(ax3d,
                                     sunny_boundaries=["planetary-boundary", "social-foundation"],
                                     **aws.grid_parameters, **aws.boundary_parameters)

    if args.save_pic:
        if args.split:
            figs = {
                fig_green : "green",
                fig_black : "black",
                fig_2d : "bottom",
            }
            if args.with_curve:
                figs[fig_2d] += "-with-curve"
            for fig, midfix in figs.items():
                fname = f"aws-plot-{args.option}-{midfix}.jpg"
                print("saving to {} ... ".format(fname), end="", flush=True)
                fig.savefig(fname)
                print("done")

        else:
            print("saving to {} ... ".format(args.save_pic), end="", flush=True)
            fig.savefig(args.save_pic)
            print("done")

    plt.show()




