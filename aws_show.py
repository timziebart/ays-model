
import aws_model as aws
from pyviability import helper

import numpy as np

import scipy.integrate as integ

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.ticker as ticker

import warnings as warn

import heapq as hq
import operator as op

import argparse

import pickle

import functools as ft

INFTY_SIGN = u"\u221E"

# patch to remove padding at ends of axes:
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###


@np.vectorize
def compactification(x, x_mid):
    if x == 0:
        return 0.
    if x == np.infty:
        return 1.
    return x / (x + x_mid)


@np.vectorize
def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)


def transformed_space(transform, inv_transform,
                      start=0, stop=np.infty, num=12,
                      scale=1,
                      num_minors = 50,
                      endpoint=True,
                      axis_use=False):
    add_infty = False
    if stop == np.infty and endpoint:
        add_infty = True
        endpoint = False
        num -= 1

    locators_start = transform(start)
    locators_stop = transform(stop)

    major_locators = np.linspace(locators_start,
                           locators_stop,
                           num,
                           endpoint=endpoint)

    major_formatters = inv_transform(major_locators)
    # major_formatters = major_formatters / scale

    major_combined = list(zip(major_locators, major_formatters))
    # print(major_combined)

    _minor_formatters = np.linspace(major_formatters[0], major_formatters[-1], num_minors, endpoint=False)[1:]
    minor_locators = transform(_minor_formatters)
    minor_formatters = [np.nan] * len(minor_locators)
    minor_combined = list(zip(minor_locators, minor_formatters))
    # print(minor_combined)

    combined = list(hq.merge(minor_combined, major_combined, key = op.itemgetter(0)))

    # print(combined)

    locators, formatters = map(np.array, zip(*combined))
    formatters = formatters / scale

    if add_infty:
        # assume locators_stop has the transformed value for infinity already
        locators = np.concatenate((locators, [locators_stop]))
        formatters = np.concatenate(( formatters, [ np.infty ]))

    if not axis_use:
        return formatters

    else:
        string_formatters = np.zeros_like(formatters, dtype="|U10")
        mask_nan = np.isnan(formatters)
        if add_infty:
            string_formatters[-1] = INFTY_SIGN
            mask_nan[-1] = True
        string_formatters[~mask_nan] = np.round(formatters[~mask_nan], decimals=2).astype(int).astype("|U10")
        return string_formatters, locators


def create_figure(*bla, S_scale = 1e9, W_scale = 1e12, W_mid = None, S_mid = None, boundaries = None, **kwargs):


    kwargs = dict(kwargs)

    fig = plt.figure(figsize=(16,9))
    ax3d = plt3d.Axes3D(fig)
    ax3d.set_xlabel("\n\nexcess atmospheric carbon\nstock A [GtC]")
    ax3d.set_ylabel("\nwelfare W [%1.0e USD/yr]"%W_scale)
    ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [%1.0e GJ]"%S_scale)

    # make proper tickmarks:
    if "A_max" in kwargs:
        A_max = kwargs.pop("A_max")
        Aticks = np.linspace(0,A_max,11)
        ax3d.w_xaxis.set_major_locator(ticker.FixedLocator(Aticks))
        ax3d.w_xaxis.set_major_formatter(ticker.FixedFormatter(Aticks.astype("int")))
        ax3d.set_xlim(Aticks[0],Aticks[-1])
    elif "A_mid" in kwargs:
        A_mid = kwargs.pop("A_mid")
        transf = ft.partial(compactification, x_mid=A_mid)
        inv_transf = ft.partial(inv_compactification, x_mid=A_mid)

        formatters, locators = transformed_space(transf, inv_transf, axis_use=True)
        ax3d.w_xaxis.set_major_locator(ticker.FixedLocator(locators))
        ax3d.w_xaxis.set_major_formatter(ticker.FixedFormatter(formatters))

        ax3d.set_xlim(0,1)

    else:
        raise KeyError("can't find proper key for 'A' in kwargs that determines which representation of 'A' has been used")

    if kwargs:
        warn.warn("omitted arguments: {}".format(", ".join(sorted(kwargs))), stacklevel=2)

    transf = ft.partial(compactification, x_mid=W_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=W_mid)

    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, scale=W_scale)
    ax3d.w_yaxis.set_major_locator(ticker.FixedLocator(locators))
    ax3d.w_yaxis.set_major_formatter(ticker.FixedFormatter(formatters))

    ax3d.set_ylim(0,1)


    transf = ft.partial(compactification, x_mid=S_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=S_mid)

    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, scale=S_scale)
    ax3d.w_zaxis.set_major_locator(ticker.FixedLocator(locators))
    ax3d.w_zaxis.set_major_formatter(ticker.FixedFormatter(formatters))

    ax3d.set_zlim(0,1)

    # if not boundaries is None:
        # print(boundaries)
        # ax3d.set_xlim(0.1, 0.6)
        # ax3d.set_xlim(*boundaries[0])
        # ax3d.set_ylim(*boundaries[1])
        # ax3d.set_zlim(*boundaries[2])

    ax3d.view_init(30, -140)

    return fig, ax3d


def add_boundary(ax3d, boundary= "PB", add_outer=False, **parameters):
    # show boundaries of undesirable region:
    if boundary == "PB":
        A_PB = parameters["A_PB"]
        if "A_max" in parameters:
            pass # no transformation necessary
        elif "A_mid" in parameters:
            A_PB = A_PB / (A_PB + parameters["A_mid"])
        else:
            assert False, "couldn't identify how the A axis is scaled"
        boundary_surface_PB = plt3d.art3d.Poly3DCollection([[[A_PB,0,0],[A_PB,1,0],[A_PB,1,1],[A_PB,0,1]]])
        boundary_surface_PB.set_color("gray"); boundary_surface_PB.set_edgecolor("gray"); boundary_surface_PB.set_alpha(0.25)
        ax3d.add_collection3d(boundary_surface_PB)
    elif boundary == "both":
        raise NotImplementedError("will be done soon")
        boundary_surface_both = plt3d.art3d.Poly3DCollection([[[0,.5,0],[0,.5,1],[A_PB,.5,1],[A_PB,.5,0]],
                                                        [[A_PB,.5,0],[A_PB,1,0],[A_PB,1,1],[A_PB,.5,1]]])
        boundary_surface_both.set_color("gray"); boundary_surface_both.set_edgecolor("gray"); boundary_surface_both.set_alpha(0.25)
        ax3d.add_collection3d(boundary_surface_both)
    else:
        raise NameError("Unkown boundary {!r}".format(boundary))

    if add_outer:
        # add outer limits of undesirable view from standard view perspective:
        undesirable_outer_stdview = plt3d.art3d.Poly3DCollection([[[0,0,0],[0,0,1],[0,.5,1],[0,.5,0]],
                                            [[A_PB,1,0],[aws.A_max,1,0],[aws.A_max,1,1],[A_PB,1,1]],
                                            [[0,0,0],[0,.5,0],[A_PB,.5,0],[A_PB,1,0],[aws.A_max,1,0],[aws.A_max,0,0]]])
        undesirable_outer_stdview.set_color("gray"); undesirable_outer_stdview.set_edgecolor("gray"); undesirable_outer_stdview.set_alpha(0.25)
        ax3d.add_collection3d(undesirable_outer_stdview)


RUN_OPTIONS = [aws.DEFAULT_NAME] + list(aws.MANAGEMENTS)

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
                        help="number of initial conditions")
    parser.add_argument("--no-boundary", dest="draw_boundary", action="store_false",
                        help="remove the boundary inside the plot")
    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")

    args = parser.parse_args()


    # small hack for now
    args.options =[args.option]

    num = args.num
    aws_0 = np.random.rand(num,3)  # args.mode == "all"
    if args.mode == "lake":
        aws_0[0] = aws_0[0] * aws.A_PB / (aws.A_PB + aws.A_mid)

    fig, ax3d = create_figure(A_mid=aws.A_mid, W_mid=aws.W_mid, S_mid=aws.S_mid)

    ########################################
    # prepare the integration
    ########################################
    time = np.linspace(0, 300, 1000)

    # parameter_dicts = []
    parameter_lists = []
    for management in args.options:
        parameter_dict = aws.get_management_parameter_dict(management, aws.AWS_parameters)
        parameter_lists.append( helper.get_ordered_parameters(aws._AWS_rhs, parameter_dict))
    # colors = ["green", "blue", "red"]
    # assert len(parameter_lists) <= len(colors), "need to add colors"

    colortop = "green"
    colorbottom = "black"

    for i in range(num):
        x0 = aws_0[i]
        # management trajectory with degrowth:
        for parameter_list in parameter_lists:
            traj = integ.odeint(aws.AWS_rescaled_rhs, x0, time, args=parameter_list)
            ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                        color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.3)

        # below traj was default and traj2 was degrowth
        # if traj2[:,0].max() > aws.A_PB > traj[:,0].max() and traj[-1,2] < 1e10 and traj2[-1,2] > 1e10: # lake candidate!
            # # JH: transform so that W_SF,sigma_default go to 1/2 and infinity goes to 1:
            # ax3d.plot3D(xs=traj[:,0], ys=traj[:,1]/(aws.W_mid+traj[:,1]), zs=traj[:,2]/(aws.S_mid+traj[:,2]),
                        # color="red" if traj[-1,2]<1000 else "blue", alpha=.7)
            # ax3d.plot3D(xs=traj2[:,0], ys=traj2[:,1]/(aws.W_mid+traj2[:,1]), zs=traj2[:,2]/(aws.S_mid+traj2[:,2]),
                        # color="orange" if traj2[-1,2]<1000 else "cyan", alpha=.7)
            # #print(traj2[:,0].max() - traj[:,0].max())


    if args.draw_boundary:
        add_boundary(ax3d, **aws.grid_parameters, **aws.boundary_parameters)

    if args.save_pic:
        print("saving to {} ... ".format(args.save_pic), end="", flush=True)
        fig.savefig(args.save_pic)
        print("done")

    plt.show()




