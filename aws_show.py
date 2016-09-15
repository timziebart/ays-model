
import aws_model as aws

import numpy as np

import scipy.integrate as integ

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.ticker as ticker

import warnings as warn

import argparse

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
                      endpoint=True,
                      axis_use=False):
    add_infty = False
    if stop == np.infty and endpoint:
        add_infty = True
        endpoint = False
        num -= 1

    locators_start = transform(start)
    locators_stop = transform(stop)

    locators = np.linspace(locators_start,
                           locators_stop,
                           num,
                           endpoint=endpoint)

    formatters = inv_transform(locators)
    formatters = formatters / scale

    if add_infty:
        # assume locators_stop has the transformed value for infinity already
        locators = np.concatenate((locators, [locators_stop]))
    if axis_use:
        formatters = np.round(formatters, decimals=2)
        formatters = formatters.astype(int)
        formatters = np.concatenate((formatters, [ INFTY_SIGN ]))
        return formatters, locators
    else:
        formatters = np.concatenate(( formatters, [ np.infty ]))
        return formatters


def create_figure(*, S_scale = 1e9, W_scale = 1e12, W_mid = None, S_mid = None, **kwargs):


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

        num_minor = 5
        minor_formatters = np.array([])
        for i in range(len(formatters) - 2):
            minor_formatters = np.concatenate((minor_formatters, np.linspace(float(formatters[i]), float(formatters[i+1]), num_minor, endpoint=False)))
        minor_locators = transf(minor_formatters)
        minor_locators = np.concatenate(( minor_locators, locators[-2:] ))
        _minor_formatters = np.array([])
        for f in formatters[:-2]:
            _minor_formatters = np.concatenate(( _minor_formatters, np.array([f]), np.array([""]*(num_minor-1)) ))
        _minor_formatters = np.concatenate(( _minor_formatters, formatters[-2:] ))


        # ax3d.w_xaxis.set_major_locator(ticker.FixedLocator(locators))
        # ax3d.w_xaxis.set_major_formatter(ticker.FixedFormatter(formatters))
        ax3d.w_xaxis.set_major_locator(ticker.FixedLocator(minor_locators))
        ax3d.w_xaxis.set_major_formatter(ticker.FixedFormatter(_minor_formatters))
        ax3d.set_xlim(0,1)
    else:
        raise KeyError("can't find proper key for 'A' in kwargs that determines which representation of 'A' has been used")

    if kwargs:
        warn.warn("omitted arguments: {}".format(", ".join(sorted(kwargs))), stacklevel=2)

    transf = ft.partial(compactification, x_mid=W_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=W_mid)
    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, scale=W_scale)
    # ax3d.w_yaxis.set_major_locator(ticker.FixedLocator(locators))
    # ax3d.w_yaxis.set_major_formatter(ticker.FixedFormatter(formatters))
    num_minor = 5
    minor_formatters = np.array([])
    for i in range(len(formatters) - 2):
        minor_formatters = np.concatenate((minor_formatters, np.linspace(float(formatters[i])*W_scale, float(formatters[i+1])*W_scale, num_minor, endpoint=False)))
    minor_locators = transf(minor_formatters)
    minor_locators = np.concatenate(( minor_locators, locators[-2:] ))
    _minor_formatters = np.array([])
    for f in formatters[:-2]:
        _minor_formatters = np.concatenate(( _minor_formatters, np.array([f]), np.array([""]*(num_minor-1)) ))
    _minor_formatters = np.concatenate(( _minor_formatters, formatters[-2:] ))
    ax3d.w_yaxis.set_major_locator(ticker.FixedLocator(minor_locators))
    ax3d.w_yaxis.set_major_formatter(ticker.FixedFormatter(_minor_formatters))
    ax3d.set_ylim(0,1)


    transf = ft.partial(compactification, x_mid=S_mid)
    inv_transf = ft.partial(inv_compactification, x_mid=S_mid)
    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, scale=S_scale)
    # ax3d.w_zaxis.set_major_locator(ticker.FixedLocator(locators))
    # ax3d.w_zaxis.set_major_formatter(ticker.FixedFormatter(formatters))
    num_minor = 5
    minor_formatters = np.array([])
    for i in range(len(formatters) - 2):
        minor_formatters = np.concatenate((minor_formatters, np.linspace(float(formatters[i])*S_scale, float(formatters[i+1])*S_scale, num_minor, endpoint=False)))
    minor_locators = transf(minor_formatters)
    minor_locators = np.concatenate(( minor_locators, locators[-2:] ))
    _minor_formatters = np.array([])
    for f in formatters[:-2]:
        _minor_formatters = np.concatenate(( _minor_formatters, np.array([f]), np.array([""]*(num_minor-1)) ))
    _minor_formatters = np.concatenate(( _minor_formatters, formatters[-2:] ))
    ax3d.w_zaxis.set_major_locator(ticker.FixedLocator(minor_locators))
    ax3d.w_zaxis.set_major_formatter(ticker.FixedFormatter(_minor_formatters))
    # Sticks = np.concatenate((np.linspace(0, S_mid, 11)[:-1],np.linspace(0, S_mid*10, 6)[1:]))
    # ax3d.w_zaxis.set_major_locator(ticker.FixedLocator(np.concatenate((Sticks/(S_mid+Sticks),[1]))))
    # ax3d.w_zaxis.set_major_formatter(ticker.FixedFormatter(np.concatenate(((Sticks/1e9).astype("int"),["inf"]))))
    ax3d.set_zlim(0,1)

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


if __name__ == "__main__":

    # a small hack to make all the parameters available as global variables
    # aws.globalize_dictionary(aws.AWS_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters, module=aws)
    aws.globalize_dictionary(aws.boundary_parameters, module=aws)

    parser = argparse.ArgumentParser(description="sample trajectories of the AWS model")

    parser.add_argument("-m", "--mode", choices=["all", "lake"], default="all",
                        help="which parts should be sampled")
    parser.add_argument("-n", "--num", type=int, default=1000,
                        help="number of initial conditions")

    args = parser.parse_args()

    print(args)

    num = args.num
    aws_0 = np.random.rand(num,3)  # args.mode == "all"
    if args.mode == "lake":
        aws_0[0] = aws_0[0] * aws.A_PB / (aws.A_PB + aws.A_mid)

    # sample = np.random.rand(num,3)
    # AWS_0 = np.zeros((num, 3))
    # sample whole space:
    #AWS_0[:,0] = sample[:,0] * aws.A_max
    #AWS_0[:,1:] = sample[:,1:]/(1-sample[:,1:]) * WS_mid
    # sample DG + ET + PB lake candidates:
    # AWS_0[:,0] = sample[:,0] * aws.A_PB
    # AWS_0[:,1] = sample[:,1]/(1-sample[:,1]) * aws.W_mid
    # AWS_0[:,2] = sample[:,2]/(1-sample[:,2]) * aws.S_mid

    fig, ax3d = create_figure(aws.A_max, aws.W_mid, aws.S_mid)

    for i in range(num):
        x0 = AWS_0[i]
        time = np.linspace(0, 1000, 1000)
        # management trajectory with degrowth:
        beta = aws.beta_DG
        aws.sigma = aws.sigma_default
        traj = integ.odeint(aws.AWS_rhs, x0, time, args=(beta,))
        # alternative trajectory with tax/subsidy:
        beta = aws.beta_default
        aws.sigma = aws.sigma_ET
        traj2 = integ.odeint(aws.AWS_rhs, x0, time, args=(beta,))
        if traj2[:,0].max() > aws.A_PB > traj[:,0].max() and traj[-1,2] < 1e10 and traj2[-1,2] > 1e10: # lake candidate!
            # JH: transform so that W_SF,sigma_default go to 1/2 and infinity goes to 1:
            ax3d.plot3D(xs=traj[:,0], ys=traj[:,1]/(aws.W_mid+traj[:,1]), zs=traj[:,2]/(aws.S_mid+traj[:,2]),
                        color="red" if traj[-1,2]<1000 else "blue", alpha=.7)
            ax3d.plot3D(xs=traj2[:,0], ys=traj2[:,1]/(aws.W_mid+traj2[:,1]), zs=traj2[:,2]/(aws.S_mid+traj2[:,2]),
                        color="orange" if traj2[-1,2]<1000 else "cyan", alpha=.7)
            #print(traj2[:,0].max() - traj[:,0].max())


    add_boundary(ax3d, aws.A_PB)


    plt.show()




