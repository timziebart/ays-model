#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import pyviability as viab
from pyviability import libviability as lv

from aws_general import __version__, __version_info__
import aws_model as aws
import aws_show, aws_general

from scipy import spatial as spat
from scipy.spatial import ckdtree
import numpy as np
import pickle, argparse, argcomplete
import ast, sys
import itertools as it

import datetime as dt
import functools as ft

import matplotlib.pyplot as plt

def RegionName2Option(vname, style="long"):
    if style=="long":
        return vname.replace("_", "-").lower()
    elif style=="short":
        return "".join(map(lambda x: x[0], vname.lower().split("_")))
    raise ValueError("unkown style: {!r}".format(style))


# check that there are no short option used twice
_all_regions_short = list(map(lambda x: RegionName2Option(x, style="short"), lv.REGIONS))
assert len(_all_regions_short) == len(set(_all_regions_short))
del _all_regions_short

def get_changed_parameters(pars, default_pars):
    changed_pars = {}
    for par, val in pars.items():
        if not par in default_pars:
            changed_pars[par] = (val, None)
        elif isinstance(default_pars[par], np.ndarray):
            if not np.allclose(default_pars[par], val):
                changed_pars[par] = (val, default_pars[par])
        elif default_pars[par] != val:
            changed_pars[par] = (val, default_pars[par])

    return changed_pars

# prepare all the stuff needed for the regions argument parsing
regions_dict_short = { RegionName2Option(region, style="short") : region for region in lv.REGIONS }
regions_dict_long = { RegionName2Option(region, style="long") : region for region in lv.REGIONS }
regions_dict = dict(regions_dict_long)
regions_dict.update(regions_dict_short)
regions_arguments = [("all", "a")] + list(zip(map(RegionName2Option, lv.REGIONS), map(ft.partial(RegionName2Option, style="short"), lv.REGIONS)))
regions_arguments_flattened = sorted([item for sublist in regions_arguments for item in sublist])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="show the TSM results of the AWS model")
    parser.add_argument("input_file", metavar="input-file",
                        help="input file with the contents from the TSM analysis")

    # parser.add_argument("-a", "--animate", action="store_true",
                        # help="animate the 3d plot")
    parser.add_argument("-b", "--plot-boundaries", metavar="boundaries",
                        help="set the boundaries as a list with shape (3,2)")
    parser.add_argument("-d", "--defaults", default=[], nargs="+",
                        choices=["grid", "model", "boundary"],
                        help="show all the default values")

    paths_parser = parser.add_argument_group(title="paths tool",
                                             description="tools to show paths in the plot")
    paths_parser.add_argument("-p", "--show-path", nargs=2, metavar=("point", "distance"),
                              help="show a path for all points, that are closer to 'point' than 'distance'")
    paths_parser.add_argument("--paths-outside", action="store_true",
                              help="paths go go out of the plotting boundaries")
    paths_parser.add_argument("--no-paths-lake-fallback", action="store_false", dest="paths_lake_fallback",
                              help="fallback to PATHS if NO INFO in PATHS_LAKE")

    regions_parser = parser.add_argument_group(title="plot regions",
                                               description="choose which regions are plotted and how")
    regions_parser.add_argument("-r", "--show-region", metavar="region", dest="regions", 
                                default=[], nargs="+", choices=regions_arguments_flattened,
                                help="choose the regions to be shown in the plot: " + 
                                     ", ".join(["{} ({})".format(region_long, region_short) for region_long, region_short in regions_arguments]))
    region_plotting_styles = ["points", "surface"]
    regions_parser.add_argument("--regions-style", choices=region_plotting_styles, default=region_plotting_styles[0],
                                help="choose the plotting style from: " + ", ".join(region_plotting_styles))

    parser.add_argument("--reformat", action="store_true",
                        help="automatically reformat 'input-file' if necessary")
    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")
    # parser.add_argument("--save-video", metavar="file", 
                        # help="save a video of the 3d result")
    parser.add_argument("-t", "--transformed-formatters", action="store_true",
                        help="show from 0 to 1 at each axis instead of 0 to infty")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase verbosity can be used as -v, -vv ...")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # if args.save_video and not args.animate:
        # parser.error("no use to produce a video without animating the plot")

    if args.defaults:
        for d in args.defaults:
            print("defaults for {}:".format(d))
            if d == "grid":
                dic = aws.grid_parameters
            elif d == "model":
                dic = aws.AWS_parameters
            elif d == "boundary":
                dic = aws.boundary_parameters
            else:
                raise ValueError("Did you forget to change something here?")
            print(aws.recursive_dict2string(dic))
            print()
        sys.exit(0)

    # evaluate the boundaries string to an array
    if args.plot_boundaries:
        args.plot_boundaries = np.array(eval(args.plot_boundaries))

    # resolve the chosen regions and translate them to the names in pyviability
    if args.regions:
        if "a" in args.regions or "all" in args.regions:
            args.regions = lv.REGIONS
        else:
            args.regions = list(set(map(regions_dict.__getitem__, args.regions)))

    try:
        header, data = aws_general.load_result_file(args.input_file, auto_reformat=args.reformat, verbose=1)
    except IOError:
            parser.error("seems to be an older aws file version, please use the '--reformat' option")
    print()

    if not header["viab-backscaling-done"]:
        raise NotImplementedError("there is no plotting for unrescaled systems yet (and probably won't ever be)")

    if not args.show_path is None:
        if not header["remember-paths"]:
            parser.error("'{}' does not contain recorded paths".format(args.input_file))
        # else
        path_x0 = np.array(eval(args.show_path[0]))
        path_dist = float(eval(args.show_path[1]))
        assert path_x0.shape == (3,)


    grid = data["grid"]
    states = data["states"]

    print("date: {}".format(dt.datetime.fromtimestamp(header["start-time"]).ctime()))
    print("duration: {!s}".format(dt.timedelta(seconds=header["run-time"])))
    print()
    print("management options: {}".format(", ".join(header["managements"]) if header["managements"] else "(None)"))
    pars = header["model-parameters"]  # just to make it shorter here
    for m in header["managements"]:
        ending = "_" + aws.MANAGEMENTS[m].upper()
        changed = False
        for key in pars:
            # choose the variables that are changed by the ending
            if key.endswith(ending):
                default_key = key[:-len(ending)]
                print("{} = {} <--> {} = {}".format(key, pars[key], default_key, pars[default_key]))
    print()
    assert header["boundaries"] == ["planetary-boundary"], "only PB is implemented for showing"
    # print("boundaries: {}".format(", ".join(header["boundaries"])))
    print("boundaries:")
    A_PB = header["boundary-parameters"]["A_PB"]
    A_mid = header["grid-parameters"]["A_mid"]
    A_offset = header["model-parameters"]["A_offset"]
    print("planetary / CO2 concentration:", end=" ")
    print("A_PB = {:6.2f} GtC above equ. <=> {:6.2f} ppm <=> a_PB = {:5.3f}".format(A_PB, (A_PB + A_offset) / 840 * 400 , A_PB / (A_mid + A_PB)))
    print()
    print("stepsize / gridstepsize: {:<5.3f}".format(header["stepsize"] / header["xstep"]))
    print()
    print("points per dimension: {:4d}".format(header["grid-parameters"]["n0"]))
    print()
    print("paths recorded: {}".format(header["remember-paths"]))
    if args.show_path:
        print("showing for", path_x0, path_dist)
    print()

    model_changed_pars = get_changed_parameters(header["model-parameters"], aws.AWS_parameters)
    grid_changed_pars = get_changed_parameters(header["grid-parameters"], aws.grid_parameters)
    boundary_changed_pars = get_changed_parameters(header["boundary-parameters"], aws.boundary_parameters)
    if model_changed_pars:
        print("changed model parameters:")
        for par in sorted(model_changed_pars):
            print("{} = {!r} (default: {!r})".format(par, *model_changed_pars[par]))
        print()
    if grid_changed_pars:
        print("changed grid parameters:")
        for par in sorted(grid_changed_pars):
            print("{} = {!r} (default: {!r})".format(par, *grid_changed_pars[par]))
        print()
    if boundary_changed_pars:
        print("changed boundary parameters:")
        for par in sorted(boundary_changed_pars):
            print("{} = {!r} (default: {!r})".format(par, *boundary_changed_pars[par]))
        print()


    viab.print_evaluation(states)

    if not args.regions and args.show_path is None:
        print("no regions for plotting chosen")
    else:

        # if args.boundaries:
        # header["grid-parameters"]["boundaries"] = args.boundaries
        figure_parameters = dict(header["grid-parameters"])
        figure_parameters["boundaries"] = args.plot_boundaries
        fig, ax3d = aws_show.create_figure(transformed_formatters=args.transformed_formatters, **figure_parameters)
        print()

        ax_parameters = dict(header["boundary-parameters"])  # make a copy
        ax_parameters.update(header["grid-parameters"])
        aws_show.add_boundary(ax3d, plot_boundaries=args.plot_boundaries, **ax_parameters)

        def isinside(x, bounds):
            if bounds is None:
                return np.ones(np.shape(x)[:-1], dtype=bool)
            return np.all((bounds[:, 0] <= x) & ( x <= bounds[:, 1]), axis=-1)

        mask2 = isinside(grid, args.plot_boundaries)

        if args.regions_style == "points":
            for region in args.regions:
                region_num = getattr(lv, region)
                mask = (states == region_num) &  mask2
                ax3d.plot3D(xs=grid[:, 0][mask], ys=grid[:, 1][mask], zs=grid[:, 2][mask],
                                color=lv.COLORS[region_num],
                            alpha=1/header["grid-parameters"]["n0"],
                            linestyle="", marker=".", markersize=30,
                            )
        else:
            raise NotImplementedError("plotting style '{}' is not yet implemented".format(args.regions_style))
        if args.show_path:
            # paths = data["paths"]
            print("compute starting indices ... ", end="", flush=True)
            diff = grid - path_x0
            mask = (np.linalg.norm(diff, axis=-1) <= path_dist)
            starting_indices = np.where(mask)[0].tolist()
            _starting_indices = list(starting_indices)
            print("done")
            print()
            if not starting_indices:
                print("your point and distance do not match any grid points")
            else:
                print("matched:")
                _matched_states = states[mask]
                matched_states = sorted(np.unique(_matched_states))
                for s in matched_states:
                    print("{:>2} : {:>2}".format(s, np.count_nonzero(_matched_states == s)))
                print()
                plotting = lambda traj, choice: ax3d.plot3D(xs=traj[0], ys=traj[1], zs=traj[2],
                                                            color="lightblue" if choice == 0 else "black")
                bounds = args.plot_boundaries
                paths_outside = args.paths_outside
                if paths_outside or bounds is None:
                    path_isinside = aws_general.dummy_isinside
                else:
                    def path_isinside(x):
                        return np.all((bounds[:, 0] <= x) & ( x <= bounds[:, 1]))
                aws_general.follow_indices(starting_indices, 
                                           grid=grid, 
                                           states=states, 
                                           paths=data["paths"], 
                                           trajectory_hook=plotting, 
                                           verbose=args.verbose, 
                                           isinside=path_isinside)

                if lv.LAKE in matched_states:
                    if args.verbose < 2:
                        print("following lake inside of manageable region ...", end="", flush=True)
                    else:
                        print()
                        print("following LAKE points inside of manageable region")
                    starting_indices = [index for index in _starting_indices if states[index] == lv.LAKE]
                    plotting = lambda traj, choice: ax3d.plot3D(xs=traj[0], ys=traj[1], zs=traj[2],
                                                                color="green" if choice == 0 else "brown")
                    aws_general.follow_indices(starting_indices, 
                                               grid=grid, 
                                               states=states, 
                                               paths=data["paths-lake"], 
                                               fallback_paths=data["paths"] if args.paths_lake_fallback else None,
                                               trajectory_hook=plotting, 
                                               verbose=args.verbose, 
                                               isinside=path_isinside)

        if args.save_pic:
            print("saving to {} ... ".format(args.save_pic), end="", flush=True)
            fig.savefig(args.save_pic)
            print("done")

        sys.stdout.flush()
        sys.stderr.flush()
        plt.show()









