#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import PyViability as viab

import aws_model as aws
import aws_show

import numpy as np
import pickle, argparse

import datetime as dt

import matplotlib.pyplot as plt

# import argcomplete  # didn't get it to work


def RegionName2Option(vname, style="long"):
    if style=="long":
        return vname.replace("_", "-").lower()
    elif style=="short":
        return "".join(map(lambda x: x[0], vname.lower().split("_")))
    raise ValueError("unkown style: {!r}".format(style))


# check that there are no short option used twice
_all_regions_short = list(map(lambda x: RegionName2Option(x, style="short"), viab.REGIONS))
assert len(_all_regions_short) == len(set(_all_regions_short))
del _all_regions_short

def get_changed_parameters(pars, default_pars):
    changed_pars = {}
    for par, val in pars.items():
        # print(val)
        if not par in default_pars:
            changed_pars[par] = (val, None)
        elif default_pars[par] != val:
            changed_pars[par] = (val, default_pars[par])

    return changed_pars


if __name__ == "__main__":

    parser = argparse.ArgumentParser("show the TSM results of the AWS model")
    parser.add_argument("input_file", metavar="input-file",
                        help="input file with the contents from the TSM analysis")

    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")

    regions_parser = parser.add_argument_group("regions", "choose which regions you want to be plotted")
    regions_parser.add_argument("--a", "--all", action="store_true", dest="all_regions",
                                help="plot all regions")
    [regions_parser.add_argument("--"+RegionName2Option(region, style="short"),
                        "--"+RegionName2Option(region),
                        dest="regions",
                        action="append_const", const=region)
                        for region in viab.REGIONS]

    args = parser.parse_args()

    if args.all_regions:
        args.regions = viab.REGIONS

    with open(args.input_file, "rb") as f:
        header, data = pickle.load(f)

    assert header["viab-backscaling-done"]

    grid = data["grid"]
    states = data["states"]
    del data

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
    print("boundaries: {}".format(", ".join(header["boundaries"])))
    print()
    print("stepsize / gridstepsize: {:<5.3f}".format(header["stepsize"] / header["xstep"]))
    print()
    print("points per dimension: {:4d}".format(header["grid-parameters"]["n0"]))
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

    if args.regions is None:
        print("no regions for plotting chosen")
    else:
        # a small hack to make all the parameters available as global variables
        # aws.globalize_dictionary(header["model-parameters"], module=aws)
        # aws.globalize_dictionary(header["grid-parameters"], module=aws)

        fig, ax3d = aws_show.create_figure(**header["grid-parameters"])

        ax_parameters = dict(header["boundary-parameters"])  # make a copy
        ax_parameters.update(header["grid-parameters"])
        aws_show.add_boundary(ax3d, **ax_parameters)

        for region in args.regions:
            region_num = getattr(viab, region)
            mask = (states == region_num)
            ax3d.plot3D(xs=grid[:, 0][mask], ys=grid[:, 1][mask], zs=grid[:, 2][mask],
                            color=viab.COLORS[region_num],
                        alpha=1/header["grid-parameters"]["n0"],
                        linestyle="", marker=".", markersize=30,
                        )

        if args.save_pic:
            print("saving to {} ... ".format(args.save_pic), end="", flush=True)
            fig.savefig(args.save_pic)
            print("done")
        plt.show()









