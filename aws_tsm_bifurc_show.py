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
import itertools as it

import datetime as dt
import functools as ft
import os, sys

import matplotlib.pyplot as plt


FILE_ERROR_MESSAGE = "{!r} seems to be an older aws file version or not a proper aws file, please use the '--reformat' option"

TRANSLATION = {
        "sigma" : r"$\sigma$",
        "beta_DG" : r"$\beta_{GR}$",
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="show the TSM results of the AWS model")
    parser.add_argument("parameter", metavar="bifurcation-parameter",
                        help="the parameter which changes for the expected bifurcation")
    parser.add_argument("input_files", metavar="input-file", nargs="+",
                        help="input files with the contents from the TSM analysis")

    parser.add_argument("-s", "--save-pic", metavar="file", default="",
                        help="save the picture to 'file'")

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase verbosity can be used as -v, -vv ...")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    bifurcation_parameter = args.parameter

    for in_file in args.input_files:
        if not os.path.isfile(in_file):
            parser.error("can't find input file {!r}".format(in_file))

    cmp_list = [
        "model-parameters",
        "grid-parameters",
        "boundary-parameters",
        "managements",
        ]
    stacking_order = lv.REGIONS

    if args.verbose:
        print("stacking_order:", stacking_order)
        print()
        print("header comparison keys:", cmp_list)
        print()

    try:
        print("getting reference ... ", end="")
        reference_header, _ = aws_general.load_result_file(args.input_files[0], verbose=1)
    except IOError:
        parser.error(FILE_ERROR_MESSAGE.format(args.input_file))

    # remove the bifurcation_parameter from the reference and check at the same time that it really was in there
    reference_header["model-parameters"].pop(bifurcation_parameter)

    # check correct parameters
    bifurcation_parameter_list = []
    volume_lists = {r:[] for r in lv.REGIONS}
    for in_file in args.input_files:
        try:
            header, data = aws_general.load_result_file(in_file, verbose=1)
        except IOError:
            parser.error(FILE_ERROR_MESSAGE.format(args.input_file))
        # append the value of the bifurcation parameter to the list and check at the same time that it really was in there
        bifurcation_parameter_list.append(header["model-parameters"].pop(bifurcation_parameter))
        
        for el in cmp_list:
            if aws_general.recursive_difference(reference_header[el], header[el]):
                raise ValueError("incompatible headers")
        grid = np.asarray(data["grid"])
        states = np.asarray(data["states"])

        num_all = states.size

        for r in lv.REGIONS:
            volume_lists[r].append(np.count_nonzero(states == getattr(lv, r))/num_all)
    print()

    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = fig.add_subplot(111)

    argsort_param = np.argsort(bifurcation_parameter_list)
    bifurcation_parameter_list = np.asarray(bifurcation_parameter_list)[argsort_param]
    for key in volume_lists:
        volume_lists[key] = np.asarray(volume_lists[key])[argsort_param]

    y_before = np.zeros_like(volume_lists[key]) # using the key from the for-loop before
    for r in stacking_order:
        y_now = volume_lists[r] + y_before
        # mask = ~np.isclose(volume_lists[r], 0)
        ax.fill_between(
                bifurcation_parameter_list,#[mask], 
                y_before,#[mask], 
                y_now,#[mask], 
                facecolor=lv.COLORS[getattr(lv, r)], lw=2, edgecolor="white")
        y_before += volume_lists[r]

    ax.set_xlim(bifurcation_parameter_list[0], bifurcation_parameter_list[-1])
    ax.set_ylim(0, 1)
    xlabel = bifurcation_parameter
    if xlabel in TRANSLATION:
        xlabel = TRANSLATION[xlabel]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative volume in phase space")

    if args.save_pic:
        print("saving to {} ... ".format(args.save_pic), end="", flush=True)
        fig.savefig(args.save_pic)
        print("done")

    sys.stdout.flush()
    sys.stderr.flush()
    plt.show()









