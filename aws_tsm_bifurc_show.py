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

import matplotlib.pyplot as plt


FILE_ERROR_MESSAGE = "{!r} seems to be an older aws file version or not a proper aws file, please use the '--reformat' option"


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

    try:
        print("getting reference ... ", end="")
        reference_header, _ = aws_general.load_result_file(args.input_file, auto_reformat=args.reformat, verbose=1)
    except IOError:
        parser.error(FILE_ERROR_MESSAGE.format(args.input_file))

    # remove the bifurcation_parameter from the reference and check at the same time that it really was in there
    reference_header["model-parameters"].pop(bifurcation_parameter)
    print()

    # check correct parameters
    bifurcation_parameter_list = []
    for in_file in args.input_files:
        try:
            header, _ = aws_general.load_result_file(args.input_file, auto_reformat=args.reformat, verbose=1)
        except IOError:
            parser.error(FILE_ERROR_MESSAGE.format(args.input_file))
        # append the value of the bifurcation parameter to the list and check at the same time that it really was in there
        bifurcation_parameter_list.append(header.pop(bifurcation_parameter))
        
        for el in cmp_list:
            if isinstance(reference_header[el], dict):
                assert not aws_general.get_changed_parameters(reference_header[el], header[el])
            elif isinstance(reference_header[el]



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
                print("{} = {} <--> {} = {}".format(key, aws_general.formatted_value(pars[key]), default_key, aws_general.formatted_value(pars[default_key])))

    aws_general.print_changed_parameters(reference["model-parameters"], aws.AWS_parameters, prefix="changed model parameters:")
    aws_general.print_changed_parameters(reference["grid-parameters"], aws.grid_parameters, prefix="changed grid parameters:")
    aws_general.print_changed_parameters(reference["boundary-parameters"], aws.boundary_parameters, prefix="changed boundary parameters:")


    if args.verbose >= 2:
        viab.print_evaluation(states)

    if args.save_pic:
        print("saving to {} ... ".format(args.save_pic), end="", flush=True)
        fig.savefig(args.save_pic)
        print("done")

    sys.stdout.flush()
    sys.stderr.flush()
    plt.show()









