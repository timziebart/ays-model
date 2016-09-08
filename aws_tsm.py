#!/usr/bin/env python

import aws_model as aws

import PyViability as viab
import helper

import numpy as np

import time
import datetime as dt

import sys
import types
import argparse
import signal
import warnings as warn

import pickle

MANAGEMENTS = {
    "degrowth": "dg",
    "solar-radiation": "srm",
    "energy-transformation": "et",
    "carbon-capture-storage": "ccs",
}


ALL_SIGNALS = { x: getattr(signal, x)  for x in dir(signal) if x.startswith("SIG") }
NUMER_TO_SIGNAL = { val: key for key, val in ALL_SIGNALS.items() }
def signal_handler(signal, frame):  sys.exit(signal)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze the AWS model with TSM using the PyViability package.",
    )

    # positional arguments
    parser.add_argument("output_file", metavar="output-file",
                        help="output file where the TSM data is saved to")

    # optional arguments
    parser.add_argument("-b", "--no-backscaling", action="store_false", dest="backscaling",
                        help="do not backscale the result afterwards")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="do a dry run; perpare everything but then do not"
                        " actually run the TSM computation nor save a file")
    parser.add_argument("-e", "--eddies", action="store_true",
                        help="include eddies in the computation")
    parser.add_argument("-f", "--force", action="store_true",
                        help="if output-file exists already, overwrite it")
    parser.add_argument("-i", "--integrate", action="store_const",
                        dest="run_type", const="integration", default="linear",
                        help="integrate instead of using linear approx.")
    parser.add_argument("-n", "--no-save", action="store_true",
                        help="don't save the result")
    parser.add_argument("--num", type=int, default=aws.grid_parameters["n0"],
                        help="number of points per dimension for the grid")
    parser.add_argument("-r", "--remember", action="store_true",
                        help="remember already calculated points in a dict")

    # management arguments
    management_group = parser.add_argument_group("management options")
    [management_group.add_argument("--"+MANAGEMENTS[m], "--"+m, action="append_const",
                                  dest="managements", const=m,
                                  default=[])
                            for m in MANAGEMENTS]

    # add verbosity check
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase the output")

    # do the actual parsing of the arguments
    args = parser.parse_args()

    # register all possible signals
    for sig in ALL_SIGNALS:
        try:
            signal.signal(getattr(signal, sig), signal_handler)
        except Exception as e:
            print("ignoring signal registration: {} [{}] ({}: {!s})".format(sig, ALL_SIGNALS[sig], e.__class__.__name__, e))
    print()

    aws.grid_parameters["n0"] = args.num

    # a small hack to make all the parameters available as global variables
    aws.globalize_dictionary(aws.boundary_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters)


    # generate the grid, normalized to 1 in each dimension
    grid, scaling_vector, offset, x_step = viab.generate_grid(boundaries,
                                                         n0,
                                                         grid_type,
                                                         verbosity=args.verbosity)
    # viab.generate_grid sets stepsize, reset it here
    viab.STEPSIZE = 2 * x_step

    # generate the fitting states array
    states = np.zeros(grid.shape[:-1], dtype=np.int16)

    # mark the fixed point in infinity as shelter already
    states[ np.linalg.norm(grid - [0, 1, 1], axis=-1) < 0.5 ] = -viab.SHELTER

    run_args = [offset, scaling_vector]
    run_kwargs = dict(returning=args.run_type, remember=args.remember)

    default_run = viab.make_run_function(aws.AWS_rescaled_rhs,
                                         helper.get_ordered_parameters(aws._AWS_rhs, aws.AWS_parameters),
                                         *run_args, **run_kwargs)

    management_runs = []
    for m in args.managements:
        management_dict = dict(aws.AWS_parameters) # make a copy
        ending = "_" + MANAGEMENTS[m].upper()
        changed = False
        for key in management_dict:
            # choose the variables that are changed by the ending
            if key+ending in management_dict:
                changed = True
                management_dict[key] = management_dict[key+ending]
        if not changed:
            raise NameError("didn't find any parameter for management option "\
                            "'{}' (ending '{}')".format(m, ending))
        management_run = viab.make_run_function(aws.AWS_rescaled_rhs,
                                             helper.get_ordered_parameters(aws._AWS_rhs, management_dict),
                                             *run_args, **run_kwargs)
        management_runs.append(management_run)



    sunny = viab.scaled_to_one_sunny(aws.AWS_sunny, offset, scaling_vector)

    out_of_bounds = [[False, True],   # A still has A_max as upper boundary
                     [False, False],  # W compactified as w
                     [False, False]]  # S compactified as s

    start_time = time.time()
    print("started: {}".format(dt.datetime.fromtimestamp(start_time).ctime()))
    if not args.dry_run:
        try:
            viab.topology_classification(grid, states, [default_run], management_runs,
                                            sunny, grid_type=grid_type,
                                            compute_eddies=args.eddies,
                                            out_of_bounds=out_of_bounds,
                                            )
        except SystemExit as e:
            print()
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("interrupted by SystemExit or Signal {} [{}]".format(NUMER_TO_SIGNAL[e.args[0]], e.args[0]))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print()
    time_passed = time.time() - start_time

    print()
    print("run time: {!s}".format(dt.timedelta(seconds=time_passed)))
    print()

    if args.backscaling:
        grid = viab.backscaling_grid(grid, scaling_vector, offset)

    viab.print_evaluation(states)

    if not args.no_save:
        header = {
                "model": "AWS",
                "managements": args.managements,
                "boundaries": ["planetary-boundary"],
                "grid-parameters": aws.grid_parameters,
                "model-parameters": aws.AWS_parameters,
                "boundary-parameters": aws.boundary_parameters,
                "start-time": start_time,
                "run-time": time_passed,
                "viab-backscaling-done": args.backscaling,
                "viab-scaling-vector": scaling_vector,
                "viab-scaling-offset": offset,
                "input-args": args,
                "stepsize": viab.STEPSIZE,
                "xstep" : x_step,
                "out-of-bounds": out_of_bounds,
                }
        data = {"grid": grid,
                "states": states,
                }
        print("saving to {!r} ... ".format(args.output_file), end="", flush=True)
        if not args.dry_run:
            with open(args.output_file, "wb") as f:
                pickle.dump((header, data), f)
        print("done")












