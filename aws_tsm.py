#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import aws_model as aws

import pyviability as viab
from pyviability import helper
from pyviability import libviability as lv

import numpy as np
import scipy.optimize as opt

import time
import datetime as dt

import sys, os
import types
import ast
import argparse, argcomplete
import signal
import warnings as warn

import pickle


ALL_SIGNALS = { x: getattr(signal, x)  for x in dir(signal)
               if x.startswith("SIG")
               and not x.startswith("SIG_")  # because they are just duplicates
               and not getattr(signal, x) == 0  # can register only for signals >0
               and not getattr(signal, x) == 28 # SIGWINCH [28] is sent when resizing the terminal ...
               and not x in ["SIGSTOP", "SIGKILL"]  # can't register these because you can't actually catch them (:
               }
NUMER_TO_SIGNAL = { val: key for key, val in ALL_SIGNALS.items() }

def signal_handler(sig, frame):
    sys.exit(sig)

def register_signals(sigs = set(ALL_SIGNALS), handler=signal_handler, verbose=True):
    """
    register a signal handler for all given signals
    sigs:       (multiply iterable) providing all the signals to be registered
                default: all possible signals 'ALL_SIGNALS'
    handler:    (function) the signal handler to be used
                default: signal_handler, which just raises a 'sys.exit(sig)' for the signal 'sig'
    verbose:    (bool) print a notification if the signal registering failed
    """
    # register all possible signals
    for sig in ALL_SIGNALS:
        sigclass = getattr(signal, sig)
        signum = sigclass.value
        # the line below checks whether the signal has been given for
        # registering in the form of either the name, the signal class or the
        # signal number
        if set([sig, sigclass, signum]).intersection(sigs):
            try:
                signal.signal(getattr(signal, sig), signal_handler)
            except Exception as e:
                if verbose:
                    print("ignoring signal registration: [{:>2d}] {} (because {}: {!s})".format(ALL_SIGNALS[sig], sig, e.__class__.__name__, e), file=sys.stderr)


MANAGEMENTS = aws.MANAGEMENTS


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
    parser.add_argument("-p", "--set-parameter", nargs=2, metavar=("par", "val"),
                        action="append", dest="changed_parameters", default=[],
                        help="set a parameter 'par' to value 'val' "\
                        "(caution, eval is used for the evaluation of 'val'")
    parser.add_argument("--remember-computed", action="store_true",
                        help="remember already computed points in a dict")
    parser.add_argument("--record-paths", action="store_true",
                        help="record the paths, direction and default / management option used, "\
                        "so a path can be reconstructed")
    parser.add_argument("--stop-when-finished", default="all", metavar="region",
                        help="stop when the computation of 'region' is finished") 
    parser.add_argument("-z", "--zeros", action="store_true",
                        help="estimate the fixed point(s)")

    # management arguments
    management_group = parser.add_argument_group("management options")
    [management_group.add_argument("--"+MANAGEMENTS[m], "--"+m, action="append_const",
                                  dest="managements", const=m,
                                  default=[])
                            for m in MANAGEMENTS]

    # # add verbosity check
    # parser.add_argument("-v", "--verbosity", action="count", default=0,
                        # help="increase the output")

    verbosity = 2

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    # do the actual parsing of the arguments
    args = parser.parse_args()

    OUTPUT_FILE_SUFFIX = ".out"
    if not args.dry_run and not args.output_file.endswith(OUTPUT_FILE_SUFFIX):
        parser.error("please use the suffix '{}' for 'output-file' (reason is actually the '.gitignore' file)".format(OUTPUT_FILE_SUFFIX))

    if not (args.force or args.dry_run):
        if os.path.isfile(args.output_file):
            parser.error("'{}' exists already, use '--force' option to overwrite".format(args.output_file))

    print()

    aws.grid_parameters["n0"] = args.num

    print("managements: {}".format(", ".join(args.managements) if args.managements else "(None)"))
    print()

    if args.changed_parameters:
        print("parameter changing:")
        combined_parameters = dict(aws.AWS_parameters)
        combined_parameters.update(aws.grid_parameters)
        combined_parameters.update(aws.boundary_parameters)
        for par, val in args.changed_parameters:
            for d in [aws.AWS_parameters, aws.grid_parameters, aws.boundary_parameters]:
                if par in d:
                    try:
                        val2 = eval(val, combined_parameters)
                    except BaseException as e:
                        print("couldn't evaluate {!r} for parameter '{}' because of {}: {}".format(val, par, e.__class__.__name__, str(e)))
                        sys.exit(1)
                    print("{} = {!r} <-- {}".format(par, val2, val))
                    d[ par ] = val2
                    break
            else:
                parser.error("'{}' is an unknown parameter".format(par))
    print()

    # a small hack to make all the parameters available as global variables
    aws.globalize_dictionary(aws.boundary_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters)

    # generate the grid, normalized to 1 in each dimension
    grid, scaling_vector, offset, x_step = viab.generate_grid(boundaries,
                                                         n0,
                                                         grid_type,
                                                         verbosity=verbosity)
    lv.STEPSIZE = 2 * x_step * max([1, np.sqrt( n0 / 80 )])  # prop to 1 / sqrt(n0)
    print("stepsize / gridstepsize: {:<5.3f}".format(lv.STEPSIZE / x_step))
    print()

    # generate the fitting states array
    states = np.zeros(grid.shape[:-1], dtype=np.int16)

    # mark the fixed point in infinity as shelter already
    states[ np.linalg.norm(grid - [0, 1, 1], axis=-1) < 5 * x_step] = -lv.SHELTER

    run_args = [offset, scaling_vector]
    run_kwargs = dict(returning=args.run_type, remember=args.remember_computed)

    default_run = viab.make_run_function(aws.AWS_rescaled_rhs,
                                         helper.get_ordered_parameters(aws._AWS_rhs, aws.AWS_parameters),
                                         *run_args, **run_kwargs)

    print("recording-paths: {}".format(args.record_paths))
    print()

    if args.zeros:
        x0 = [0.5, 0.5, 0] # a, w, s
        # x0 = [aws.boundary_parameters["A_PB"], 0.5, 0] # A, w, s
        print(x0)
        print("fixed point(s) of default:")
        # below the '0' is for the time t
        print(opt.fsolve(aws.AWS_rescaled_rhs, x0,
                         args=(0., ) + helper.get_ordered_parameters(aws._AWS_rhs, aws.AWS_parameters)))
        print()


    management_runs = []
    for m in args.managements:
        management_dict = aws.get_management_parameter_dict(m, aws.AWS_parameters)
        management_run = viab.make_run_function(aws.AWS_rescaled_rhs,
                                             helper.get_ordered_parameters(aws._AWS_rhs, management_dict),
                                             *run_args, **run_kwargs)
        management_runs.append(management_run)
        if args.zeros:
            print("fixed point(s) of {}:".format(m))
            # below the '0' is for the time t
            print(opt.fsolve(aws.AWS_rescaled_rhs, x0,
                            args=(0., ) + helper.get_ordered_parameters(aws._AWS_rhs, management_dict)))
            print()

    sunny = viab.scaled_to_one_sunny(aws.AWS_sunny, offset, scaling_vector)

    # out_of_bounds = [[False, True],   # A still has A_max as upper boundary
                     # [False, False],  # W compactified as w
                     # [False, False]]  # S compactified as s

    out_of_bounds = False # in a, w, s representation, doesn't go out of bounds of [0, 1]^3 by definition

    register_signals()

    start_time = time.time()
    print("started: {}".format(dt.datetime.fromtimestamp(start_time).ctime()))
    print()
    if not args.dry_run:
        try:
            viab.topology_classification(grid, states, [default_run], management_runs,
                                            sunny, grid_type=grid_type,
                                            compute_eddies=args.eddies,
                                            out_of_bounds=out_of_bounds,
                                            remember_paths=args.record_paths,
                                            verbosity=verbosity,
                                            stop_when_finished=args.stop_when_finished,
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
                "stepsize": lv.STEPSIZE,
                "xstep" : x_step,
                "out-of-bounds": out_of_bounds,
                "remember-paths": args.record_paths,
                }
        data = {"grid": grid,
                "states": states,
                }
        if args.record_paths:
            data["paths"] = lv.PATHS
            data["paths-lake"] = lv.PATHS_LAKE
        print("saving to {!r} ... ".format(args.output_file), end="", flush=True)
        if not args.dry_run:
            with open(args.output_file, "wb") as f:
                pickle.dump((header, data), f)
        print("done")












