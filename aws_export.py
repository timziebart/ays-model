#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

from __future__ import generators, print_function, division

from aws_general import __version__, __version_info__
import aws_general
import aws_model as aws
import pyviability as viab
from pyviability import libviability as lv

import numpy as np


# import sys
import os
import argparse, argcomplete
import pickle

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Export an AWS - TSM file to text.",
    )
    parser.add_argument("input_file", metavar="input-file",
                        help="file with the tsm data")
    parser.add_argument("txt_file", metavar="txt-file", nargs="?", default="",
                        help="output text file")
    parser.add_argument("-f", "--force", action="store_true",
                        help="overwrite text file if already existing")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.txt_file and (not args.force) :
        if os.path.isfile(args.txt_file):
            parser.error("'{}' exists already, use '--force' option to overwrite".format(args.txt_file))

    if args.txt_file == args.input_file:
        parser.error("'txt-file' and 'output-file' should be different from each other, not both '{}'".format(args.input_file))

    header, data = aws_general.load_result_file(args.input_file)

    header_txt = "#"*80 + "\n"
    header_txt += aws.recursive_dict2string(header)
    header_txt += "#"*80 + "\n"

    for region in lv.REGIONS:
        header_txt += "{} = {:>2d}\n".format(region, getattr(lv, region))
    header_txt += "#"*80

    states = data["states"]

    print(header_txt)

    if args.txt_file:
        print("saving to {!r} ... ".format(args.txt_file), end="", flush=True)
        np.savetxt(args.txt_file, states, fmt="%3i", header=header_txt, comments="")
        print("done")
