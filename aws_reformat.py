#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

from aws_general import __version__, __version_info__
import aws_general
import argparse, argcomplete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="update the format of AWS TSM result files")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="file with the (presumably) old format")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    for current_file in args.files:

        aws_general.reformat(current_file, verbose=1)




