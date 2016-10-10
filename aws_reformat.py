#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import argparse, pickle, argcomplete


DEFAULT_HEADER = {"model": "AWS",
                "managements": "unknown",
                "boundaries": ["unknown"],
                "grid-parameters": {},
                "model-parameters": {},
                "boundary-parameters": {},
                "start-time": 0,
                "run-time": 0,
                "viab-backscaling-done": None,
                "viab-scaling-vector": None,
                "viab-scaling-offset": None,
                "input-args": None,
                "stepsize": 0.,
                "xstep" : 1.,
                "out-of-bounds": None,
                "remember-paths": False,
                }

def reformat(filename):
    print("reading '{}' ... ".format(filename), end="", flush=True)
    with open(filename, "rb") as f:
        header, data = pickle.load(f)
    print("done")

    # management has been renamed with the plural
    if "management" in header:
        header["managements"] = header.pop("management")
    if not "boundary-parameters" in header:
        header["boundary-parameters"] = {
            "A_PB": header["model-parameters"].pop("A_PB"),
            "W_SF": header["model-parameters"].pop("W_SF"),
        }

    if "paths" in data and isinstance(data["paths"], tuple):
        new_paths = {}
        new_paths["reached point"] = data["paths"][0]
        new_paths["next point index"] = data["paths"][1]
        new_paths["choice"] = data["paths"][2]
        data["paths"] = new_paths

    new_header = dict(DEFAULT_HEADER)  # copy it in case several files are processed
    new_header.update(header)

    print("writing '{}' ... ".format(filename), end="", flush=True)
    with open(filename, "wb") as f:
        pickle.dump((new_header, data), f)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="update the format of an AWS TSM result file")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="file with the (presumably) old format")

    # use argcomplete auto-completion
    argcomplete.autocomplete(parser)

    args =parser.parse_args()

    for current_file in args.files:

        reformat(current_file)




