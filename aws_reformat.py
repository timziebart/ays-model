import argparse, pickle


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
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="update the format of an AWS TSM result file")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="file with the (presumably) old format")

    args =parser.parse_args()

    for current_file in args.files:

        print("reading '{}' ... ".format(current_file), end="", flush=True)
        with open(current_file, "rb") as f:
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

        new_header = dict(DEFAULT_HEADER)  # copy it, if several files are processed
        new_header.update(header)

        print("writing '{}' ... ".format(current_file), end="", flush=True)
        with open(current_file, "wb") as f:
            pickle.dump((new_header, data), f)
        print("done")



