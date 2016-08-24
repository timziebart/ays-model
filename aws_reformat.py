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
                "out-of-bounds": None,
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="update the format of an AWS TSM result file")
    parser.add_argument("in_file", metavar="in-file", type=str,
                        help="input file with the (presumably) old format")
    parser.add_argument("out_file", metavar="out-file", type=str, nargs="?", default="",
                        help="output file; if not provided, save to input file")

    args =parser.parse_args()

    print("reading '{}' ... ".format(args.in_file), end="", flush=True)
    if not args.out_file:
        args.out_file = args.in_file
    print("done")

    with open(args.in_file, "rb") as f:
        header, data = pickle.load(f)

    # management has been renamed with the plural
    if "management" in header:
        header["managements"] = header.pop("management")
    if not "boundary-parameters" in header:
        header["boundary-parameters"] = {
            "A_PB": header["model-parameters"].pop("A_PB"),
            "W_SF": header["model-parameters"].pop("W_SF"),
        }

    new_header = DEFAULT_HEADER
    new_header.update(header)

    print("writing '{}' ... ".format(args.in_file), end="", flush=True)
    with open(args.out_file, "wb") as f:
        pickle.dump((header, data), f)
    print("done")



