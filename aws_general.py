
import pickle
import signal
import sys

from pyviability import libviability as lv


version_info = __version_info__ = (0, 1)
version = __version__ = ".".join(map(str, version_info))

def dummy_plot(*args, **kwargs):
    pass

def dummy_isinside(x):
    return True

def follow_indices(starting_indices, *,
        grid, states, paths,
        plot=dummy_plot, isinside=dummy_isinside,
        fallback_paths=None,
        verbose = 0
        ):
    if verbose:
        print("starting points and states for paths:")
        for ind in starting_indices:
            print("{!s} --- {:>2}".format(grid[ind], states[ind]))
        print()
    plotted_indices = set()
    if verbose < 2:
        print("following and plotting paths ... ", end="", flush=True)
    for ind in starting_indices:
        if ind in plotted_indices:
            continue
        plotted_indices.add(ind)
        x0 = grid[ind]
        x1 = paths["reached point"][ind]
        if verbose >= 2:
            print("({}| {:>2d}) {} via {} ".format(ind, states[ind], x0, x1), end="")
        if isinside([x0, x1]):
            traj = list(zip(x0, x1))
            plot(traj, paths["choice"][ind])
            next_ind = paths["next point index"][ind]
            if next_ind == lv.PATHS_INDEX_DEFAULT and fallback_paths is not None:
                if verbose >= 2:
                    print("FALLBACK ", end="")
                next_ind = fallback_paths["next point index"][ind]
            if next_ind != lv.PATHS_INDEX_DEFAULT:
                if next_ind != ind:
                    if verbose >= 2:
                        print("to ({}| {:>2d}) {}".format(next_ind, states[next_ind], grid[next_ind]))
                    starting_indices.append(next_ind)
                elif verbose >= 2:
                    print("STAYING")
            elif verbose >= 2:
                print("NO INFO")
        elif verbose >= 2:
            print("OUTSIDE")
    if verbose < 2:
        print("done")


DEFAULT_HEADER = {
                "aws-version-info": (0, 1),
                "model": "AWS",
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
    print("reformatting: reading '{}' ... ".format(filename), end="", flush=True)
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


ALL_SIGNALS = { x: getattr(signal, x)  for x in dir(signal)
               if x.startswith("SIG")
               and not x.startswith("SIG_")  # because they are just duplicates
               and not getattr(signal, x) == 0  # can register only for signals >0
               and not getattr(signal, x) == 28 # SIGWINCH [28] is sent when resizing the terminal ...
               and not x in ["SIGSTOP", "SIGKILL"]  # can't register these because you can't actually catch them (:
               }
NUMBER_TO_SIGNAL = { val: key for key, val in ALL_SIGNALS.items() }

def signal_handler(sig, frame):
    sys.exit(sig)

def register_signals(sigs = set(ALL_SIGNALS), handler=signal_handler, verbose=True):
    """
    register a signal handler for all given signals
    sigs:       (set-like) providing all the signals to be registered
                default: all possible signals 'ALL_SIGNALS'
    handler:    (function) the signal handler to be used
                default: signal_handler, which just raises a 'sys.exit(sig)' for the signal 'sig'
    verbose:    (bool) print a notification to stderr if the signal registering failed
    """
    sigs = set(sigs)
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



