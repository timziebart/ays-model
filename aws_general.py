
from pyviability import libviability as lv

import numpy as np
import pickle
import signal
import sys
import warnings as warn


def versioninfo2version(v_info):
    return ".".join(map(str, v_info))

DEFAULT_VERSION_INFO = (0, 1)  # that's where it all started

version_info = __version_info__ = (0, 3)
version = __version__ = versioninfo2version(__version_info__)


"""
aws-file version changes:
0.3: added 'computation-status'
0.2: the first ones with actual versioning, adding 'paths-lake' if paths has been given
no version or 0.1: the stuff from the beginning
"""

def formatted_value(val):
    fmt = "!r"
    try:
        float(val)
    except (TypeError, ValueError):
        pass
    else:
        fmt = ":4.2e"
    return ("{"+fmt+"}").format(val)

def get_changed_parameters(pars, default_pars):
    changed_pars = {}
    for par, val in pars.items():
        if not par in default_pars:
            changed_pars[par] = (val, None)
        elif isinstance(default_pars[par], np.ndarray):
            if not np.allclose(default_pars[par], val):
                changed_pars[par] = (val, default_pars[par])
        elif default_pars[par] != val:
            changed_pars[par] = (val, default_pars[par])

    return changed_pars

def print_changed_parameters(pars, default_pars, prefix=""):
    model_changed_pars = get_changed_parameters(pars, default_pars)
    if model_changed_pars:
        if prefix:
            print(prefix)
        for par in sorted(model_changed_pars):
            print(("{} = {} (default: {})").format(par, *map(formatted_value, model_changed_pars[par])))
        print()

def recursive_dict2string(dic, prefix="", spacing=" "*4):
    ret = ""
    for key in sorted(dic):
        assert isinstance(key, str)
        ret += prefix + key + " = "
        if isinstance(dic[key], dict):
            ret += "{\n"
            ret += recursive_dict2string(dic[key], prefix=prefix+spacing, spacing=spacing)
            ret += "}\n"
        else:
            ret += formatted_value(dic[key]) + "\n"
    if ret:
        ret = ret[:-1]
    return ret

def dummy_hook(*args, **kwargs):
    pass

def dummy_isinside(x):
    return True

def follow_indices(starting_indices, *,
        grid, states, paths,
        trajectory_hook=dummy_hook, isinside=dummy_isinside,
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
            trajectory_hook(traj, paths["choice"][ind])
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


def reformat(filename, *, verbose=0):
    """load file and then update header and data"""
    header, data = load_result_file(filename, version_check=False, verbose=verbose)

    # the actually change of the format is done in _reformat
    header, data = _reformat(header, data, verbose=verbose)

    save_result_file(filename, header, data, verbose=verbose)


def save_result_file(fname, header, data, *, verbose=0):
    """save 'header' and 'data' to 'fname'"""
    try:
        _check_format(header, data)
    except AssertionError:
        warn.warn("the generated 'header' and 'data' failed at least one consistency check, saving anyway")

    if verbose:
        print("saving to {!r} ... ".format(fname), end="", flush=True)
    with open(fname, "wb") as f:
        pickle.dump((header, data), f)
    if verbose:
        print("done")


def load_result_file(fname, *, 
                     version_check=True,
                     consistency_check=True,
                     auto_reformat=False,
                     verbose=0
                     ):
    """loads the file 'fname' and performs some checks
    
    note that the options are interdependent: 'auto_reformat' needs 'consistency_check' needs 'version_check'
    """
    if verbose:
        print("loading {} ... ".format(fname), end="", flush=True)
    with open(fname, "rb") as f:
        header, data = pickle.load(f)
    if verbose:
        print("done", flush=True)
    if not version_check:
        return header, data
    if "aws-version-info" in header and header["aws-version-info"] == __version_info__:
        return header, data
    if auto_reformat:
        header, data = _reformat(header, data, verbose=verbose)
        return header, data
    raise IOError("please reformat the file (from version {} to {})".format(versioninfo2version(header.pop("aws-version-info", DEFAULT_VERSION_INFO)), __version__))

        




DEFAULT_HEADER = {
                "aws-version-info": DEFAULT_VERSION_INFO,
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
                "computation-status": "",
                }



def _check_format(header, data):
    """consistency checks"""

    assert header["aws-version-info"] == __version_info__

    # check the header contains the right keys
    if set(header) != set(DEFAULT_HEADER):
        print("maybe your header was orrupted:")
        new_header_unknown = set(header).difference(DEFAULT_HEADER)
        new_header_missing = set(DEFAULT_HEADER).difference(header.keys())
        if new_header_unknown:
            print("unknown keys: " + ", ".join(new_header_unknown))
        if new_header_missing:
            print("missing keys: " + ", ".join(new_header_missing))
        raise KeyError("header has not the proper key set")


    # keys for data
    data_mandatory_keys = ["grid", "states"]
    data_optional_keys  = ["paths", "paths-lake"]
    # check data contains all necessary keys
    assert set(data_mandatory_keys).issubset(data.keys())
    # check data contains not more than possible keys
    assert set(data_mandatory_keys + data_optional_keys).issuperset(data.keys())
    # check that paths and paths-lake only arise together
    assert len(set(["paths", "paths-lake"]).intersection(data.keys())) in [0, 2]


def _reformat(header, data, verbose=0):
    """updating header and data and check consistency"""

    if verbose:
        print("startin reformatting ... ", end="", flush=True)

    if "aws-version-info" not in header:
        header["aws-version-info"] = (0, 1)

    # 0.1 or no version given
    if header["aws-version-info"] == (0, 1):
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

        header = new_header

    # 0.2 add paths-lake if paths is given in data
    if header["aws-version-info"] < (0, 2):
        if "paths" in data and not "paths-lake" in data:
            data["paths-lake"] = np.array([])

    # 0.3 add computation-status
    if header["aws-version-info"] < (0, 3):
        header["computation-status"] = ""  # everything ran through

        # always at the last step
        # set the new version-info
        header["aws-version-info"] = __version_info__

    if verbose:
        print("checking consistency of new header and data ... ", end="", flush=True)
    _check_format(header, data)

    if verbose:
        print("done")

    return header, data






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



