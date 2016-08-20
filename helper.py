
import contextlib as ctl

import os
import sys
import inspect


REMEMBERED = {}  # used by remembering decorator

def get_parameter_order(func):
    args, _, _, defaults = inspect.getargspec(func)
    assert len(args) >= 2, "your rhs function takes only %i arguments, but it "\
        "should take at least x0 and t for odeint to work with it" % len(args)
    return args[2:]


def get_ordered_parameters(func, parameter_dict):
    ordered_parameters = get_parameter_order(func)
    assert set(ordered_parameters).issubset(parameter_dict), "you did not " \
        "provide all parameters"
    return tuple([parameter_dict[par] for par in ordered_parameters])


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@ctl.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    http://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard
    # stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


class remembering(object):

    def __init__(self, remember=True):
        self.remember = remember

    def __call__(self, f):
        if not self.remember:
            return f

        global REMEMBERED
        REMEMBERED[f] = {}

        def remembering_f(p, stepsize):
            global REMEMBERED
            p_tuple = tuple(p)
            if p_tuple in REMEMBERED[f]:
                return REMEMBERED[f][p_tuple]
            p2 = f(p, stepsize)
            REMEMBERED[f][p_tuple] = p2
            return p2

        return remembering_f
