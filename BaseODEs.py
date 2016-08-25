
import inspect

import numba as nb

import scipy.integrate as integ

def getParameterArgs(func):
    args, _, _, defaults = inspect.getargspec(func)
    assert len(args) >= 2, "your rhs function takes only %i arguments, but it should take at least x0 and t for odeint to work with it"%len(args)
    return args[2:]

NUMBA_NOPYTHON = True
NUMBA_NOGIL = True

class BaseODEs(object):
    def __init__(self,
            rhs,
            params,
            params_order = None,
            rhs_fast = None,
            rhs_PS = None,
            comment = "",
            ):

        self.comment = comment
        self._params = params
        self._rhs = rhs

        # generate Parameter argument order if not provided
        if params_order is None:
            self._params_order = getParameterArgs(rhs)
        else:
            self._params_order = params_order # generate automatically ?? # odeint doesn't support kwargs
        assert set(self._params_order).issubset(params), "please provide all necessary parameters given to the rhs function, even keyword arguments"

        # generated self._odeint_params
        self.__create_odeint_params()

        # numba compile rhs if rhs_fast not given
        if rhs_fast is None:
            self._rhs_fast = nb.jit(rhs, 
                    nopython = NUMBA_NOPYTHON,
                    nogil = NUMBA_NOGIL,
                    )
        elif rhs_fast == "Python":
            self._rhs_fast = rhs
        else:
            self._rhs_fast = rhs_fast

        # set _rhs_PS if not given
        if rhs_PS is None:
            self._rhs_PS = rhs
        else:
            self._rhs_PS = rhs_PS

    def __str__(self):
        ret = self.__class__.__name__
        if self.comment:
            ret += "[{}]".format(self.comment)
        ret += "("
        ret += ", ".join([ "{} : {!r}".format(par, self._params[par]) for par in self._params_order ])
        ret += ")"
        return ret

    def __repr__(self):
        return self.__str__

    def __create_odeint_params(self):
        self._odeint_params = tuple([self._params[par] for par in self._params_order])

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, val):
        self._params[key] = val
        self.__create_odeint_params()

    def rhs(self, x0, t = 0, params = None, use_fast = True):
        if params is None:
            params = self._params
        if use_fast:
            return self._rhs_fast(x0, t, **params)
        else:
            return self._rhs(x0, t, **params)

    def rhs_PS(self, x0, t = 0, params = None):
        if params is None:
            params = self._params
        return self._rhs_PS(x0, t, **params)


    def integrate(self, x0, t, **kwargs):
        return integ.odeint(self._rhs_fast, x0, t, args = self._odeint_params, **kwargs)

