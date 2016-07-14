
from __future__ import print_function, division

import PTopologyL as topo

import helper

import periodic_kdtree as periodkdt

import numpy as np
import numpy.linalg as la

import numba as nb

import scipy.integrate as integ
import scipy.spatial as spat

import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings as warn

import itertools as it

# raise the odeing warning as an error because it indicates that we are at a
# fixed point (or the grid is not fine enough)
warn.filterwarnings("error", category=integ.odepack.ODEintWarning)

# these are automatically set during grid generation but need to be manually set
# when using own grid
BOUNDS_EPSILON = None # should be set during grid Generation
STEPSIZE = None

# some constants so the calculation does end
MAX_EVOLUTION_NUM = 20
MAX_ITERATION_EDDIES = 10
VERBOSE = 0


# The ones below are just used byt the default pre-calculation hook and the
# default state evaluation. They are just here so they are not used for
# something else.
KDTREE = None
STATES = None
BOUNDS = None
COORDINATES = None

# ---- states ----
# encode the different states as integers, so arrays of integers can be used
# later in numpy arrays (which are very fast on integers)
# None state should never be used as it is used to indicate out of bounds
UNSET = 0
SHELTER = 1
GLADE = 2
LAKE = 3
SUNNY_UP = 4
DARK_UP = 5
BACKWATERS = 6
SUNNY_DOWN = 7
DARK_DOWN = 8
SUNNY_EDDIES = 9
DARK_EDDIES = 10
SUNNY_ABYSS = 11
DARK_ABYSS = 12
TRENCH = 13

OTHER_STATE = 14  # used for computation reasons only


# ---- Colors ----
# identify the states with the corresponding colors in order to be consistent
# with the color definitions from the original paper
COLORS = {
        UNSET: "blue",
        -SHELTER: "blue",
        -GLADE: "blue",
        SHELTER: topo.cShelter,
        GLADE: topo.cGlade,
        LAKE: topo.cLake,
        SUNNY_UP: topo.cSunnyUp,
        DARK_UP: topo.cDarkUp,
        BACKWATERS: topo.cBackwaters,
        SUNNY_DOWN: topo.cSunnyDown,
        DARK_DOWN: topo.cDarkDown,
        SUNNY_EDDIES: topo.cSunnyEddie,
        DARK_EDDIES: topo.cDarkEddie,
        SUNNY_ABYSS: topo.cSunnyAbyss,
        DARK_ABYSS: topo.cDarkAbyss,
        TRENCH: topo.cTrench,
        }


def generate_2Dgrid(xmin, xmax, x_num, ymin, ymax):
    x_len = xmax - xmin
    y_len = ymax - ymin
    x_step = max(x_len, y_len) / x_num
    x_half_step = x_step / 2
    x = np.linspace(xmin, xmax, x_num + 1)
    x = (x[:-1] + x[1:]) / 2
    y = np.linspace(ymin, ymax, x_num + 1)
    y = (y[:-1] + y[1:]) / 2
    xy = np.asarray(np.meshgrid(x, y))
    xy = np.rollaxis(xy, 0, 3)
    return [x_step, x_half_step, xy]


def hexGridSeriesGen(dim):
    a = 1/4  # comes straight from the calculation
    yield 1  # 1 is set as inital condition
    for d in range(1, dim):
        yield np.sqrt( (1-(-a)**(d+1)) / (1 + a) )


def generate_grid(boundaries, n0, grid_type, periodicity = [], verbosity = True):
    global MAX_NEIGHBOR_DISTANCE, BOUNDS_EPSILON, STEPSIZE, x_step

    assert grid_type in ["simplex-based", "orthogonal"], "unkown grid type '{!s}'".format(grid_type)

    boundaries = np.asarray(boundaries)
    periodicity = np.asarray(periodicity)

    dim = boundaries.shape[0]
    offset = boundaries[:,0]
    scaling_factor = boundaries[:,1] - boundaries[:,0]

    if not periodicity.size:
        periodicity = - np.ones((dim,))

    assert periodicity.shape == (dim,), "given boundaries do not match periodicity input"

    if grid_type in ["orthogonal"]:
        grid_prep_aperiodic = np.linspace(0, 1, n0)
        grid_prep_periodic = np.linspace(0, 1, n0-1, endpoint=False)
        # the last point is not set as it would be the same as the first one in
        # a periodic grid
        grid_args = [grid_prep_periodic if periodicity[d] > 0 else grid_prep_aperiodic for d in range(dim)]

        # create the grid
        grid = np.asarray(np.meshgrid(*grid_args))

        # move the axis with the dimenion to the back
        grid = np.rollaxis(grid, 0, dim + 1)

        x_step = grid_prep_periodic[1]
        assert x_step == grid_prep_aperiodic[1], "bug?"
        MAX_NEIGHBOR_DISTANCE = 1.5 * x_step
        BOUNDS_EPSILON = 0.1 * x_step
        STEPSIZE = 1.5 * x_step
    elif grid_type in ["simplex-based"]:
        if dim > 2:
            raise NotImplementedError("the current implementation is not correct for dimension >2")

        Delta_0 = 1/n0
        eps = Delta_0 / 3 # introduced so in the first line there are exactly n0 and there are no shifts over 1 in any dimension afterwards
        Delta_all = Delta_0 * np.array(list(hexGridSeriesGen(dim)))

        x_step = Delta_0 # Delta_0 is side length of the simplices

        grid = np.array(np.meshgrid( *[ np.arange(0, 1 - eps , Delta_d) for Delta_d in Delta_all ], indexing = "ij" ))
        grid = np.rollaxis( grid, 0, grid.ndim )

        assert len(grid.shape) == dim + 1
        assert grid.shape[-1] == dim

        # this is the problematic stuff, only true for dim == 2
        shifts = Delta_all / 2

        all_slice = slice(0, None)
        jump_slice = slice(1, None, 2)

        for d in range(1, dim): # starting at 1, because in dimension 0 nothing changes anyway
            # in the last dimension, shift every second line
            slicelist = [all_slice] * dim
            slicelist[d] = jump_slice
            slicelist += (slice(0,d), )
            grid[tuple(slicelist)] += shifts[:d]

        # when recursively going through, then add the direct neighbors only
        MAX_NEIGHBOR_DISTANCE = 1.01 * Delta_0
        BOUNDS_EPSILON = 0.1 * Delta_0
        STEPSIZE = 1.5 *Delta_0 # seems to be correct

    # flattening the array
    grid = np.reshape(grid, (-1, dim))

    if verbosity:
        print("created {:d} points".format(grid.shape[0]))

    return grid, scaling_factor, offset, x_step


def hexGrid(boundaries, n0, verb = False):
    """
    boundaries = array with shape (d, 2), first index for dimension, second index for minimal and maximal values
    """
    global MAX_NEIGHBOR_DISTANCE, x_step, MAX_FINAL_DISTANCE, BOUNDS_EPSILON, STEPSIZE
    boundaries = np.asarray(boundaries)
    dim = boundaries.shape[0]
    offset = boundaries[:,0]
    print(boundaries)
    scaling_factor = boundaries[:,1] - boundaries[:,0]

    Delta_0 = 1/n0
    x_step = Delta_0 # Delta_0 is side length of the simplices
    eps = Delta_0 / 3 # introduced so in the first line there are exactly n0 and there are no shifts over 1 in any dimension afterwards

    # print(np.array(list(hexGridSeriesGen(dim))))
    Delta_all = Delta_0 * np.array(list(hexGridSeriesGen(dim)))
    # print(Delta_all)
    # args = [ np.arange(0, 1 - eps , Delta_d) for Delta_d in Delta_all ]
    # print(len(args))
    # print(list(map(len, args)))
    # print(args)
    # grid = np.asarray(np.meshgrid( *args , indexing = "ij"))
    # assert False
    grid = np.array(np.meshgrid( *[ np.arange(0, 1 - eps , Delta_d) for Delta_d in Delta_all ], indexing = "ij" ))
    grid = np.rollaxis( grid, 0, grid.ndim )
    assert len(grid.shape) == dim + 1
    assert grid.shape[-1] == dim
    # print(grid[:,:,1])
    # print( grid.shape )
    if verb:
        print("created n = %i points"%np.prod(grid.shape[:-1]))

    shifts = Delta_all / 2
    # print(shifts)

    assert len(grid.shape) == dim + 1, "something is strange"

    all_slice = slice(0, None)
    jump_slice = slice(1, None, 2)
    for d in range(1, dim): # starting at 1, because in dimension 0 nothing changes anyway
        slicelist = [all_slice] * dim
        slicelist[d] = jump_slice
        slicelist += (slice(0,d), )
        # print(slicelist)
        # print(grid.shape, grid[tuple(slicelist)].shape)
        grid[tuple(slicelist)] += shifts[:d]
        # grid[tuple(slicelist)][:d] = shifts[:d]

    # flatten the array
    grid = np.reshape(grid, (-1, dim))
    # print(grid)

    MAX_NEIGHBOR_DISTANCE = 1.01 * Delta_0
    # MAX_FINAL_DISTANCE = 0.7 * Delta_0
    BOUNDS_EPSILON = 0.1 * Delta_0
    STEPSIZE = 1.5 * Delta_0
    warn.warn("proper estimation of MAX_FINAL_DISTANCE is still necessary")

    return grid, scaling_factor, offset, x_step


def normalized_grid(boundaries, x_num):
    """generates a normalized grid  in any dimension and gets the scaling factors and linear shift of each axis"""
    global MAX_NEIGHBOR_DISTANCE, x_step, BOUNDS_EPSILON, STEPSIZE

    dim = int(len(boundaries)/2)

    scaling_factor = np.ones(dim)

    offset = np.zeros(dim)

    for index in range(0, dim):
        scaling_factor[index] = boundaries[index + dim] - boundaries[index]

        if boundaries[index] != 0:
            offset[index] = boundaries[index]

    grid_prep = np.linspace(0, 1, x_num)
    # grid_prep = np.linspace(0, 1, x_num + 1)
    # grid_prep = (grid_prep[:-1] + grid_prep[1:]) / 2

    meshgrid_arg = [grid_prep for _ in range(dim)]

    grid = np.asarray(np.meshgrid(*meshgrid_arg))

    grid = np.rollaxis(grid, 0, dim + 1)

    # reshaping coordinates and states in order to use kdtree
    grid = np.reshape(grid, (-1, np.shape(grid)[-1]))

    x_step = 1/(x_num-1)

    MAX_NEIGHBOR_DISTANCE = 1.5 * x_step
    # MAX_FINAL_DISTANCE = np.sqrt(dim) * x_step / 2
    BOUNDS_EPSILON = 0.1 * x_step
    STEPSIZE = 1.5 * x_step

    return grid, scaling_factor, offset, x_step


def viability_single_point(coordinate_index, coordinates, states, stop_states, succesful_state, else_state,
                           evolutions, state_evaluation):
    """Calculate whether a coordinate with value 'stop_value' can be reached from 'coordinates[coordinate_index]'."""

    start = coordinates[coordinate_index]
    start_state = states[coordinate_index]

    global VERBOSE
    # VERBOSE = (coordinate_index == (10 * 80 - 64,))
    # VERBOSE = la.norm(start - np.array([1,0.1])) < 0.05
    # VERBOSE = VERBOSE or la.norm(start - np.array([0.1, 0.606])) < 0.02
    # VERBOSE = True

    if VERBOSE:
        print()

    for evol_num, evol in enumerate(evolutions):
        point = start

        for _ in range(MAX_EVOLUTION_NUM):
            point = evol(point, STEPSIZE)

            if np.max(np.abs(point - start)) < x_half_step:
                # not yet close enough to a different point
                # so run the evolution function again
                continue # not yet close enough to another point

            final_state = state_evaluation(point)

            if final_state in stop_states: # and constraint(point) and final_distance < MAX_FINAL_DISTANCE:

                if VERBOSE:
                    print( "%i:"%evol_num, coordinate_index, start, start_state, "-->", final_state )
                return succesful_state

            # break and run the other evolutions to check whether they can reach a point with 'stop_state'
            if VERBOSE:
                print("%i:"%evol_num, coordinate_index, start, start_state, "## break")
            break

    # didn't find an option leading to a point with 'stop_state'
    if VERBOSE:
        print("all:", coordinate_index, start, start_state, "-->", else_state)
    return else_state


def state_evaluation_kdtree(point):
    if np.any( BOUNDS[:,0] > point ) or np.any( BOUNDS[:,1] < point ) :  # is the point out-of-bounds?
        return None  # "out-of-bounds state"
    final_distance, tree_index = KDTREE.query(point, 1)
    # if final_distance > MAX_FINAL_DISTANCE:  # <-- deprecated
        # return None
    return STATES[tree_index]


def create_kdtree(coordinates, states, is_sunny, periodicity):
    global KDTREE, STATES, BOUNDS
    STATES = states
    periodicity = np.asarray(periodicity)
    # check, if there are periodic boundaries and if so, use different tree form
    if not periodicity.size:
        KDTREE = spat.cKDTree(coordinates)

    else:
        assert np.shape(coordinates)[-1] == len(periodicity), "Given boundaries don't match with " \
                                                                            "dimensions of coordinates. " \
                                                                            "Write '-1' if boundary is not periodic!"
        KDTREE = periodkdt.PeriodicCKDTree(periodicity, coordinates)
    dim = coordinates.shape[-1]
    BOUNDS = np.zeros((dim, 2))
    for d in range(dim):
        if periodicity.size and periodicity[d] > 0:
            BOUNDS[d,:] = -np.inf, np.inf
            # this basically means, because of periodicity, the trajectories
            # cannot run out-of-bounds
        else:
            BOUNDS[d,:] = np.min(coordinates[:,d]) - BOUNDS_EPSILON, np.max(coordinates[:,d]) + BOUNDS_EPSILON


def viability_kernel_step(coordinates, states, good_states, bad_state, succesful_state, work_state,
                          evolutions, state_evaluation):
    """do a single step of the viability calculation algorithm by checking which points stay immediately within the good_states"""

    changed = False

    shape = coordinates.shape[:-1]

    for base_index in np.ndindex(shape):
        neighbors = [base_index]

        for index in neighbors: # iterate over the base_index and, if any changes happened, over the neighbors, too
            old_state = states[index]

            if old_state == work_state:
                new_state = viability_single_point(index, coordinates, states, good_states, succesful_state, bad_state,
                                                   evolutions, state_evaluation)

                if new_state != old_state:
                    changed = True
                    states[index] = new_state
                    # get_neighbor_indices(index, shape, neighbor_list = neighbors)
                    get_neighbor_indices_via_cKD(index,  neighbor_list=neighbors)

    return changed


def get_neighbor_indices_via_cKD(index, neighbor_list=[]):
    """extend 'neighbor_list' by 'tree_neighbors', a list that contains the nearest neighbors found trough cKDTree"""

    index = np.asarray(index).astype(int)

    tree_neighbors = KDTREE.query_ball_point(KDTREE.data[index].flatten(), MAX_NEIGHBOR_DISTANCE)
    tree_neighbors = [(x,) for x in tree_neighbors if not x in neighbor_list]

    neighbor_list.extend(tree_neighbors)

    return neighbor_list


def get_neighbor_indices(index, shape, neighbor_list = []):
    """append all neighboring indices of 'index' to 'neighbor_list' if they are within 'shape'"""

    index = np.asarray(index)
    shape = np.asarray(shape)

    for diff_index in it.product([-1, 0, 1], repeat = len(index)):
        diff_index = np.asarray(diff_index)
        new_index = index + diff_index

        if np.count_nonzero(diff_index) and np.all( new_index >= 0 ) and np.all( new_index < shape ):
            neighbor_list.append(tuple(new_index))

    return neighbor_list


def viability_kernel(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions,
                     periodic_boundaries = [],
                     pre_calculation_hook = create_kdtree,  # None means nothing to be done
                     state_evaluation = state_evaluation_kdtree,
                     ):
    """calculate the viability kernel by iterating through the viability kernel steps
    until convergence (no further change)"""
    # assert coordinates.shape[:-1] == states.shape[:-1], "'coordinates' and 'states' don't match in shape"

    assert "x_step" in globals() # needs to be set by the user for now ... will be changed later
    assert "BOUNDS_EPSILON" in globals() # needs to be set by the user for now ... will be changed later
    # assert "MAX_FINAL_DISTANCE" in globals() # needs to be set by the user for now ... will be changed later
    assert "MAX_NEIGHBOR_DISTANCE" in globals() # needs to be set by the user for now ... will be changed later
    global x_half_step
    x_half_step = x_step/2
    if not "STEPSIZE" in globals():
        global STEPSIZE
        # fix stepsize on that for now if nothing else has been given by the
        # user
        STEPSIZE = 2 * x_step

    if not pre_calculation_hook is None:
        # run the pre-calculation hook (defaults to creation of the KD-Tree)
        pre_calculation_hook(coordinates, states, None, periodic_boundaries)

    # actually only on step is needed due to the recursive checks (i.e. first
    # checking all neighbors of a point that changed state)
    return viability_kernel_step(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, state_evaluation)


def viability_capture_basin(coordinates, states, target_states, reached_state, bad_state, work_state, evolutions,
                    pre_calculation_hook = create_kdtree,  # None means nothing to be done
                    state_evaluation = state_evaluation_kdtree,
                    ):
    """reuse the viability kernel algorithm to calculate the capture basin"""

    if work_state in states and any( ( target_state in states for target_state in target_states) ):
        # num_work = np.count_nonzero(work_state == states)
        viability_kernel(coordinates, states, target_states + [reached_state], work_state, reached_state,
                                     work_state, evolutions, pre_calculation_hook = pre_calculation_hook ,state_evaluation = state_evaluation)
        # changed = (num_work == np.count_nonzero(reached_state == states))
    else:
        print("empty work or target set")
        # changed = False
    # all the points that still have the state work_state are not part of the capture basin and are set to be bad_states
    changed = (work_state in states)
    states[ states == work_state ] = bad_state
    return changed

# below are just helper functions


def plot_points(coords, states):
    """plot the current states in the viability calculation as points"""

    assert set(states.flatten()).issubset(COLORS)
    for color_state in COLORS:
        plt.plot(coords[ states == color_state , 0], coords[ states == color_state , 1], color = COLORS[color_state],
                 linestyle = "", marker = ".", markersize = 30 ,zorder=0)


def plot_areas(coords, states):
    """plot the current states in the viability calculation as areas"""

    states = states.flatten()
    assert set(states).issubset(COLORS)
    coords = np.reshape(coords, states.shape + (-1,))
    x, y = coords[:,0], coords[:,1]

    color_states = sorted(COLORS)
    cmap = mpl.colors.ListedColormap([ COLORS[state] for state in color_states ])
    bounds = color_states[:1] + [ state + 0.5 for state in color_states[:-1]] + color_states[-1:]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.tripcolor(x, y, states, cmap = cmap, norm = norm, shading = "gouraud")


def make_run_function(rhs,
                      ordered_params,
                      offset,
                      scaling_factor,
                      returning = "run-function",
                      remember = True
                      ):

    #----------- just for 2D Phase-Space-plot to check the scaled right-hand-side
    def rhs_scaled_to_one_PS(x0, t):
        """\
rescales space only, because that should be enough for the phase space plot
"""
        # x = np.zeros_like(x0)
        # x[0] = scaling_factor[0] * x0[0] + offset[0]
        # x[1] = scaling_factor[1] * x0[1] + offset[1]
        val = rhs(scaling_factor[:,None, None] * x0 + offset[:, None, None], t, *ordered_params)  # calculate the rhs
        val /= scaling_factor[: ,None , None]
        # val[0] /= scaling_factor[0]
        # val[1] /= scaling_factor[1]
        return val
    # ---------------------------------------------------------------------------


    @nb.jit
    def rhs_scaled_to_one(x0, t, *args):
        x = scaling_factor * x0 + offset
        val = rhs(x, t, *args) / scaling_factor # calculate the rhs
        return val


    @nb.jit
    def trajectory_length_normalized_rhs(x0, t, *args):
        val = rhs_scaled_to_one(x0, t, *args)  # calculate the rhs
        return val / np.sqrt(np.sum(val ** 2, axis=-1))  # normalize it

    @nb.jit
    def distance_normalized_rhs(x, lam, x0, *args):
        val = rhs_scaled_to_one(x, lam, *args)  # calculate the rhs
        if lam == 0:
            return val / np.sqrt(np.sum( val ** 2, axis=-1) )
        return val * lam / np.sum( (x-x0) * val, axis=-1)

    @helper.remembering(remember = remember)
    def model_run(p, stepsize):
        if VERBOSE:
            integ_time = np.linspace(0, stepsize, 100)
        else:
            integ_time = [0, stepsize]
        try:
            with helper.stdout_redirected():
                traj = integ.odeint(distance_normalized_rhs, p, integ_time,
                                    args = (p,) + ordered_params,
                                    printmessg = False
                                    )
            if np.any(np.isnan(traj[-1])): # raise artifiially the warning if nan turns up
                raise integ.odepack.ODEintWarning("got a nan")
        except integ.odepack.ODEintWarning:
            warn.warn("got an integration warning; assume {!s} to be a stable fixed point".format(p),
                      category=RuntimeWarning)
            return p
        if VERBOSE:
            plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=3)

        return traj[-1]

    if returning == "run-function":
        return model_run
    elif returning == "PS":
        return rhs_scaled_to_one_PS


def scaled_to_one_sunny(is_sunny, offset, scaling_factor):

    @nb.jit
    def scaled_sunny(grid):
        new_grid = grid * scaling_factor + offset
        val = is_sunny(new_grid)  # calculate the rhs
        return val  # normalize it

    return scaled_sunny


def make_run_function2(model_object, timestep,
        backend = "odeint"
        ):
    """"""
    if backend == "simple":

        def model_run(p):
            v = model_object.eval(*p)
            v_norm = la.norm(v)
            # resize all steps that might be too long
            if v_norm > x_step:
                v *= x_step / v_norm
            p2 = p + v
            plt.plot([p[0], p2[0]], [p[1], p2[1]], color = "blue", linewidth = 5)
            return p2
    elif backend == "odeint":

        def model_run(p):
            model_object.setInitialCond(p)
            t, traj = model_object.run(0, timestep, steps = 1e2)
            final_index = trajectory_length_index(traj, 1.5 * x_step)
            traj = traj[:final_index]
##             if length > 1.5*x_step: # better: use a random walk criteria to find the factor (and a possible exponent)
##             # shorten the trajectory until length is roughly x_step
            if VERBOSE:
                plt.plot(traj[:, 0], traj[:, 1], color = "red", linewidth = 3)
            #else:
             #   plt.plot(traj[:, 0], traj[:, 1], color = "blue", linewidth = 5)
            return traj[-1]
    elif backend == "new":

        def model_run(p):
            t = np.linspace(0, timestep, 1e2)
            traj = model_object.integrate(p, t)
            ##             final_index = trajectory_length_index(traj, 1.5 * x_step)
            ##             traj = traj[:final_index]
            length = trajectory_length(traj)
            if length > 1.5 * x_step:  # better: use a random walk criteria to find the factor (and a possible exponent)
                # shorten the trajectory until length is roughly x_step
                traj = traj[:int(traj.shape[0] * 1.5 * x_step / length)]
            if VERBOSE:
                plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=3)
                ##             else:
                ##                 plt.plot(traj[:, 0], traj[:, 1], color = "blue", linewidth = 1)
            return traj[-1]
    else:

        raise NotImplementedError("backend = %s does not exist"%repr(backend))

    return model_run


def trajectory_length(traj):
    return np.sum( la.norm( traj[1:] - traj[:-1], axis = -1) )


def trajectory_length_index(traj, target_length):
    lengths = np.cumsum( la.norm( traj[1:] - traj[:-1], axis = -1) )

    if target_length < lengths[-1]:
        return traj.shape[0] # incl. last element
    index_0, index_1 = 0, traj.shape[0] - 1

    while not index_0 in [index_1, index_1 - 1]:
        middle_index = int( (index_0 + index_1)/2 )

        if lengths[middle_index] <= target_length:
            index_0 = middle_index
        else:
            index_1 = middle_index

    return index_1


def backscaling_grid(grid, scalingfactor, offset):
    return grid * scalingfactor + offset


def topology_classification(coordinates, states, default_evols, management_evols, is_sunny,
                            periodic_boundaries = [],
                            upgradeable_initial_states = False,
                            compute_eddies = True,
                            pre_calculation_hook = create_kdtree,  # None means nothing to be done
                            state_evaluation = state_evaluation_kdtree,
                            ):
    """calculates different regions of the state space using viability theory algorithms"""

    # upgreading initial states to higher
    if upgradeable_initial_states:
        raise NotImplementedError("upgrading of initally given states not yet implemented")

    if not pre_calculation_hook is None:
        # run the pre-calculation hook (defaults to creation of the KD-Tree)
        pre_calculation_hook(coordinates, states, is_sunny, periodic_boundaries)

    # make sure, evols can be treated as lists
    default_evols = list(default_evols)
    management_evols = list(management_evols)

    all_evols = management_evols + default_evols

    # better remove this and use directly the lower level stuff, see issue #13
    viability_kwargs = dict(
        pre_calculation_hook = None,
        state_evaluation = state_evaluation,
    )

    shelter_empty = False
    backwater_empty = False

    # calculate shelter
    print('###### calculating shelter')
    states[(states == UNSET) & is_sunny(coordinates)] = SHELTER # initial state for shelter calculation
    # viability_kernel(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, **viability_kwargs)
    viability_kernel(coordinates, states, [SHELTER, -SHELTER], UNSET, SHELTER, SHELTER, default_evols, **viability_kwargs)

    if not np.any(states == SHELTER):
        print('shelter empty')
        shelter_empty = True

    if not shelter_empty:
        # calculate glade
        print('###### calculating glade')

        states[(states == UNSET) & is_sunny(coordinates)] = SUNNY_UP

        #viability_capture_basin(coordinates, states, target_states, reached_state, bad_state, work_state, evolutions, **viability_kwargs):
        viability_capture_basin(coordinates, states, [SHELTER, -SHELTER], GLADE, UNSET, SUNNY_UP, all_evols, **viability_kwargs)

        # calculate remaining upstream dark and sunny
        print('###### calculating rest of upstream (lake, dark and sunny)')
        states[(states == UNSET)] = DARK_UP
        viability_capture_basin(coordinates, states, [SHELTER, GLADE, -SUNNY_UP, -DARK_UP, -LAKE], SUNNY_UP, UNSET, DARK_UP, all_evols, **viability_kwargs)

        states[~is_sunny(coordinates) & (states == SUNNY_UP)] = DARK_UP

        # calculate Lake
        print('###### calculating lake')
        states[is_sunny(coordinates) & (states == SUNNY_UP)] = LAKE
        viability_kernel(coordinates, states, [SHELTER, GLADE, LAKE, -LAKE], SUNNY_UP, LAKE, LAKE, all_evols, **viability_kwargs)

    # calculate Bachwater
    print('###### calculating backwater')
    states[is_sunny(coordinates) & (states == UNSET)] = BACKWATERS
    viability_kernel(coordinates, states, [BACKWATERS, -BACKWATERS], UNSET, BACKWATERS, BACKWATERS, all_evols, **viability_kwargs)

    if not np.any(states == BACKWATERS):
        print('backwater empty')
        backwater_empty = True

    if not backwater_empty:
        # calculate remaining downstream dark and sunny
        print('###### calculating remaining downstream (dark and sunny)')
        states[(states == UNSET)] = DARK_DOWN
        viability_capture_basin(coordinates, states, [BACKWATERS, -SUNNY_DOWN, -DARK_DOWN], SUNNY_DOWN, UNSET, DARK_DOWN, all_evols, **viability_kwargs)
        states[~is_sunny(coordinates) & (states == SUNNY_DOWN)] = DARK_DOWN




    # calculate trench and set the rest as preliminary estimation for the eddies
    print('###### calculating dark Eddies/Abyss')
    states[is_sunny(coordinates) & (states == UNSET)] = SUNNY_EDDIES

    # look only at the coordinates with state == UNSET
    viability_capture_basin(coordinates, states,
                            [SHELTER, GLADE, SUNNY_UP, DARK_UP, LAKE, BACKWATERS, SUNNY_DOWN, SUNNY_EDDIES, SUNNY_ABYSS, -SHELTER, -GLADE, -SUNNY_UP, -DARK_UP , -LAKE, -BACKWATERS, -SUNNY_DOWN, -SUNNY_EDDIES, -SUNNY_ABYSS],
                            DARK_EDDIES, TRENCH, UNSET, all_evols, **viability_kwargs)
    if compute_eddies:

        # the preliminary estimations for sunny and dark eddie are set
        states[(states == SUNNY_EDDIES)] = UNSET
        viability_capture_basin(coordinates, states,
                                [DARK_EDDIES, -DARK_EDDIES],
                                SUNNY_EDDIES, SUNNY_ABYSS, UNSET, all_evols, **viability_kwargs)


        for num in range(MAX_ITERATION_EDDIES):
            states[(states == DARK_EDDIES)] = UNSET
            changed = viability_capture_basin(coordinates, states,
                                    [SUNNY_EDDIES, -SUNNY_EDDIES],
                                            DARK_EDDIES, DARK_ABYSS, UNSET, all_evols, **viability_kwargs)
            if not changed:
                break
            states[(states == SUNNY_EDDIES)] = UNSET
            changed = viability_capture_basin(coordinates, states,
                                    [DARK_EDDIES, -DARK_EDDIES],
                                    SUNNY_EDDIES, SUNNY_ABYSS, UNSET, all_evols, **viability_kwargs)
            if not changed:
                break
        else:
            warn.warn("reached MAX_ITERATION_EDDIES = %i during the Eddies calculation"%MAX_ITERATION_EDDIES)
    else:
        # assume all eddies are abysses
            states[(states == SUNNY_EDDIES)] = SUNNY_ABYSS
            states[(states == UNSET)] = DARK_ABYSS


    # All initially given states are set to positive counterparts
    states[(states < UNSET)] *= -1

    return states


