
from __future__ import print_function, division

import PTopologyL as topo

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

MAX_EVOLUTION_NUM = 20
VERBOSE = 0

COLORS = {
        0: "red",
        -1: "red",
        -2: "red",
        1: topo.cShelter,
        2: topo.cGlade,
        3: topo.cSunnyUp,
        4: topo.cDarkUp,
        5: topo.cLake,
        6: topo.cBackwaters,
        7: topo.cSunnyDown,
        8: topo.cDarkDown,
        9: topo.cSunnyAbyss,
        10: topo.cDarkAbyss,
        11: topo.cTrench,
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
    a = 1/4 # comes straight from the calculation
    yield 1 # 1 is set as inital condition
    for d in range(1, dim):
        yield np.sqrt( (1-(-a)**(d+1)) / (1 + a) )


def hexGrid(boundaries, n0, verb = False):
    """
    boundaries = list with shape (d, 2), first index for dimension, second index for minimal and maximal values
    """
    global MAX_NEIGHBOR_DISTANCE, x_step, MAX_FINAL_DISTANCE
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
        print(slicelist)
        print(grid.shape, grid[tuple(slicelist)].shape)
        grid[tuple(slicelist)] += shifts[:d]
        # grid[tuple(slicelist)][:d] = shifts[:d]

    # flatten the array
    grid = np.reshape(grid, (-1, dim))
    # print(grid)

    MAX_NEIGHBOR_DISTANCE = 1.01 * Delta_0
    MAX_FINAL_DISTANCE = 0.7 * Delta_0
    warn.warn("proper estimation of MAX_FINAL_DISTANCE is still necessary")

    return grid, scaling_factor, offset, x_step



def dummy_constraint(p):
    """used when no constraint is applied"""
    return np.ones(p.shape[:-1]).astype(np.bool)


def viability_single_point(coordinate_index, coordinates, states, stop_states, succesful_state, else_state, evolutions,
                           tree, constraint = dummy_constraint):
    """Calculate whether a coordinate with value 'stop_value' can be reached from 'coordinates[coordinate_index]'."""

    start = coordinates[coordinate_index]
    start_state = states[coordinate_index]

    global VERBOSE
    # VERBOSE = (coordinate_index == (10 * 80 - 64,))
    # VERBOSE = la.norm(start - np.array([0.125, 0.649])) < 0.02
    # VERBOSE = VERBOSE or la.norm(start - np.array([0.1, 0.606])) < 0.02
    # VERBOSE = True

    if VERBOSE:
        print()

    for evol_num, evol in enumerate(evolutions):
        point = start

        for _ in range(MAX_EVOLUTION_NUM):
            point = evol(point, STEPSIZE)

            if np.any(np.isnan(point)):
                warn.warn("point {!r} became nan for one option, so I'm assuming it's a fixed point".format(start))
                print(coordinate_index)

                if start_state in stop_states:
                    final_state = succesful_state

                    if VERBOSE:
                        print("%i:" % evol_num, coordinate_index, start, start_state, "-->", final_state)
                    return final_state
                # else
                break

            if np.max(np.abs(point - start)) < x_half_step:
                # not yet close enough to a different point
                # so run the evolution function again
                continue # not yet close enough to another point

            final_distance, tree_index = tree.query(point, 1)
            final_state = states[tree_index]

            if VERBOSE:
                print(coordinates[tree_index], final_state, constraint(point), final_distance, x_step)
                # print('----', tree_index, coordinates[tree_index])
                print(final_state in stop_states, constraint(point),final_distance < MAX_FINAL_DISTANCE)
                print(final_distance, MAX_FINAL_DISTANCE)

            if final_state in stop_states and constraint(point) and final_distance < MAX_FINAL_DISTANCE:

                if VERBOSE:
                    print( "%i:"%evol_num, coordinate_index, start, start_state, "-->", final_state )
                return succesful_state

            # break and run the other evolutions to check whether they can reach a point with 'stop_state'
            if VERBOSE:
                print("%i:"%evol_num, coordinate_index, start, start_state, "## break")
            break

        else:
            # didn't find an option leading to a point with 'stop_state'
            if VERBOSE:
                print("%i:"%evol_num, coordinate_index, start, start_state, "-->", start_state, "didn't leave")
            return start_state

    if VERBOSE:
        print("all:", coordinate_index, start, start_state, "-->", else_state)
    return else_state


def viability_kernel_step(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, tree):
    """do a single step of the viability calculation algorithm by checking which points stay immediately within the good_states"""

    changed = False

    shape = coordinates.shape[:-1]

    for base_index in np.ndindex(shape):
        neighbors = [base_index]

        for index in neighbors: # iterate over the base_index and, if any changes happened, over the neighbors, too
            old_state = states[index]

            if old_state == work_state:
                new_state = viability_single_point(index, coordinates, states, good_states, succesful_state, bad_state, evolutions, tree)

                if new_state != old_state:
                    changed = True
                    states[index] = new_state
                    # get_neighbor_indices(index, shape, neighbor_list = neighbors)
                    get_neighbor_indices_via_cKD(index, tree,  neighbor_list=neighbors)

    return changed


def get_neighbor_indices_via_cKD(index, tree, neighbor_list=[]):
    """extend 'neighbor_list' by 'tree_neighbors', a list that contains the nearest neighbors found trough cKDTree"""

    index = np.asarray(index).astype(int)

    tree_neighbors = tree.query_ball_point(tree.data[index].flatten(), MAX_NEIGHBOR_DISTANCE)
    tree_neighbors = [(x,) for x in tree_neighbors]

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


def viability_kernel(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, tree):
    """calculate the viability kernel by iterating through the viability kernel steps
    until convergence (no further change)"""
    # assert coordinates.shape[:-1] == states.shape[:-1], "'coordinates' and 'states' don't match in shape"

    assert "x_step" in globals() # needs to be set by the user for now ... will be changed later
    assert "MAX_FINAL_DISTANCE" in globals() # needs to be set by the user for now ... will be changed later
    assert "MAX_NEIGHBOR_DISTANCE" in globals() # needs to be set by the user for now ... will be changed later
    global x_half_step
    x_half_step = x_step/2
    if not "STEPSIZE" in globals():
        global STEPSIZE
        # fix stepsize on that for now if nothing else has been given by the
        # user
        STEPSIZE = 2 * x_step

    # actually only on step is needed due to the recursive checks (i.e. first
    # checking all neighbors of a point that changed state)
    viability_kernel_step(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, tree)


def viability_capture_basin(coordinates, states, target_states, reached_state, bad_state, work_state, evolutions, tree):
    """reuse the viability kernel algorithm to calculate the capture basin"""

    return_value =  viability_kernel(coordinates, states, target_states + [reached_state], work_state, reached_state,
                                     work_state, evolutions, tree)
    # all the points that still have the state work_state are not part of the capture basin and are set to be bad_states
    states[ states == work_state ] = bad_state
    return return_value

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
                      returning = "run-function"
                      ):

    #----------- just for 2D Phase-Space-plot to check the scaled right-hand-side
    @nb.jit
    def rhs_scaled_to_one_PS(x0, t):
        x = np.zeros_like(x0)
        x[0] = scaling_factor[0] * x0[0] + offset[0]
        x[1] = scaling_factor[1] * x0[1] + offset[1]
        val = rhs(x, t, *ordered_params)  # calculate the rhs
        val[0] /= scaling_factor[0]
        val[1] /= scaling_factor[1]
        return val
    # ---------------------------------------------------------------------------


    @nb.jit
    def rhs_scaled_to_one(x0, t, *args):
        x = scaling_factor * x0 + offset
        val = rhs(x, t, *args) / scaling_factor # calculate the rhs
        return val


    @nb.jit
    def normalized_rhs(x0, t, *args):
        val = rhs_scaled_to_one(x0, t, *args)  # calculate the rhs
        return val / np.sqrt(np.sum(val ** 2, axis=-1))  # normalize it


    @nb.jit
    def model_run(p, stepsize):
        traj = integ.odeint(normalized_rhs, p, [0, stepsize], args = ordered_params)
        if VERBOSE:
            plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=3)

        return traj[-1]

    if returning == "run-function":
        return model_run
    elif returning == "PS": # the other one was too long, nobody can remember that
        # to check scaled right-hand-side
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


def normalized_grid(boundaries, x_num):
    """generates a normalized grid  in any dimension and gets the scaling factors and linear shift of each axis"""
    global MAX_NEIGHBOR_DISTANCE, x_step, MAX_FINAL_DISTANCE

    dim = int(len(boundaries)/2)

    scaling_factor = np.ones(dim)

    offset = np.zeros(dim)

    for index in range(0, dim):
        scaling_factor[index] = boundaries[index + dim] - boundaries[index]

        if boundaries[index] != 0:
            offset[index] = boundaries[index]

    grid_prep = np.linspace(0, 1, x_num + 1)
    grid_prep = (grid_prep[:-1] + grid_prep[1:]) / 2

    meshgrid_arg = [grid_prep for _ in range(dim)]

    grid = np.asarray(np.meshgrid(*meshgrid_arg))

    grid = np.rollaxis(grid, 0, dim + 1)

    # reshaping coordinates and states in order to use kdtree
    grid = np.reshape(grid, (-1, np.shape(grid)[-1]))

    x_step = 1/(x_num-1)

    MAX_NEIGHBOR_DISTANCE = 1.5 * x_step
    MAX_FINAL_DISTANCE = np.sqrt(dim) * x_step / 2

    return grid, scaling_factor, offset, x_step


def backscaling_grid(grid, scalingfactor, offset):
    return grid * scalingfactor + offset


def topology_classification(coordinates, states, default_evols, management_evols, is_sunny,
                            periodic_boundaries = [], upgradeable_initial_states = False
                            ):
    """calculates different regions of the state space using viability theory algorithms"""

    # upgreading initial states to higher
    if upgradeable_initial_states:
        raise NotImplementedError

    # check, if there are periodic boundaries and if so, use different tree form
    if periodic_boundaries == []:
        tree = spat.cKDTree(coordinates)
    else:
        assert np.shape(coordinates)[-1] == len(periodic_boundaries), "Given boundaries don't match with " \
                                                                            "dimensions of coordinates. " \
                                                                            "Write '-1' if boundary is not periodic!"
        tree = periodkdt.PeriodicCKDTree(periodic_boundaries, coordinates)

    # checking data-type of input evolution functions
    if isinstance(default_evols, list):
        default_evols_list = default_evols
        # print('default_evols is a list')
    else:
        default_evols_list = [default_evols]
        # print('default_evols is not a list')

    if isinstance(management_evols, list):
        management_evols_list = management_evols
        # print('management_evols is a list')
    else:
        management_evols_list = [management_evols]
        # print('management_evols is not a list')

    all_evols = management_evols_list + default_evols_list

    shelter_empty = False
    backwater_empty = False

    # calculate shelter
    print('###### calculating shelter')
    states[(states == 0) & is_sunny(coordinates)] = 1 # initial state for shelter calculation
    # viability_kernel(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, tree)
    viability_kernel(coordinates, states, [1, -1], 0, 1, 1, default_evols_list, tree)

    if not np.any(states == 1):
        print('shelter empty')
        shelter_empty = True

    if not shelter_empty:
        # calculate glade
        print('###### calculating glade')

        states[(states == 0) & is_sunny(coordinates)] = 3

        #states[~is_sunny(coordinates)] = 0 #??????????????????????
        #viability_capture_basin(coordinates, states, target_states, reached_state, bad_state, work_state, evolutions, tree):
        viability_capture_basin(coordinates, states, [1, -1], 2, 0, 3, all_evols, tree)

        # calculate remaining upstream dark and sunny
        print('###### calculating rest of upstream (lake, dark and sunny)')
        states[(states == 0)] = 4
        viability_capture_basin(coordinates, states, [1, 2, -3, -4, -5], 3, 0, 4, all_evols, tree)

        states[~is_sunny(coordinates) & (states == 3)] = 4

        # calculate Lake
        print('###### calculating lake')
        states[is_sunny(coordinates) & (states == 3)] = 5
        viability_kernel(coordinates, states, [1, 2, 5, -5], 3, 5, 5, all_evols, tree)

    # calculate Bachwater
    print('###### calculating backwater')
    states[is_sunny(coordinates) & (states == 0)] = 6
    viability_kernel(coordinates, states, [6, -6], 0, 6, 6, all_evols, tree)

    if not np.any(states == 6):
        print('backwater empty')
        backwater_empty = True

    if not backwater_empty:
        # calculate remaining downstream dark and sunny
        print('###### calculating remaining downstream (dark and sunny)')
        states[(states == 0)] = 8
        viability_capture_basin(coordinates, states, [6, -7, -8], 7, 0, 8, all_evols, tree)
        states[~is_sunny(coordinates) & (states == 7)] = 8

    # set sunny Eddies/Abyss
    print('###### set sunny Eddies/Abyss')
    states[is_sunny(coordinates) & (states == 0)] = 9

    # calculate dark Eddies/Abyss
    print('###### calculating dark Eddies/Abyss')
    # look only at the coordinates with state == 0
    states[(states == 0)] = 12
    viability_capture_basin(coordinates, states,
                            [1, 2, 3, 4, 5, 6, 7, 9, -1, -2, -3, -4 , -5, -6, -7, -9], 10, 0, 12, all_evols, tree)

    # calculate trench
    print('###### set trench')
    states[(states == 0)] = 11
    # Konsistenzcheck? [1, 2, 3, 4, 5, 6, 7, 9] sollten alle nicht erreicht werden:
    # viability_capture_basin(coordinates, states, [1, 2, 3, 4, 5, 6, 7, 9], 0, 11, 11, all_evols, tree)

    # All initially given states are set to positive counterparts
    states[(states < 0)] *= -1

    return states


