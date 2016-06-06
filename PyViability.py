
from __future__ import print_function, division

import PTopologyL as topo

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
#MAX_STEP_NUM = 10  braucht man die noch?
VERBOSE = 0

COLORS = {
        0: "red",
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


def dummy_constraint(p):
    """used when no constraint is applied"""
    return np.ones(p.shape[:-1]).astype(np.bool)


def viability_single_point(coordinate_index, coordinates, states, stop_states, succesful_state, else_state, evolutions,
                           tree, constraint = dummy_constraint):
    """Calculate whether a coordinate with value 'stop_value' can be reached from 'coordinates[coordinate_index]'."""
    # coordinates_reshaped = np.reshape(coordinates, (-1, np.shape(coordinates)[-1]))
    # states_reshaped = np.reshape(states, (-1, 1))

    start = coordinates[coordinate_index]
    start_state = states[coordinate_index]

    # empty_dims = (np.newaxis, ) * len(coordinate_index)

    global VERBOSE
    VERBOSE = (coordinate_index == (13, 2))

    if VERBOSE:
        print()

    for evol_num, evol in enumerate(evolutions):
        point = start

        for _ in range(MAX_EVOLUTION_NUM):
            point = evol(point, STEPSIZE)

            if np.any(np.isnan(point)):
                warn.warn("point {!r} became nan for one option, so I'm assuming it's a fixed point".format(start))

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

            # index = np.argmin(la.norm(point[empty_dims] - coordinates, axis = -1))  # als funktion auslagern
            # final_distance = la.norm(point - coordinates[index])
            final_distance, tree_index = tree.query(point, 1)
            final_state = states[tree_index]

            if VERBOSE:
                print(final_state, constraint(point), final_distance, x_step)

            if final_state in stop_states and constraint(point) and final_distance < x_step:

                if VERBOSE:
                    print( "%i:"%evol_num, coordinate_index, start, start_state, "-->", final_state )
                return succesful_state
            # break and run the other evolutions to check whether they can reach a point with 'stop_state'
            if VERBOSE:
                print("%i:"%evol_num, "break")
            break

        else:
            # didn't find an option leading to a point with 'stop_state'
            if VERBOSE:
                print("%i:"%evol_num, coordinate_index, start, start_state, "-->", start_state, "didn't leave")
            return start_state

    if VERBOSE:
        print("all:", coordinate_index, start, start_state, "-->", else_state)
    return else_state


class NeighborList(list):
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        try:
            return self.pop(0)
        except IndexError:
            raise StopIteration


def viability_kernel_step(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, tree):
    """do a single step of the viability calculation algorithm by checking which points stay immediately within the good_states"""

    coordinates_reshaped = coordinates
    states_reshaped = states

    shape = coordinates.shape[:-1]

    changed = False
##    neighbors_length = 0

    shape_reshaped = coordinates_reshaped.shape[:-1]


    for base_index in np.ndindex(shape_reshaped):
        neighbors = [base_index]

        for index in neighbors: # iterate over the base_index and, if any changes happened, over the neighbors, too
            #
            # print(neighbors)
            old_state = states_reshaped[index]

            if old_state == work_state:
                new_state = viability_single_point(index, coordinates_reshaped, states_reshaped, good_states, succesful_state, bad_state, evolutions, tree)

                if new_state != old_state:
                    changed = True
                    states_reshaped[index] = new_state
                    #get_neighbor_indices(index, shape, neighbor_list = neighbors)
                    get_neighbor_indices_via_cKD(index, tree,  neighbor_list=neighbors)

##        neighbors_length += len(neighbors)
    return changed


def get_neighbor_indices_via_cKD(index, tree, neighbor_list=[]):
    """extend 'neighbor_list' by 'tree_neighbors', a list that contains the nearest neighbors found trough cKDTree"""
    index = np.asarray(index)
    index = index.astype(int)

    tree_neighbors = tree.query_ball_point(tree.data[index].flatten(), 1.5 * x_step)
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
    #assert coordinates.shape[:-1] == states.shape[:-1], "'coordinates' and 'states' don't match in shape"

    assert "x_step" in globals() # needs to be set by the user for now ... will be changed later
    global x_half_step
    x_half_step = x_step/2
    if not STEPSIZE in globals():
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
                 linestyle = "", marker = ".", markersize = 10 ,zorder=0)


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
                      ordered_params
                      ):

    @nb.jit
    def normalized_rhs(x0, t, *args):
        val = rhs(x0, t, *args)  # calculate the rhs
        return val / np.sqrt(np.sum(val ** 2, axis=-1))  # normalize it


    @nb.jit
    def model_run(p, stepsize):
        traj = integ.odeint(normalized_rhs, p, [0, stepsize], args = ordered_params)

        return traj[-1]

    return model_run


def make_run_function2(model_object,
        timestep,
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

def topology_classification(coordinates, states, default_evols, management_evols, is_sunny):
    """calculates different regions of the state space using viability theory algorithms"""

    coordinates = np.reshape(coordinates, (-1, np.shape(coordinates)[-1]))
    states = np.reshape(states, (-1))
    tree = spat.cKDTree(coordinates)

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
    states[(is_sunny(coordinates))] = 1 # initial state for shelter calculation
    viability_kernel(coordinates, states, [1], 0, 1, 1, default_evols_list, tree)


    if not np.any(states == 1):
        print('shelter empty')
        shelter_empty = True

    if not shelter_empty:
        # calculate glade
        print('###### calculating glade')

        states[(states == 0) & is_sunny(coordinates)] = 3

        states[~is_sunny(coordinates)] = 0
        viability_capture_basin(coordinates, states, [1], 2, 0, 3, all_evols, tree)

        # calculate remaining upstream dark and sunny
        print('###### calculating rest of upstream (lake, dark and sunny)')
        states[(states == 0)] = 4
        viability_capture_basin(coordinates, states, [1, 2], 3, 0, 4, all_evols, tree)

        states[~is_sunny(coordinates) & (states == 3)] = 4

        # calculate Lake
        print('###### calculating lake')
        states[is_sunny(coordinates) & (states == 3)] = 5
        viability_kernel(coordinates, states, [1, 2, 5], 3, 5, 5, all_evols, tree)

    # calculate Bachwater
    print('###### calculating backwater')
    states[is_sunny(coordinates) & (states == 0)] = 6
    viability_kernel(coordinates, states, [6], 0, 6, 6, all_evols, tree)

    if not np.any(states == 6):
        print('backwater empty')
        backwater_empty = True

    if not backwater_empty:
        # calculate remaining downstream dark and sunny
        print('###### calculating remaining downstream (dark and sunny)')
        states[(states == 0)] = 8
        viability_capture_basin(coordinates, states, [6], 7, 0, 8, all_evols, tree)
        states[~is_sunny(coordinates) & (states == 7)] = 8

    # set sunny Eddies/Abyss
    states[is_sunny(coordinates) & (states == 0)] = 9

    # calculate dark Eddies/Abyss
    print('###### calculating dark Eddies/Abyss')
    # look only at the coordinates with state == 0
    states[(states == 0)] = 12
    viability_capture_basin(coordinates, states, [9], 10, 0, 12, all_evols, tree)
    # Konsistenzcheck? [1, 2, 3, 4, 5, 6, 7, 9] sollten alle nicht erreicht werden

    # calculate trench
    print('###### set trench')
    states[(states == 0)] = 11
    #viability_capture_basin(coordinates, states, [1, 2, 3, 4, 5, 6, 7, 9], 0, 11, 11, all_evols, tree)

    return states
