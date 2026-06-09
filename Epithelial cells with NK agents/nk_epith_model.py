import numpy as np
import scipy.stats as stats
from numba import njit


"""time_max, max_steps = 10.0, 100000
ani_step_save = 250

grid_size = 250
MOI = 0.1

a, b = 1.0, 3.0
max_NK, NK_ratio = 3, 1.0

M_I, max_inf_state = 4, 10
time_delay = 0

t_prog = 1.0
beta_spread, t_spread = 1.0, 50.0
gamma_ind, t_ind = 1.0, 300.0
gamma_dep, t_dep = 3.0, 100.0"""


#----- grids initialization functions -----
def init_epith_grid(MOI, grid_size):
    "initialize epithelial grids with given MOI"

    p_inf = 1 - np.exp(-MOI)
    epith_grid = (np.random.random((grid_size, grid_size)) < p_inf).astype(np.int64)
    
    return epith_grid

def init_NK_grid(NK_ratio, max_NK, grid_size):
    "initialize NK grid with given NK:epithelial ratio"

    k_values = np.arange(max_NK + 1)
    probs = stats.poisson.pmf(k_values, mu = NK_ratio)
    probs /= probs.sum()
    NK_grid = np.random.choice(k_values, size = (grid_size, grid_size), p = probs)

    return NK_grid

@njit(cache=True)
def get_hex_neighbors(r, c, grid_size):
    "returns an (6, 2) array of hexagonal grid neighbors with periodic boundary conditions"

    if r % 2 == 0:
        directions = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0))
    else:
        directions = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1))

    result = np.empty((6, 2), dtype = np.int64)
    
    for i in range(6):
        result[i, 0] = (r + directions[i][0]) % grid_size
        result[i, 1] = (c + directions[i][1]) % grid_size

    return result


#----- propensity functions -----
@njit(cache=True)
def prog_prop(t_prog):
    "propensity of an infected cell to progress from state S to state S+1"
    return 1 / t_prog

@njit(cache=True)
def infect_prop(r, c, epith_grid, epith_is_alive, grid_size, beta_spread, t_spread, M_I):
    """
    propensity of a healthy cell to become infected
    - calculated for a cell that can be infected, not for infectious neighbors
    - depends on the states of neighboring cells
    """
    total_inf_prop = 0.0

    neighbors = get_hex_neighbors(r, c, grid_size)

    for i in range(6):
        nr, nc = neighbors[i, 0], neighbors[i, 1]
        
        if not epith_is_alive[nr, nc]: #dead neighboring cell cannot infect other cells
            continue

        S_neigh = epith_grid[nr, nc]

        if S_neigh > M_I:
            total_inf_prop += (beta_spread * S_neigh) / t_spread
    
    return total_inf_prop

@njit(cache=True)
def death_ind_prop(r, c, epith_grid, gamma_ind, t_ind, M_I):
    "propensity to die independent of its local neighborhood"

    S_cell = epith_grid[r, c]

    if S_cell > M_I:
        return (gamma_ind + S_cell) / t_ind

    return gamma_ind / t_ind


@njit(cache=True)
def death_dep_prop(r, c, epith_grid, NK_grid, gamma_dep, t_dep, M_I):
    "propensity to die provoked by the presence of NK cells"
    
    NK_cells = NK_grid[r, c]
    S_cell = epith_grid[r, c]

    if S_cell > M_I:
        return (gamma_dep + S_cell) * NK_cells / t_dep

    return gamma_dep * NK_cells / t_dep


@njit(cache=True)
def NK_move_prop(r, c, NK_grid, max_NK, grid_size, a, b):
    "- propensity to initiate a migration step from a given node"
    "- returns 0 if no move are possible or there are none NK cells"

    NK_count = NK_grid[r, c]

    if NK_count == 0:
        return 0.0
    
    neighs = get_hex_neighbors(r, c, grid_size)

    move_possible = False
    for i in range(6):

        if NK_grid[neighs[i, 0], neighs[i, 1]] < max_NK:
            move_possible = True
            break

    if not move_possible:
        return 0.0
    
    denominator = 1.0 + b * float(max(NK_count - 1, 0))

    return (a * float(NK_count)) / denominator
    
#----- helper functions for simulation -----

@njit(cache=True)
def select_event(matrix, rand_val):
    "returns (r, c) of a randomly chosen event"
    cum = 0.0
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            cum += matrix[r, c]
            
            if cum >= rand_val - 1e-9:
                return r, c
            
    #for edge case: returning last non-zero cell
    for r in range(matrix.shape[0] - 1, -1, -1):
        for c in range(matrix.shape[1] - 1, -1, -1):
            if matrix[r, c] > 0.0:
                return r, c
            
    return 0, 0



@njit(cache=True)
def run_simulation(time_max, time_delay, max_steps, ani_step_save, grid_size,
                   max_NK, a, b, max_inf_state, t_prog, beta_spread, t_spread, M_I,
                   gamma_ind, t_ind, gamma_dep, t_dep,
                   epith_grid, NK_grid):
    
    #preparing grids
    epith_is_alive = np.full((grid_size, grid_size), True, dtype = np.bool_)
    NK_move_m = np.zeros((grid_size, grid_size), dtype = np.float64)
    inf_evol_m = np.zeros((grid_size, grid_size), dtype = np.float64)
    death_m = np.zeros((grid_size, grid_size), dtype = np.float64)
    inf_spread_m = np.zeros((grid_size, grid_size), dtype = np.float64)

    total_NK_move = 0.0
    total_inf_evol = 0.0
    total_inf_spread = 0.0
    total_death = 0.0


    NK_introduced = False
    time = 0.0
    steps = 0

    max_frames = (max_steps // ani_step_save) + 2
    frames_NK = np.zeros((max_frames, grid_size, grid_size), dtype = NK_grid.dtype)
    frames_epith = np.zeros((max_frames, grid_size, grid_size), dtype = epith_grid.dtype)
    stats_array = np.zeros((max_frames, 4 + max_inf_state + 1), dtype = np.float64)
    frame_idx = 0

    #----- helper functions to update one entry -----
    #we change total values of certain propens by the value that has chaned, then we replace proper node
    #in the proper matrix with this value

    def _set_evol(r, c, val):
        nonlocal total_inf_evol
        total_inf_evol += val - inf_evol_m[r, c]
        inf_evol_m[r, c] = val

    def _set_spread(r, c, val):
        nonlocal total_inf_spread
        total_inf_spread += val - inf_spread_m[r, c]
        inf_spread_m[r, c] = val

    def _set_death(r, c, val):
        nonlocal total_death
        total_death += val - death_m[r, c]
        death_m[r, c] = val

    def _set_NK(r, c, val):
        nonlocal total_NK_move
        total_NK_move += val - NK_move_m[r, c]
        NK_move_m[r, c] = val

    
    #function to update all epith propens for given (r, c) and its neighbors
    
    def _update_epith(r, c):
        
        #----- dead cell -----
        if not epith_is_alive[r, c]:
            _set_evol(r, c, 0.0)
            _set_death(r, c, 0.0)
            _set_spread(r, c, 0.0)
            
            #we have to update also neighbors' propens to get infected
            neighbors = get_hex_neighbors(r, c, grid_size)
            
            for i in range(6):
                nr, nc = neighbors[i, 0], neighbors[i, 1]
                if epith_is_alive[nr, nc] and epith_grid[nr, nc] == 0:
                    #our neighbor is both alive and healty, so it can get infected
                    _set_spread(nr, nc, infect_prop(nr, nc, epith_grid, epith_is_alive, grid_size, beta_spread, t_spread, M_I))

            return

        #----- alive cell -----
        S = epith_grid[r, c]
        _set_evol(r, c, prog_prop(t_prog) if 0 < S < max_inf_state else 0.0)
        _set_death(r, c, death_ind_prop(r, c, epith_grid, gamma_ind, t_ind, M_I) + death_dep_prop(r, c, epith_grid, NK_grid, gamma_dep, t_dep, M_I))

        neighbors = get_hex_neighbors(r, c, grid_size)
        for i in range(6):
            nr, nc = neighbors[i, 0], neighbors[i, 1]
            if epith_is_alive[nr, nc] and epith_grid[nr, nc] == 0:
                    #our neighbor is both alive and healty, so it can get infected
                    _set_spread(nr, nc, infect_prop(nr, nc, epith_grid, epith_is_alive, grid_size, beta_spread, t_spread, M_I))

    
    def _update_NK(r, c):
        _set_NK(r, c, NK_move_prop(r, c, NK_grid, max_NK, grid_size, a, b))


    #initial propensities -> NK not yet introduced
    for r in range(grid_size):
        for c in range(grid_size):
            _update_epith(r, c)


    #----- evolution process ------
    while time <= time_max:

        #introducing NK cells
        if not NK_introduced and time >= time_delay:
            for r in range(grid_size):
                for c in range(grid_size):
                    _update_NK(r, c)
                    _update_epith(r, c) #deth_dep terms have changed after introducing NK cells
                    
            NK_introduced = True

        
        #saving frames for animation
        if steps % ani_step_save == 0:
            frames_NK[frame_idx] = NK_grid.copy()
            frames_epith[frame_idx] = epith_grid.copy()

            #----- statistics -----
            healthy = 0
            infected = 0
            dead = 0
            for r in range(grid_size):
                for c in range(grid_size):
                    if not epith_is_alive[r, c]:
                        dead += 1
                    elif epith_grid[r, c] == 0:
                        healthy += 1
                    else:
                        infected += 1

            stats_array[frame_idx, 0] = float(healthy)
            stats_array[frame_idx, 1] = float(infected)
            stats_array[frame_idx, 2] = float(dead)
            stats_array[frame_idx, 3] = float(time)

            inf_histogram = np.zeros(max_inf_state + 1, dtype = np.float64)

            for r in range(grid_size):
                for c in range(grid_size):
                    S = epith_grid[r, c]
                    if epith_is_alive[r, c]:
                        inf_histogram[S] += 1
            
            stats_array[frame_idx, 4:] = inf_histogram

            frame_idx += 1

        PROP_total = total_NK_move + total_inf_evol + total_inf_spread + total_death

        if PROP_total <= 0.0:
            break

        if total_inf_evol + total_inf_spread == 0.0 and NK_introduced:
            #infection is gone, only NK cells remains
            break
        

        #----- Gillespie algorithm -----
        
        #choosing random time (r1) and event (r2)
        r1 = np.random.random()
        r2 = np.random.random()

        time += -np.log(r1) / PROP_total

        rand_val = r2 * PROP_total
        
        #executing chosen event

        #----- event: NK cell move -----
        if rand_val < total_NK_move:
            r, c = select_event(NK_move_m, rand_val)
            neighbors = get_hex_neighbors(r, c, grid_size)

            valid_r = np.empty(6, dtype = np.int64)
            valid_c = np.empty(6, dtype = np.int64)
            n_valid = 0 #number of valid neighbors

            for i in range(6):
                nr, nc = neighbors[i, 0], neighbors[i, 1]

                if NK_grid[nr, nc] < max_NK:
                    valid_r[n_valid] = nr
                    valid_c[n_valid] = nc
                    n_valid += 1

            if n_valid > 0:
                #move possible -> we can choose where to move
                idx = np.random.randint(0, n_valid)
                new_r, new_c = valid_r[idx], valid_c[idx]

                NK_grid[r, c] -= 1
                NK_grid[new_r, new_c] += 1

                #propensity update for NK cells: source, dest and all their neighbors
                _update_NK(r, c)
                nb_source = get_hex_neighbors(r, c, grid_size)
                for i in range(6):
                    _update_NK(nb_source[i, 0], nb_source[i, 1])

                _update_NK(new_r, new_c)
                nb_dest = get_hex_neighbors(new_r, new_c, grid_size)
                for i in range(6):
                    _update_NK(nb_dest[i, 0], nb_dest[i, 1])

                #propensity update for epithelial cells: source and dest:
                #deth_dep_prop -> source and dest needs to be updated, NK presence on certain node does not affect neighboring cells, 
                #only its own node
                _update_epith(r, c)
                _update_epith(new_r, new_c)
                

        #----- event: infection evolution -----
        elif rand_val < total_NK_move + total_inf_evol:
            adj_rand_val = rand_val - total_NK_move
            r, c = select_event(inf_evol_m, adj_rand_val)

            #updating infection state and propensities
            epith_grid[r, c] += 1
            _update_epith(r, c)

        #----- event: infection spread to a healthy neighbor-----
        elif rand_val < total_NK_move + total_inf_evol + total_inf_spread:
            adj_rand_val = rand_val - total_NK_move - total_inf_evol
            
            r, c = select_event(inf_spread_m, adj_rand_val) #selecting cell to become infected

            epith_grid[r, c] = 1
            _update_epith(r, c)


        #----- event: cell death (last type of event) -----
        else:
            adj_rand_val = rand_val - total_NK_move - total_inf_evol - total_inf_spread
            r, c = select_event(death_m, adj_rand_val) #selecting cell to kill
            
            epith_is_alive[r, c] = False
            epith_grid[r, c] = -1
            _update_epith(r, c)

        steps += 1

    return frames_NK[:frame_idx], frames_epith[:frame_idx], stats_array[:frame_idx], time, steps


#----- simulation ------

def run_with_params(p: dict):
    "run simulation with p as parameters dictionary"
    epith_grid = init_epith_grid(p["MOI"], p["grid_size"])
    NK_grid = init_NK_grid(p["NK_ratio"], p["max_NK"], p["grid_size"])

    frames_NK, frames_epith, stats_array, total_time, total_steps = run_simulation(p["time_max"], p["time_delay"], 
                                                                    p["max_steps"], p["ani_step_save"], p["grid_size"],
                                                                    p["max_NK"], p["a"], p["b"], p["max_inf_state"],
                                                                    p["t_prog"], p["beta_spread"], p["t_spread"], p["M_I"],
                                                                    p["gamma_ind"], p["t_ind"], p["gamma_dep"], p["t_dep"],
                                                                    epith_grid, NK_grid)

    return {"stats_array" : stats_array,
            "total_time": total_time,
            "total_steps": total_steps,
            "params": p}


"""if __name__ == "__main__":
    default_params = dict(time_max = time_max, time_delay = time_delay, 
                          max_steps = max_steps, ani_step_save = ani_step_save, grid_size = grid_size, 
                          max_NK = max_NK, a = a, b = b, max_inf_state = max_inf_state, 
                          t_prog = t_prog, beta_spread = beta_spread, t_spread = t_spread, M_I = M_I,
                          gamma_ind = gamma_ind, t_ind = t_ind, gamma_dep = gamma_dep, t_dep = t_dep,
                          MOI = MOI, NK_ratio = NK_ratio)
    
    res = run_with_params(default_params)
    print(f"Gotowe: t={res['total_time']:.3f} h, kroki={res['total_steps']}")"""

"""#grids initialization
epith_grid = init_epith_grid(MOI)
NK_grid = init_NK_grid(NK_ratio)


frames_NK, frames_epith, stats_array, total_time, total_steps = run_simulation(time_max, time_delay, max_steps, ani_step_save, grid_size,
                   max_NK, a, b, max_inf_state, t_prog, beta_spread, t_spread, M_I,
                   gamma_ind, t_ind, gamma_dep, t_dep,
                   epith_grid, NK_grid)

np.savez_compressed("simulation_history.npz",
                    frames_NK = frames_NK,
                    frames_epith = frames_epith,
                    stats_array = stats_array,
                    grid_size = np.array(grid_size),
                    time_max = np.array(time_max),
                    total_time = np.array(total_time),
                    total_steps = np.array(total_steps),
                    MOI = np.array(MOI),
                    NK_ratio = np.array(NK_ratio))

print(f'Finished: time = {total_time:.3f} h, steps = {total_steps}.')
print(f'Frames saved: {len(frames_epith)}.')
if len(stats_array) > 0:
    last = stats_array[-1]
    print(f"Final state: healthy = {int(last[0])}, infected = {int(last[1])}, dead = {int(last[2])}.")"""