import numpy as np
import scipy.stats as stats
from numba import njit

LINE = True #! with periodic boundary conditions only vertically
SQUARE = False

#----- grids initialization functions -----
def init_epith_grid(MOI, grid_r, grid_c):
    "initialize epithelial grids with given MOI"
    if LINE:
        epith_grid = np.zeros((grid_r, grid_c)).astype(np.int64)
        epith_grid[:,:1] = 1

    elif SQUARE:
        epith_grid = np.zeros((grid_r, grid_c)).astype(np.int64)
        epith_grid[40:60,40:60] = 1

    else:
        p_inf = 1 - np.exp(-MOI)
        epith_grid = (np.random.random((grid_r, grid_c)) < p_inf).astype(np.int64)
    
    return epith_grid

def init_NK_grid(NK_ratio, max_NK, grid_r, grid_c):
    "initialize NK grid with given NK:epithelial ratio"

    k_values = np.arange(max_NK + 1)
    probs = stats.poisson.pmf(k_values, mu = NK_ratio)
    probs /= probs.sum()
    NK_grid = np.random.choice(k_values, size = (grid_r, grid_c), p = probs)

    return NK_grid

@njit(cache=True)
def get_hex_neighbors(r, c, grid_r, grid_c):
    "returns an (6, 2) array of hexagonal grid neighbors with periodic boundary conditions"

    if r % 2 == 0:
        directions = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0))
    else:
        directions = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1))

    result = np.empty((6, 2), dtype = np.int64)
    
    for i in range(6):

        result[i, 0] = (r + directions[i][0]) % grid_r

        if LINE:

            nc = c + directions[i][1]
            if nc < 0 or nc >= grid_c:
                result[i, 1] = -1 #mark for non-periodic boundary, no neighbor here
            else:
                result[i, 1] = nc

        else:
            result[i, 1] = (c + directions[i][1]) % grid_c

    return result


#----- propensity functions -----
@njit(cache=True)
def prog_prop(t_prog, with_infection_progression):
    "propensity of an infected cell to progress from state S to state S+1"

    if with_infection_progression == False:
        return 0.0

    return 1 / t_prog

@njit(cache=True)
def infect_prop(r, c, epith_grid, epith_is_alive, grid_r, grid_c, beta_spread, t_spread, M_I, with_infection_spread):
    """
    propensity of a healthy cell to become infected
    - calculated for a cell that can be infected, not for infectious neighbors
    - depends on the states of neighboring cells
    """

    if with_infection_spread == False:
        return 0.0
    
    total_inf_prop = 0.0

    neighbors = get_hex_neighbors(r, c, grid_r, grid_c)

    for i in range(6):
        nr, nc = neighbors[i, 0], neighbors[i, 1]
        
        if nc == -1: #no neighbor here due to being outside grid vertically
            continue

        if not epith_is_alive[nr, nc]: #dead neighboring cell cannot infect other cells
            continue

        S_neigh = epith_grid[nr, nc]

        if S_neigh > M_I:
            total_inf_prop += (beta_spread * S_neigh) / t_spread
    
    return total_inf_prop

@njit(cache=True)
def death_I_prop(r, c, epith_grid, c_0, c_I, M_D, with_death_I):
    "propensity to die independent of its local neighborhood"

    if with_death_I == False:
        #no independent death
        return 0.0

    S_cell = epith_grid[r, c]

    if S_cell > M_D:
        return c_0 + c_I * S_cell

    return c_0


@njit(cache=True)
def death_NK_prop(r, c, epith_grid, NK_grid, c_NK, c_I_NK, M_D, with_death_NK):
    "propensity to die provoked by the presence of NK cells"
    
    if with_death_NK == False:
        return 0.0

    NK_cells = NK_grid[r, c]
    S_cell = epith_grid[r, c]

    if S_cell > M_D:
        return NK_cells * (c_NK + c_I_NK * S_cell)

    return NK_cells * c_NK


@njit(cache=True)
def NK_move_prop(r, c, NK_grid, max_NK, grid_r, grid_c, a, b):
    "- propensity to initiate a migration step from a given node"
    "- returns 0 if no move are possible or there are none NK cells"

    NK_count = NK_grid[r, c]

    if NK_count == 0:
        return 0.0
    
    neighs = get_hex_neighbors(r, c, grid_r, grid_c)

    move_possible = False
    for i in range(6):

        if neighs[i, 1] == -1:
            continue

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
def run_simulation(time_max, virus_delay, NK_delay, max_steps, ani_step_save, grid_r, grid_c,
                   max_NK, a, b, max_inf_state, t_prog, beta_spread, t_spread, M_I, M_D,
                   c_0, c_I, c_NK, c_I_NK,
                   epith_grid_delayed, empty_grid, NK_grid_delayed,
                   with_death_I, with_death_NK, with_infection_spread, with_infection_progression):
    
    #preparing grids
    NK_grid = empty_grid.copy()
    epith_grid = empty_grid.copy()
    epith_is_alive = np.full((grid_r, grid_c), True, dtype = np.bool_)
    NK_move_m = np.zeros((grid_r, grid_c), dtype = np.float64)
    inf_evol_m = np.zeros((grid_r, grid_c), dtype = np.float64)
    death_m = np.zeros((grid_r, grid_c), dtype = np.float64)
    inf_spread_m = np.zeros((grid_r, grid_c), dtype = np.float64)
    dead_inf_hist = np.zeros(max_inf_state + 1, dtype = np.float64)

    total_NK_move = 0.0
    total_inf_evol = 0.0
    total_inf_spread = 0.0
    total_death = 0.0


    NK_introduced = False
    virus_introduced = False

    time = 0.0
    steps = 0

    max_frames = (max_steps // ani_step_save) + 2
    frames_NK = np.zeros((max_frames, grid_r, grid_c), dtype = NK_grid.dtype)
    frames_epith = np.zeros((max_frames, grid_r, grid_c), dtype = epith_grid.dtype)
    stats_array = np.zeros((max_frames, 4 + 2 * (max_inf_state + 1)), dtype = np.float64)
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
            neighbors = get_hex_neighbors(r, c, grid_r, grid_c)
            
            for i in range(6):
                nr, nc = neighbors[i, 0], neighbors[i, 1]

                if nc == -1:
                    continue

                if epith_is_alive[nr, nc] and epith_grid[nr, nc] == 0:
                    #our neighbor is both alive and healty, so it can get infected
                    _set_spread(nr, nc, infect_prop(nr, nc, epith_grid, epith_is_alive, grid_r, grid_c, beta_spread, t_spread, M_I, with_infection_spread))

            return

        #----- alive cell -----
        S = epith_grid[r, c]
        _set_evol(r, c, prog_prop(t_prog, with_infection_progression) if 0 < S < max_inf_state else 0.0)
        _set_death(r, c, death_I_prop(r, c, epith_grid, c_0, c_I, M_D, with_death_I) + death_NK_prop(r, c, epith_grid, NK_grid, c_NK, c_I_NK, M_D, with_death_NK))

        neighbors = get_hex_neighbors(r, c, grid_r, grid_c)
        for i in range(6):
            nr, nc = neighbors[i, 0], neighbors[i, 1]

            if nc == -1:
                continue

            if epith_is_alive[nr, nc] and epith_grid[nr, nc] == 0:
                    #our neighbor is both alive and healty, so it can get infected
                    _set_spread(nr, nc, infect_prop(nr, nc, epith_grid, epith_is_alive, grid_r, grid_c, beta_spread, t_spread, M_I, with_infection_spread))

    
    def _update_NK(r, c):
        _set_NK(r, c, NK_move_prop(r, c, NK_grid, max_NK, grid_r, grid_c, a, b))

    def _save_frame():
        nonlocal frame_idx

        frames_NK[frame_idx] = NK_grid.copy()
        frames_epith[frame_idx] = epith_grid.copy()

        #----- statistics -----
        healthy = 0
        infected = 0
        dead = 0
        for r in range(grid_r):
            for c in range(grid_c):
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

        for r in range(grid_r):
            for c in range(grid_c):
                S = epith_grid[r, c]
                if epith_is_alive[r, c]:
                    inf_histogram[S] += 1
            
        stats_array[frame_idx, 4: 4 + max_inf_state + 1] = inf_histogram
        stats_array[frame_idx, 4 + max_inf_state + 1 :] = dead_inf_hist
           
        frame_idx += 1

    #----- evolution process ------
    while time <= time_max:

        #introducing virus
        if not virus_introduced and time >= virus_delay:
            epith_grid = epith_grid_delayed
            for r in range(grid_r):
                for c in range(grid_c):
                    _update_epith(r, c) #propens update
                    
            virus_introduced = True


        #introducing NK cells
        if not NK_introduced and time >= NK_delay:
            NK_grid = NK_grid_delayed
            for r in range(grid_r):
                for c in range(grid_c):
                    _update_NK(r, c)
                    _update_epith(r, c) #deth_dep terms have changed after introducing NK cells
                    
            NK_introduced = True

        
        #saving frames for animation
        if steps % ani_step_save == 0:
            _save_frame()

        if 23.9 < time < 24:
            time = 24.0
            _save_frame()

        if 47.9 < time < 48:
            time = 48.0
            _save_frame()

        PROP_total = total_NK_move + total_inf_evol + total_inf_spread + total_death

        if PROP_total <= 0.0:
            if not NK_introduced and NK_delay <= time_max:
                time = NK_delay
                continue
            if not virus_introduced and virus_delay <= time_max:
                time = virus_delay
                continue
            print(f'no possible events')
            #time = time_max
            _save_frame()
            break
        
        infection_can_happen = with_infection_spread or with_infection_progression

        if infection_can_happen and total_inf_evol + total_inf_spread <= 1e-8 and virus_introduced and NK_introduced:
            #infection once was and now it is gone, only NK cells remains
            print(f'virus is gone')
            _save_frame()
            break

        elif infection_can_happen and total_inf_evol + total_inf_spread <= 1e-8 and virus_introduced and not NK_introduced:
            #infection once was and it is gone before introduction NK cells
            print(f'virus is gone with no presence of NK cells in the system')
            _save_frame()
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
            neighbors = get_hex_neighbors(r, c, grid_r, grid_c)

            valid_r = np.empty(6, dtype = np.int64)
            valid_c = np.empty(6, dtype = np.int64)
            n_valid = 0 #number of valid neighbors

            for i in range(6):
                nr, nc = neighbors[i, 0], neighbors[i, 1]

                if nc == -1:
                    continue

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
                nb_source = get_hex_neighbors(r, c, grid_r, grid_c)
                for i in range(6):
                    if nb_source[i, 1] == -1:
                        continue
                    _update_NK(nb_source[i, 0], nb_source[i, 1])

                _update_NK(new_r, new_c)
                nb_dest = get_hex_neighbors(new_r, new_c, grid_r, grid_c)
                for i in range(6):
                    if nb_dest[i, 1] == -1:
                        continue
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
            
            dead_inf_hist[epith_grid[r,c]] += 1

            epith_is_alive[r, c] = False
            epith_grid[r, c] = -1
            _update_epith(r, c)

        steps += 1

        if steps >= max_steps:
            print(f'maximum number of steps exceeded!')

            val_evol = total_inf_evol
            val_spread = total_inf_spread
            val_prop = PROP_total

            print('total_inf_evol + total_inf_spread:', val_evol, '+', val_spread)
            print('total propensity:', val_prop)

            _save_frame()
            break

    _save_frame()
    return frames_NK[:frame_idx], frames_epith[:frame_idx], stats_array[:frame_idx], time, steps



#----- simulation ------

def run_with_params(p: dict):
    "run simulation with p as parameters dictionary"
    epith_grid_delayed = init_epith_grid(p["MOI"], p["grid_r"], p["grid_c"])
    empty_grid = np.zeros((p["grid_r"], p["grid_c"]), dtype=np.int64)
    NK_grid_delayed = init_NK_grid(p["NK_ratio"], p["max_NK"], p["grid_r"], p["grid_c"])

    frames_NK, frames_epith, stats_array, total_time, total_steps = run_simulation(p["time_max"], p["virus_delay"], p["NK_delay"],
                                                                    p["max_steps"], p["ani_step_save"], p["grid_r"], p["grid_c"],
                                                                    p["max_NK"], p["a"], p["b"], p["max_inf_state"],
                                                                    p["t_prog"], p["beta_spread"], p["t_spread"], p["M_I"], p["M_D"],
                                                                    p["c_0"], p["c_I"], p["c_NK"], p["c_I_NK"],
                                                                    epith_grid_delayed, empty_grid, NK_grid_delayed,
                                                                    p["with_death_I"], p["with_death_NK"], p["with_infection_spread"], p["with_infection_progression"])

    return {"stats_array" : stats_array,
            "frames_NK": frames_NK,
            "frames_epith": frames_epith,
            "total_time": total_time,
            "total_steps": total_steps,
            "params": p}
