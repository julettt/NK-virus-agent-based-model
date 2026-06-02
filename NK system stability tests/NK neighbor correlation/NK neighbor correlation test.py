import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.stats import poisson

grid_size = 50
NK_ratio = 1.0
max_NK = 3
a = 3/4
b = 3
t_max = 3
steps_max = 100000

grid = np.zeros((grid_size, grid_size))

def init_NK_grid(NK_ratio):
    k_values = np.arange(max_NK + 1)
    probs = poisson.pmf(k_values, mu = NK_ratio)
    probs /= probs.sum()
    NK_grid = np.random.choice(k_values, size = (grid_size, grid_size), p = probs)
    return NK_grid


def get_neigh_pairs(NK_grid):
        
    r, c = np.ogrid[:grid_size, :grid_size]

    dir_even = np.array([[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]])
    dir_odd = np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]])

    num_dir = len(dir_even)

    centers_vec = []
    neighs_vec = []

    for i in range(num_dir):
        dr = np.where(r % 2 == 0, dir_even[i, 0], dir_odd[i, 0]) #offset for coordinate r
        dc = np.where(r % 2 == 0, dir_even[i, 1], dir_odd[i, 1]) #offset for coordinate c

        nr = (r + dr) % grid_size
        nc = (c + dc) % grid_size

        centers_vec.extend(NK_grid.flatten())
        neighs_vec.extend(NK_grid[nr, nc].flatten())

    return pd.DataFrame({'center_cell': centers_vec, 'neighbors': neighs_vec})


def get_directions(r):
    
    if r % 2 == 0:
        return np.array([[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]])
    else:
        return np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]])
    


def NK_move_prop(r, c, NK_grid, a = a, b = b):
    
    if NK_grid[r, c] == 0:
        return 0

    directions = get_directions(r)

    has_valid_neighbors = False
    for dr, dc in directions:
        nr = (r + dr) % grid_size
        nc = (c + dc) % grid_size
        
        if NK_grid[nr, nc] < max_NK:
            has_valid_neighbors = True
            break

    if not has_valid_neighbors:
        return 0
    
    return a * NK_grid[r, c] / (1 + b * np.max(NK_grid[r, c] - 1, 0))


def run_test(b):

    NK_grid = init_NK_grid(NK_ratio)
    all_states = list(range(max_NK + 1))

    NK_move_propens = np.zeros((grid_size, grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            NK_move_propens[r, c] = NK_move_prop(r, c, NK_grid, a, b)


    time = 0
    steps = 0

    corr_history = {}

    while steps <= steps_max:

        #finding correlation every 250 steps
        if steps % 250 == 0:

            df_pairs  = get_neigh_pairs(NK_grid)

            cond_prob_matrix = pd.crosstab(df_pairs['center_cell'], df_pairs['neighbors'],
                                           normalize = 'index').reindex(index = all_states, columns = all_states, fill_value = 0.0)

            corr_history[steps] = {'time': time, 'matrix': cond_prob_matrix}


        all_propens = NK_move_propens.flatten()
        total_propensity = all_propens.sum()

        if total_propensity == 0: break


        #randomly choosing time and an event for that time
        dt = - np.log(np.random.random()) / total_propensity
        event_id = np.random.choice(len(all_propens), p = all_propens / total_propensity)
        
        time += dt
        
        r, c = divmod(event_id, grid_size)


        directions = get_directions(r)
            
        neighbors = [((r + dr) % grid_size, (c + dc) % grid_size) for dr, dc in directions]
        valid_neighbors = [(nr, nc) for nr, nc in neighbors if NK_grid[nr, nc] < max_NK]

        idx = np.random.randint(0, len(valid_neighbors))
        new_r, new_c = valid_neighbors[idx]

        NK_grid[r, c] -= 1
        NK_grid[new_r, new_c] += 1

        #local propensities update
        cells_to_update = {(r, c), (new_r, new_c)}

        for dr, dc in directions:
            cells_to_update.add(((r + dr) % grid_size, (c + dc) % grid_size))
        
        for dr, dc in get_directions(new_r):
            cells_to_update.add(((new_r + dr) % grid_size, (new_c + dc) % grid_size))
        
        for ur, uc in cells_to_update:
            NK_move_propens[ur, uc] = NK_move_prop(ur, uc, NK_grid, a, b)

        steps += 1

    return corr_history


#b_values = [0, 1, 3, 5, 20, 100]

b_values = [0, 1, 3, 5, 20, 100]

n_real = 5

filename = f'b values3.txt'

with open(filename, 'w', encoding = 'utf-8') as f:

    for b in b_values:
        
        print(f'start for b = {b}')

        claster_matrix = None

        for r in range(n_real):
            print(f'realizacja {r + 1}/{n_real}')
            history = run_test(b)

            last_step = max(history.keys())
            final_matrix = history[last_step]['matrix']

            if claster_matrix is None:
                claster_matrix = final_matrix.copy()

            else:
                claster_matrix += final_matrix

        
        averaged_claster_matrix = claster_matrix / n_real
        
        print('------------------------------------------', file = f)
        print(f'1st neighborhood correlation; steps: {steps_max}, b = {b}, averaged over {n_real} realizations', file = f)
        print('------', file = f)

        print(averaged_claster_matrix, file = f)

        print(f'finished for b = {b}')
