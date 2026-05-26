import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as tri


grid_size = 4
moi = 0.5
t_max = 1
a = 3/4


grid = np.zeros((grid_size, grid_size))

epith_grid = np.random.poisson(moi, size = grid.shape)
epith_grid = np.where(epith_grid > 1, 1, epith_grid).astype(float)

epith_is_alive = np.full_like(epith_grid, True, dtype = 'bool')

def death_prop(epith, x = 300):
    return (1 + epith) / x

def infection_prop(epith, x = 100):
    return (epith) / x

def inf_evolution_prop(a, epith, x = 1):
    return (a * epith) / x

def update_local_propens(r, c):

    #update for killed cell:
    if not epith_is_alive[r, c]:
        inf_evol_m[r, c] = 0.0
        death_m[r, c] = 0.0
        return 
    
    #infection evolution:
    if epith_grid[r, c] < 10:
        inf_evol_m[r, c] = inf_evolution_prop(a, epith_grid[r, c])
    
    else:
        inf_evol_m[r, c] = 0.0

    #death propensity:
    death_m[r, c] = death_prop(epith_grid[r, c])
        


inf_evol_m = np.zeros((grid_size, grid_size))
death_m = np.zeros((grid_size, grid_size))

for r in range (grid_size):
    for c in range(grid_size):
        update_local_propens(r, c)

#SIMULATION
frames = []
time = 0
steps = 0
max_steps = 100


total_cell_propens = inf_evol_m + death_m
total_propensity = total_cell_propens.sum()

while steps < max_steps:

    #infection_evolution, death_propens
    if total_propensity <= 1e-12: 
        print('no possible events')
        break

    dt = - np.log(np.random.random()) / total_propensity
    time += dt

    flat_propens = total_cell_propens.flatten()

    #randomly choosing cell for an event
    cell_id = np.random.choice(len(flat_propens), p = flat_propens / total_propensity)
    r, c = divmod(cell_id, grid_size)

    #choosing event for that cell
    propens_in_cell = np.array((inf_evol_m[r, c], death_m[r, c]))
    event_type = np.random.choice(['infection evolution', 'cell death'], p = propens_in_cell / total_cell_propens[r, c])
    print(event_type)

    if event_type == 'infection evolution':
        epith_grid[r, c] +=1

    elif event_type == 'cell death':
        epith_is_alive[r, c] = False
        epith_grid[r, c] = -1

    
    #local update after event
    total_propensity -= total_cell_propens[r, c]
    update_local_propens(r, c)
    total_cell_propens[r, c] = inf_evol_m[r, c] + death_m[r, c]
    total_propensity += total_cell_propens[r, c]

    steps += 1
    frames.append(epith_grid.copy())

    if not np.any(epith_is_alive):
        print('all cells are dead')
        break