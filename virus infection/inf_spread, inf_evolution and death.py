import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

grid_size = 20
moi = 0.1
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

def hex_neighborhood(r, c):
    #possible directions are different for even and odd rows
    if r % 2 == 0:
        directions = [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]]
    else:
        directions = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]]
  
    dr, dc = directions[np.random.randint(0, 6)]

    neighbors = []
    for dr, dc in directions:
        nr, nc = (r + dr) % grid_size, (c + dc) % grid_size
        neighbors.append((nr, nc))
    return neighbors



def update_local_propens(r, c):

    #update for killed cell:
    if not epith_is_alive[r, c]:
        inf_evol_m[r, c] = 0.0
        death_m[r, c] = 0.0
        inf_spread_m[r, c] = 0.0
        return 
    
    #infection evolution -> possible only for infected cells (cells with states [1, 10]):
    if 0 < epith_grid[r, c] < 10:
        inf_evol_m[r, c] = inf_evolution_prop(a, epith_grid[r, c])
    
    else:
        inf_evol_m[r, c] = 0.0

    #infection spread -> propensity of infecting healthy cell:
    if epith_grid[r, c] == 0:
        total_spread_risk = 0
        neighbors = hex_neighborhood(r, c)
        
        for nr, nc in neighbors:
            if epith_is_alive[r, c] and epith_grid[nr, nc] >= 1: #neighboring cell must be infected in order to infect other cells
                total_spread_risk += infection_prop(epith_grid[nr, nc])
        
            inf_spread_m[r, c] = total_spread_risk

    else:
        inf_spread_m[r, c] = 0.0
        
    #death propensity:
    death_m[r, c] = death_prop(epith_grid[r, c])
        


inf_evol_m = np.zeros((grid_size, grid_size))
death_m = np.zeros((grid_size, grid_size))
inf_spread_m = np.zeros((grid_size, grid_size)) #propensities of getting infected


for r in range (grid_size):
    for c in range(grid_size):
        update_local_propens(r, c)

#SIMULATION
frames = []
times = []
time = 0
steps = 0
max_steps = 10000


total_cell_propens = inf_evol_m + death_m + inf_spread_m
total_propensity = total_cell_propens.sum()

while steps < max_steps:

    #infection_evolution, death_propens
    if total_propensity <= 1e-12: 
        print('no possible events')
        break

    dt = - np.log(np.random.random()) / total_propensity
    time += dt
    times.append(time)

    flat_propens = total_cell_propens.flatten()

    #randomly choosing cell for an event
    cell_id = np.random.choice(len(flat_propens), p = flat_propens / total_propensity)
    r, c = divmod(cell_id, grid_size)

    cells_to_update = [(r, c)]
    cells_to_update.extend(hex_neighborhood(r, c))

    #choosing event for that cell
    propens_in_cell = np.array((inf_evol_m[r, c], death_m[r, c], inf_spread_m[r, c]))
    event_type = np.random.choice(['infection evolution', 'cell death', 'getting infected'], p = propens_in_cell / total_cell_propens[r, c])
    print(f'Step: {steps}, cell: {r, c}, event: {event_type}')

    if event_type == 'infection evolution':
        epith_grid[r, c] +=1

    elif event_type == 'cell death':
        epith_is_alive[r, c] = False
        epith_grid[r, c] = -1

    elif event_type == 'getting infected':
        epith_grid[r, c] = 1

    
    #local update after event
    for row, col in set(cells_to_update):
        total_propensity -= total_cell_propens[row, col]
        update_local_propens(row, col)
        total_cell_propens[row, col] = inf_evol_m[row, col] + death_m[row, col] + inf_spread_m[row, col]
        total_propensity += total_cell_propens[row, col]

    steps += 1
    frames.append(epith_grid.copy())

    if not np.any(epith_is_alive):
        print('all cells are dead')
        break


#creating cmap for the simulation
viridis = cm.get_cmap('Reds', 256)
newcolors = viridis(np.linspace(0, 1, 256))
death_color = np.array([128/256, 128/256, 128/256, 1])
newcolors[:15, :] = death_color
newcmp = ListedColormap(newcolors)



#ANIMATION
def play_animation():
    rows, cols = grid_size, grid_size
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    X = X.astype(float)
    X[1::2] += 0.5

    X_pts = X.ravel()
    Y_pts = Y.ravel()

    fig, ax = plt.subplots(figsize=(8, 8))
    point_size = 100000 / (grid_size ** 2)
    scat = ax.scatter(X_pts, Y_pts, c = frames[0].ravel(), s = point_size, cmap = newcmp, vmin = -1, vmax = 10, edgecolors = 'k')
    #X_pts, Y_pts - constant positions
    #c - initial color, s - circle size

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, grid_size)
    ax.set_ylim(-0.5, grid_size - 0.5)

    plt.colorbar(scat, ax = ax, label = 'epithelial cell state', ticks = range(-1, 11))

    

    def update(frame_idx):
        new_data = frames[frame_idx].ravel()
        scat.set_array(new_data)
        ax.set_title(f'Epithelial cells only; time: {times[frame_idx]:.3f}')
        return [scat]

    plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

    ani = animation.FuncAnimation(fig, update, frames = len(frames), interval = 50)
    #ani.save(f'ver2_virus_evolution_without_NK_{grid_size}x{grid_size}.mp4', writer = 'ffmpeg', fps = 75, dpi = 100, extra_args=['-preset', 'ultrafast'])

    print('Animation saved')

    plt.tight_layout()
    plt.show()

play_animation()