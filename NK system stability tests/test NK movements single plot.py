import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.stats import poisson

grid_size = 100
NK_ratio = 1.0
max_NK = 3
a = 3/4
test_b = [0]
t_max = 30

grid = np.zeros((grid_size, grid_size))

def init_NK_grid(NK_ratio):
    k_values = np.arange(max_NK + 1)
    probs = poisson.pmf(k_values, mu = NK_ratio)
    probs /= probs.sum()
    NK_grid = np.random.choice(k_values, size = (grid_size, grid_size), p = probs)
    return NK_grid


def get_directions(r):
    
    if r % 2 == 0:
        return np.array([[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]])
    else:
        return np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]])
    

def NK_move_prop(r, c, NK_grid):
    
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

    NK_move_propens = np.zeros((grid_size, grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            NK_move_propens[r, c] = NK_move_prop(r, c, NK_grid)


    time = 0
    steps = 0

    history_data = []

    while time < t_max:

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
            NK_move_propens[ur, uc] = NK_move_prop(ur, uc, NK_grid)


        if steps % 250 == 0 or steps == 0:

            history_data.append({'time': time, 
                                'grid_0': np.sum(NK_grid == 0),
                                'grid_1': np.sum(NK_grid == 1),
                                'grid_2': np.sum(NK_grid == 2),
                                'grid_3': np.sum(NK_grid == 3)                                     
                                })
            
            if steps % 1000 == 0:
                print(steps)
                
        steps += 1

    return history_data


all_df = {}

for b in test_b:

    print(f'start for b = {b}')

    history = run_test(b)
    
    print(f'completed')

    df = pd.DataFrame(history)
    df.set_index('time', inplace = True)

    all_df[f'b = {b}'] = df

big_df = pd.concat(all_df, axis = 0)

fig, axes = plt.subplots(1, 1, figsize = (10, 6), sharex = True, sharey = True)
axes = [axes]
colors = ['C0', 'C1', 'C2', 'C3']

for i, b in enumerate(test_b):
    
    ax = axes[i]
    key = f'b = {b}'

    df_single = big_df.xs(key, level = 0)
    
    df_single.index = pd.to_timedelta(df_single.index, unit = 'h')

    #resampling to get better plots
    df_resampled = df_single.resample('15min').mean().interpolate(method = 'linear').bfill()

    df_ma = df_resampled.rolling(window = '3h', center = True).mean()

    hours_resampled = df_resampled.index.total_seconds().to_numpy() / 3600
    hours_ma = df_ma.index.total_seconds().to_numpy() / 3600

    df_ma.index = df_ma.index.total_seconds() / 3600
    df_resampled.index = df_resampled.index.total_seconds() / 3600


    #plotting

    labels = ['0 NK/node', '1 NK/node', '2 NK/node', '3 NK/node']

    for col_idx, col_name in enumerate(df_resampled.columns):

        ax.plot(hours_resampled, df_resampled[col_name], color = colors[col_idx], alpha = 0.3, linestyle = '--')
        ax.plot(hours_ma, df_ma[col_name], color = colors[col_idx], label = labels[col_idx])

        ax.set_title(f'b = {b}', fontsize = 12)
        ax.grid(alpha = 0.3)


    if i >= 0: ax.set_xlabel('Time [h]')
    if i % 2 == 0: ax.set_ylabel('Number of nodes')


handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(handles, legend_labels, loc = 'lower center', bbox_to_anchor = (0.5, -0.005),
           ncol = 4, fontsize = 12, frameon = True, facecolor = 'white')

plt.suptitle('Distribution of NK cells on the grid depending on b value', fontsize = 16)
fig.subplots_adjust(bottom=0.18)

filename = f'NK distribution after simulation for single 50x50 grid, t_max = 30, b = 0.png'
plt.savefig(filename, dpi = 300)
print('finished')
plt.show()


