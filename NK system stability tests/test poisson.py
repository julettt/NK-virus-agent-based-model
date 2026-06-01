import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.stats import poisson

grid_size = 100
NK_ratio = 1.0
max_NK = 3
a = 3/4
b = 3
t_max = 3
steps_max = 100000

max_n = 5

grid = np.zeros((grid_size, grid_size))

def init_NK_grid(NK_ratio):
    k_values = np.arange(max_NK + 1)
    probs = poisson.pmf(k_values, mu = NK_ratio)
    probs /= probs.sum()
    NK_grid = np.random.choice(k_values, size = (grid_size, grid_size), p = probs)
    return NK_grid

def NK_move_prop(a, b, NK):
    return a * NK / (1 + b * np.max(NK - 1, 0))


def run_test(b):

    NK_grid = init_NK_grid(NK_ratio)

    NK_move_propens = np.where(NK_grid > 0, NK_move_prop(a, b, NK_grid), 0) #NK initial move propensities

    time = 0
    steps = 0

    history_data = []

    while steps < steps_max:

        all_propens = NK_move_propens.flatten()
        total_propensity = all_propens.sum()

        if total_propensity == 0: break

        event_type = ['NK_move' for _ in all_propens]


        #randomly choosing time and an event for that time
        dt = - np.log(np.random.random()) / total_propensity
        event_id = np.random.choice(len(all_propens), p = all_propens / total_propensity)
        
        time += dt
        

        if event_type[event_id] == 'NK_move':

            r, c = divmod(event_id, grid_size)

            if r % 2 == 0:
                directions = [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]]
            else:
                directions = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]]
            
            neighbors = [((r + dr) % grid_size, (c + dc) % grid_size) for dr, dc in directions]
            valid_neighbors = [(nr, nc) for nr, nc in neighbors if NK_grid[nr, nc] < max_NK]

            if valid_neighbors:
                idx = np.random.randint(0, len(valid_neighbors))
                new_r, new_c = valid_neighbors[idx]

                NK_grid[r, c] -= 1
                NK_grid[new_r, new_c] += 1

                NK_move_propens[r, c] = NK_move_prop(a, b, NK_grid[r, c]) if NK_grid[r, c] else 0
                NK_move_propens[new_r, new_c] = NK_move_prop(a, b, NK_grid[new_r, new_c])

            else:
                pass


            #local propensities update        
            NK_move_propens[r, c] = NK_move_prop(a, b, NK_grid[r, c]) if NK_grid[r, c] else 0
            NK_move_propens[new_r, new_c] = NK_move_prop(a, b, NK_grid[new_r, new_c])


            if steps % 250 == 0 or steps == 0:

                history_data.append({'time': time, 
                                     'grid_0': np.sum(NK_grid == 0),
                                     'grid_1': np.sum(NK_grid == 1),
                                     'grid_2': np.sum(NK_grid == 2),
                                     'grid_3': np.sum(NK_grid == 3),
                                     'grid_4': np.sum(NK_grid == 4),
                                     'grid_5': np.sum(NK_grid == 5)
                                     
                                     })
            if steps % 1000 == 0:
                print(steps)
                
        steps += 1

    return history_data


b_values = [0, 1, 3, 5, 20, 100]

for b in b_values:

    n = 0
    all_starts = []
    all_ends = []


    while n < max_n:

        print(f'start test nr. {n}')

        history = run_test(b)
        
        print(f'completed')

        df = pd.DataFrame(history)

        start_state = df.iloc[[0]].drop(columns = ['time'], errors = 'ignore')
        end_state = df.iloc[[-1]].drop(columns = ['time'], errors = 'ignore')

        all_starts.append(start_state)
        all_ends.append(end_state)

        n += 1

    mean_start = pd.concat(all_starts).mean()
    mean_end = pd.concat(all_ends).mean()


    filename = f'b={b}.txt'

    with open(filename, 'w', encoding = 'utf-8') as f:

        print('------', file = f)
        print(f'results for max_NK = {max_NK}, time = {t_max}, grid_size = {grid_size}, b = {b}', file = f)

        print('grid_n - number of grid nodes with n NK cells', file = f)
        print(f'averaged over n = {n} realizations', file = f)

        print('------', file = f)

        print('mean start state:', file = f)
        print(mean_start.to_string(), file = f)

        print('------', file = f)

        print('mean end state:', file = f)
        print(mean_end.to_string(), file = f)


    print('finished')



