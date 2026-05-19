import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


grid_size = 10
NK_ratio = 1.0
max_NK = 3
a = 3/4
b = 1/3
t_max = 50

max_steps = 4

grid = np.zeros((grid_size, grid_size))

#NK_grid initial conditions
NK_grid = np.zeros(grid.shape)
NK_grid = np.random.poisson(NK_ratio, size = (grid_size, grid_size))
NK_grid = np.where(NK_grid > 3, 3, NK_grid)

NK_moves_propens = np.where(NK_grid > 0, a / (1 + b * NK_grid), 0) #NK initial move propensities

def NK_move(r, c):
    #possible directions are different for even and odd rows
    if r % 2 == 0:
        directions = [[-1, -1], [-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]]
    else:
        directions = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]]
  
    dr, dc = directions[np.random.randint(0, 6)]
    return (r + dr) % grid_size, (c + dc) % grid_size

frames = []

for _ in range(max_steps):

    all_propens = NK_moves_propens.flatten() #np.concatenate(zdarzenia_A, zdarzenia_B)
    total_propensity = all_propens.sum()

    event_type = ['NK_move' for _ in all_propens]

    #losujemy czas i zdarzenie dla tego czasu
    dt = - np.log(np.random.random()) / total_propensity
    event_id = np.random.choice(len(all_propens), p = all_propens / total_propensity)

    if event_type[event_id] == 'NK_move':

        r, c = divmod(event_id, grid_size)
        new_r, new_c = NK_move(r, c) #współrzędne komórki która ma się ruszyć
        NK_grid[r, c] -= 1
        NK_grid[new_r, new_c] += 1


        #aktualizacja propensities -> robimy ją tylko dla tych dwóch punktów w których się coś zmieniło
        NK_moves_propens[r, c] = a / 1 + b * NK_grid[r, c] if NK_grid[r, c] else 0
        NK_moves_propens[new_r, new_c] = a / 1 + b * NK_grid[r, c]

        #print('time', dt)
        #print('old:', (r, c), 'new:', (new_r, new_c))
        #print(NK_grid)

        frames.append(NK_grid.copy())


#animation

fig, ax = plt.subplots(figsize = (6, 6))

img = ax.imshow(frames[0], cmap='Reds', vmin = 0, vmax = max_NK, origin = 'lower')
plt.colorbar(img, ax = ax, label='NK_grid value')
ax.set_title("NK movement")

def update(frame_idx):
    img.set_data(frames[frame_idx])
    return [img]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)

plt.show()