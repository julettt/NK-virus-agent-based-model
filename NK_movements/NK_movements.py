import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as tri

grid_size = 100
NK_ratio = 1.0
max_NK = 3
a = 3/4
b = 3
t_max = 50

test_mode = True

grid = np.zeros((grid_size, grid_size))

#NK_grid initial conditions
NK_grid = np.zeros(grid.shape)
NK_grid = np.random.poisson(NK_ratio, size = (grid_size, grid_size))
NK_grid = np.where(NK_grid > 3, 3, NK_grid)

# -- test mode --
if test_mode == True:
    init_3 = np.sum(np.where(NK_grid == 3, 1, 0))
    init_2 = np.sum(np.where(NK_grid == 2, 1, 0))
    init_1 = np.sum(np.where(NK_grid == 1, 1, 0))
    init_0 = np.sum(np.where(NK_grid == 0, 1, 0))



def NK_move_prop(a, b, NK):
    return a * NK / (1 + b * NK)


NK_move_propens = np.where(NK_grid > 0, NK_move_prop(a, b, NK_grid), 0) #NK initial move propensities

def NK_move(r, c):
    #possible directions are different for even and odd rows
    if r % 2 == 0:
        directions = [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]]
    else:
        directions = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]]
  
    dr, dc = directions[np.random.randint(0, 6)]
    return (r + dr) % grid_size, (c + dc) % grid_size

frames = []
time = 0
steps = 0

while time < t_max:

    all_propens = NK_move_propens.flatten() #np.concatenate(zdarzenia_A, zdarzenia_B)
    total_propensity = all_propens.sum()

    if total_propensity == 0: break

    event_type = ['NK_move' for _ in all_propens]


    #randomly choosing time and an event for that time
    dt = - np.log(np.random.random()) / total_propensity
    event_id = np.random.choice(len(all_propens), p = all_propens / total_propensity)
    
    time += dt
    steps += 1
    print(time)
    

    if event_type[event_id] == 'NK_move':

        r, c = divmod(event_id, grid_size)
        new_r, new_c = NK_move(r, c) #coords of the NK cell that we want to move
        
        move = False
        while move == False:
            if NK_grid[new_r, new_c] <= 2:
                NK_grid[r, c] -= 1
                NK_grid[new_r, new_c] += 1
                move = True
            else:
                new_r, new_c = NK_move(r, c)


        #local propensities update        
        NK_move_propens[r, c] = NK_move_prop(a, b, NK_grid[r, c]) if NK_grid[r, c] else 0
        NK_move_propens[new_r, new_c] = NK_move_prop(a, b, NK_grid[new_r, new_c])

        if steps % 100 == 0:
            frames.append(NK_grid.copy())


# -- test mode --
if test_mode == True:
    new_3 = np.sum(np.where(NK_grid == 3, 1, 0))
    new_2 = np.sum(np.where(NK_grid == 2, 1, 0))
    new_1 = np.sum(np.where(NK_grid == 1, 1, 0))
    new_0 = np.sum(np.where(NK_grid == 0, 1, 0))
            
    print('-------------')
    print(f'completed for b = {b}')
    print('total time: ', time, 'steps: ', {steps})
    print(f'initial: 0: {init_0}, 1: {init_1}, 2: {init_2}, 3: {init_3}. Total: {init_0 + init_1 + init_2 + init_3}')
    print(f'after simulation: 0: {new_0}, 1: {new_1}, 2: {new_2}, 3: {new_3}')
    print(f'difference (new - old) 0: {new_0 - init_0}, 1: {new_1 - init_1}, 2: {new_2 - init_2}, 3: {new_3 - init_3}. Total = {new_0 + new_1 + new_2 + new_3}')



#ANIMATION
rows, cols = grid_size, grid_size
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

X = X.astype(float)
X[1::2] += 0.5

X_pts = X.ravel()
Y_pts = Y.ravel()

fig, ax = plt.subplots(figsize=(10, 10))
point_size = 100000 / (grid_size ** 2)
scat = ax.scatter(X_pts, Y_pts, c = frames[0].ravel(), s = point_size, cmap = 'Reds', vmin = 0, vmax = max_NK, edgecolors = 'k')
#X_pts, Y_pts - constant positions
#c - initial color, s - circle size

ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.5, grid_size)
ax.set_ylim(-0.5, grid_size - 0.5)

plt.colorbar(scat, ax = ax, label = 'NK cells number', ticks = range(max_NK + 1))
ax.set_title('NK movements')

def update(frame_idx):
    new_data = frames[frame_idx].ravel()
    scat.set_array(new_data)
    return [scat]

plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

ani = animation.FuncAnimation(fig, update, frames = len(frames), interval = 200)
ani.save(f'ver2_NK_movements_{grid_size}x{grid_size}.mp4', writer = 'ffmpeg', fps = 10, dpi = 100, extra_args=['-preset', 'ultrafast'])

plt.tight_layout()
plt.show()