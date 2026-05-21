import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as tri


grid_size = 10
moi = 0.1
t_max = 1

#epithelial cells:
#0 -> not infected
# 1-10 -> infected

grid = np.zeros((grid_size, grid_size))

epith_grid = np.random.poisson(moi, size = grid.shape)
epith_grid = np.where(epith_grid > 1, 1, epith_grid)

#epith_is_alive = np.full_like(epith_grid, True, dtype = 'bool') -> tak ma być 

epith_is_alive = np.random.randint(0, 2, size = grid.shape) # -> losowo do testów, 1 - alive, 0 - death

def death_prop(epith, x = 300):
    return (1 + epith) / x


death_propens = death_prop(epith_grid)

#SIMULATION
frames = []
time = 0
steps = 0
max_steps = 300

while steps < max_steps:

    print(time)

    all_propens = death_propens.flatten() #np.concatenate(zdarzenia_A, zdarzenia_B)
    total_propensity = all_propens.sum()

    if total_propensity == 0: break

    event_type = ['epith death' for _ in all_propens]

    #randomly choosing time for an event
    dt = - np.log(np.random.random()) / total_propensity

    #and choosing an event -> we also have to check if the event is possible, otherwise we have to choose another one
    while True:
        event_id = np.random.choice(len(all_propens), p = all_propens / total_propensity)

        if event_type[event_id] == 'epith death':
            
            r, c = divmod(event_id, grid_size)

            if epith_is_alive[r, c]:

                #event execution
                epith_is_alive[r, c] = 0
                time += dt
                steps += 1

                frames.append(epith_is_alive.copy())

                break

                
            if np.sum(epith_is_alive) == 0:
                print('all cells dead!')
                steps = max_steps
                break



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

    fig, ax = plt.subplots(figsize=(10, 10))
    point_size = 100000 / (grid_size ** 2)
    scat = ax.scatter(X_pts, Y_pts, c = frames[0].ravel(), s = point_size, cmap = 'Reds', vmin = 0, vmax = 1, edgecolors = 'k')
    #X_pts, Y_pts - constant positions
    #c - initial color, s - circle size

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, grid_size)
    ax.set_ylim(-0.5, grid_size - 0.5)

    ax.set_title('Epithelial cells death')

    def update(frame_idx):
        new_data = frames[frame_idx].ravel()
        scat.set_array(new_data)
        return [scat]

    plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

    ani = animation.FuncAnimation(fig, update, frames = len(frames), interval = 200)
    #ani.save(f'cell_deaths_{grid_size}x{grid_size}.mp4', writer = 'ffmpeg', fps = 10, dpi = 100, extra_args=['-preset', 'ultrafast'])

    plt.tight_layout()
    plt.show()

play_animation()