import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap


#----- animation settings -----
INPUT_FILE = "plot results/with NK, NK_delay=4.0.npz"
SAVE_VIDEO = True
OUTPUT_VIDEO = "active virus with NK.mp4"
SHOW_NK = False


data = np.load(INPUT_FILE)

frames_epith = data["frames_epith"]
frames_NK = data["frames_NK"]
stats_array = data["stats_array"]

times = stats_array[:, 3]
grid_r = int(data["grid_r"])
grid_c = int(data["grid_c"])
max_inf_state = int(data["max_inf_state"])


#cmap for epithelial grid
reds = cm.get_cmap('Reds', 256)
newcolors = reds(np.linspace(0, 1, 256))
death_color = np.array([128/256, 128/256, 128/256, 1])
n_death_colors = int(256 / (max_inf_state + 2))
newcolors[:n_death_colors, :] = death_color
epith_cmap = ListedColormap(newcolors)


x = np.arange(grid_c) #cols
y = np.arange(grid_r) #rows
X, Y = np.meshgrid(x, y)

X = X.astype(float)
X[1::2] += 0.5

X_pts = X.ravel()
Y_pts = Y.ravel()


#----- animation -----
def play_animation():

    if SHOW_NK:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        ax_epith, ax_NK = axes
    else:
        fig, ax_epith = plt.subplots(figsize=(8,8))

    point_size = 100000 / (grid_r * grid_c)

    scat_epith = ax_epith.scatter(X_pts, Y_pts, c=frames_epith[0].ravel(), s=point_size,
                                   cmap=epith_cmap, vmin=-1, vmax=max_inf_state, edgecolors='k', linewidths=0.3)
    ax_epith.set_aspect('equal')
    ax_epith.set_xticks([])
    ax_epith.set_yticks([])
    ax_epith.set_xlim(-0.5, grid_c)
    ax_epith.set_ylim(-0.5, grid_r - 0.5)
    #plt.colorbar(scat_epith, ax=ax_epith, label='epithelial cell state', ticks=range(-1, max_inf_state + 1, max(1, max_inf_state // 10)))

    if SHOW_NK:
        max_NK = int(data["max_NK"])
        scat_NK = ax_NK.scatter(X_pts, Y_pts, c=frames_NK[0].ravel(), s=point_size,
                                 cmap='Blues', vmin=0, vmax=max_NK, edgecolors='k', linewidths=0.3)
        ax_NK.set_aspect('equal')
        ax_NK.set_xticks([])
        ax_NK.set_yticks([])
        ax_NK.set_xlim(-0.5, grid_c)
        ax_NK.set_ylim(-0.5, grid_r - 0.5)
        plt.colorbar(scat_NK, ax=ax_NK, label='NK cell count', ticks=range(0, max_NK + 1))

    def update(frame_idx):
        scat_epith.set_array(frames_epith[frame_idx].ravel())
        ax_epith.set_title(f'Epithelial cells; time = {times[frame_idx]:.3f}')

        artists = [scat_epith]

        if SHOW_NK:
            scat_NK.set_array(frames_NK[frame_idx].ravel())
            ax_NK.set_title(f'NK cells; time = {times[frame_idx]:.3f}')
            artists.append(scat_NK)

        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(frames_epith), interval=1, blit=False)

    if SAVE_VIDEO:
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
        ani.save(OUTPUT_VIDEO, writer='ffmpeg', fps=5, dpi=400, extra_args=['-preset', 'ultrafast'])
        print(f'Animation saved to {OUTPUT_VIDEO}')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    play_animation()