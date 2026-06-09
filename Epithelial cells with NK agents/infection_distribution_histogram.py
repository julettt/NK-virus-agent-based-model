import numpy as np
import glob
import os
import matplotlib.pyplot as plt

results = {}

for path in sorted(glob.glob("plot results/*npz")):
    data = np.load(path)
    label = os.path.basename(path).removesuffix(".npz")
    results[label] = {
        "times": data["stats_array"][:, 3],
        "inf_hist": data["stats_array"][:, 4:],
        "MOI": float(data["MOI"]),
    }


PLOT_TIMES = [24.0, 48.0, 72.0]
bins = np.arange(1, 11)
labels = list(results.keys())
print(labels)
n_moi = len(labels)

fig, axes = plt.subplots(nrows = 1, ncols = len(PLOT_TIMES), sharey = True, sharex = True, figsize = (4 * len(PLOT_TIMES), 4))

for ax, t_target in zip(axes, PLOT_TIMES):

    width = 0.8 / n_moi
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_moi)

    for offset, label in zip(offsets, labels):
        r = results[label]

        idx = np.argmin(np.abs(r["times"] - t_target))
        hist = r["inf_hist"][idx, 1:] / r["inf_hist"][idx, 1:].sum() * 100

        ax.bar(bins + offset, hist, width=width, label=label, alpha=0.85)

    ax.set(xlabel="Infection state (S)", title=f"t = {t_target} h")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("S_i / all_infected_cells [%]")
axes[0].legend()
plt.suptitle("Infection distribution (without NK), M_I = 7")
plt.tight_layout()
plt.savefig("without_NK_percent_inf_distr_hist.png")
plt.show()