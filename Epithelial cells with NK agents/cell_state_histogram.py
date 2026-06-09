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
        "healthy": data["stats_array"][:, 0],
        "infected": data["stats_array"][:, 1],
        "dead": data["stats_array"][:, 2],
        "inf_hist": data["stats_array"][:, 4:],
        "MOI": float(data["MOI"]),
    }


PLOT_TIMES = [24.0, 48.0, 72.0]
bins = np.arange(1, 11)
labels = list(results.keys())
print(labels)
n_moi = len(labels)

fig, axes = plt.subplots(nrows = 1, ncols = len(PLOT_TIMES), sharey = "row", figsize = (4 * len(PLOT_TIMES), 4))

cell_types = ["healthy", "infected", "dead"]
x = np.arange(n_moi)
width = 0.8 / (n_moi * 3)
offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, len(cell_types))

for col, t_target in enumerate(PLOT_TIMES):
    ax = axes[col]
    for offset, cell_type in zip(offsets, cell_types):
        vals = []
        for label in labels:
            r = results[label]
            idx = np.argmin(np.abs(r["times"] - t_target))
            vals.append(r[cell_type][idx])

        ax.bar(x + offset, vals, width=width, label=cell_type, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([l.split("MOI=")[1] for l in labels])
    ax.set(xlabel="MOI", title=f"t = {t_target} h")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Number of cells")
axes[0].legend(fontsize=8)

plt.suptitle("Cell counts by time point (without NK)")
plt.tight_layout()
plt.savefig("cell_count_without_NK.png")
plt.show()