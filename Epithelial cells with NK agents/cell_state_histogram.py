import numpy as np
import glob
import os
import matplotlib.pyplot as plt

results = {}

for path in sorted(glob.glob("plot results/*npz")):
    data = np.load(path)
    max_inf_state = int(data["max_inf_state"])
    tau_dep = int(data["t_dep"])
    M_I = int(data["M_I"])

    label = os.path.basename(path).removesuffix(".npz")
    results[label] = {
        "times":    data["stats_array"][:, 3],
        "healthy":  data["stats_array"][:, 0],
        "infected": data["stats_array"][:, 1],
        "dead":     data["stats_array"][:, 2],
        "inf_hist":      data["stats_array"][:, 4 : 4 + max_inf_state + 1],
        "dead_inf_hist": data["stats_array"][:, 4 + max_inf_state + 1 :],
        "MOI": float(data["MOI"]),
        "gamma_dep": float(data["gamma_dep"]),
    }


with_nk = {k: v for k, v in results.items() if k.startswith("with NK")}
without_nk = {k: v for k, v in results.items() if k.startswith("without NK")}

def moi_key(label):
    return float(label.split("gamma_dep=")[1])

moi_labels_nk = sorted(with_nk.keys(), key=moi_key)
moi_labels_no_nk = sorted(without_nk.keys(), key=moi_key)
moi_values = [moi_key(l) for l in moi_labels_nk]
n_moi = len(moi_values)


PLOT_TIMES = [0, 24.0, 48.0]
cell_types = ["healthy", "infected", "dead"]
cell_colors = {"healthy": "tab:green", "infected": "tab:orange", "dead": "tab:red"}

fig, axes = plt.subplots(nrows=1, ncols=len(PLOT_TIMES), sharey="row", figsize=(4 * len(PLOT_TIMES), 4))

n_pairs = len(cell_types)
pair_gap = 0.05
bar_width = (0.8 - pair_gap * (n_pairs - 1)) / (n_pairs * 2)
pair_centers = np.linspace(-(0.8 / 2) + bar_width, (0.8 / 2) - bar_width, n_pairs)

x = np.arange(n_moi)

for col, t_target in enumerate(PLOT_TIMES):
    ax = axes[col]

    for pi, cell_type in enumerate(cell_types):
        center = pair_centers[pi]
        color = cell_colors[cell_type]

        vals_no_nk = []
        for label in moi_labels_no_nk:
            r = without_nk[label]
            idx = np.argmin(np.abs(r["times"] - t_target))
            vals_no_nk.append(r[cell_type][idx])

        vals_nk = []
        for label in moi_labels_nk:
            r = with_nk[label]
            idx = np.argmin(np.abs(r["times"] - t_target))
            vals_nk.append(r[cell_type][idx])

        ax.bar(x + center + bar_width / 2, vals_nk, width=bar_width, color=color, alpha=0.85,
            label=f"{cell_type} - with NK" if col == 0 else "_nolegend_")
        
        ax.bar(x + center - bar_width / 2, vals_no_nk, width=bar_width, color=color, alpha=0.35,
            label=f"{cell_type} - without NK" if col == 0 else "_nolegend_")

        #DORZUCIĆ WERSJĘ Z NO_NK JAK BĘDZIE POTRZEBA

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in moi_values])
    ax.set(xlabel="gamma_dep", title=f"t = {t_target} h", ylim=[0, 10000])
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Number of cells")
axes[0].legend(fontsize=8, ncol=2)


plt.suptitle(f"Cell counts by time point with NK cells, M_I = {M_I}, without death independent")
plt.tight_layout()
plt.savefig(f"test_gamma_dep=0.png")
plt.show()