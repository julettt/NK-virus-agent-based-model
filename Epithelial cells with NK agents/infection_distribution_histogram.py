import numpy as np
import glob
import os
import matplotlib.pyplot as plt

results = {}

for path in sorted(glob.glob("plot results/*npz")):
    data = np.load(path)
    max_inf_state = int(data["max_inf_state"])
    tau_dep = int(data["t_dep"])
    M_I_spread = int(data["M_I_spread"])
    M_I_death = int(data["M_I_death"])
    gamma_dep = float(data["gamma_dep"])

    label = os.path.basename(path).removesuffix(".npz")
    results[label] = {
        "times": data["stats_array"][:, 3],
        "inf_hist": data["stats_array"][:, 4:4 + max_inf_state + 1],
        "dead_inf_hist": data["stats_array"][:, 4 + max_inf_state + 1:],
        "M_I_spread": float(data["M_I_spread"]),
        "M_I_death": float(data["M_I_death"]),
        "MOI": float(data["MOI"]),
    }


PLOT_TIMES = [24.0, 48.0]
with_dead = True

bins = np.arange(1, int(max_inf_state) + 1)
labels = list(results.keys())
print(labels)
n_bars = len(labels)

fig, axes = plt.subplots(nrows = 1, ncols = len(PLOT_TIMES), sharey = True, sharex = True, figsize = (4 * len(PLOT_TIMES), 4))

for ax, t_target in zip(axes, PLOT_TIMES):

    width = 0.8 / n_bars
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_bars)

    for offset, label in zip(offsets, labels):
        r = results[label]

        idx = np.argmin(np.abs(r["times"] - t_target))
        alive_hist = r["inf_hist"][idx, 1 : max_inf_state + 1]
        dead_hist = r["dead_inf_hist"][idx, 1 : max_inf_state + 1]

        if with_dead:
            alive_pct = alive_hist
            dead_pct = dead_hist

            bars = ax.bar(bins + offset, alive_pct, width = width, label = label, alpha= 0.85)
            color = bars[0].get_facecolor()
            ax.bar(bins + offset, dead_pct, width = width, color = color, alpha = 0.35,
            bottom=alive_pct)

        else:
            alive_pct = alive_hist

            bars = ax.bar(bins + offset, alive_pct, width = width, label = label, alpha= 0.85)
            color = bars[0].get_facecolor()


    ax.set(xlabel="Infection state (S)", title=f"t = {t_target} h", ylim = [0, 2000])
    ax.grid(axis="y", alpha=0.3)


axes[0].set_ylabel("Number of cells")
axes[0].legend()
plt.suptitle(f"Infection distribution, gamma_dep = {gamma_dep}, max_inf_state={max_inf_state}.")
plt.tight_layout()
#plt.savefig(f"testtest_M_I={M_I}_gamma_dep_t_dep.png")
plt.show()